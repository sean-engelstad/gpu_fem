#include <iostream>

#include "_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "mesh/vtk_writer.h"
#include "shell/shell.h"

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    using T = double;

    std::ios::sync_with_stdio(false);  // always flush print immediately

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 300;  // 100
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    double fillin = 10.0;  // 10.0
    bool print = true;
    assembler.symbolic_factorization(fillin, print);

    // get the loads
    double Q = 100.0;  // load magnitude
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads, true);

    // setup kmat, res, variables
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    printf("Begin newton iterations\n");
    // demo Newton solve loop
    for (int inewton = 0; inewton < 10; inewton++) {
        // set new U or vars for Kmat computation
        assembler.set_variables(vars);

        // reset residual and kmat, soln
        kmat.zeroValues();
        res.zeroValues();
        soln.zeroValues();

        // now assemble new kmat, res (adding into previous)
        auto start = std::chrono::high_resolution_clock::now();
        assembler.add_jacobian(res, kmat, print);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        assembler.apply_bcs(res, print);
        assembler.apply_bcs(kmat, print);

        // add in -loads to it, need to add this routine
        res.axpy(-1.0, loads);
        res.scale(-1.0);  // res = F  - Fint(u) = rhs
        // then K(u0) * du = rhs = -res(u0)

        // report initial residual norm?
        // TODO : need code to get this..
        double nrm_R;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, res.getSize(), res.getPtr(), 1, &nrm_R))
        printf("\tnewton step %d, res = %.4e\n", inewton, nrm_R);

        // do new linear solve
        CUSPARSE::direct_LU_solve_old<T>(kmat, res, soln, true);
    }

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "plate.vtk");

    delete[] my_loads;

    return 0;
};

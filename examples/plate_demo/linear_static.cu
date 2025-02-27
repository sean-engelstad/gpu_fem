#include "_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"
#include <iostream>
#include "mesh/vtk_writer.h"

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    using T = double;

    std::ios::sync_with_stdio(false); // always flush print immediately

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 300; // 100
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    double fillin = 10.0; // 10.0
    bool print = true;
    assembler.symbolic_factorization(fillin, print);

    // init variables u;
    auto vars = assembler.createVarsVec();
    assembler.set_variables(vars);

    // setup matrix & vecs
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat, print);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    assembler.apply_bcs(res, print);
    assembler.apply_bcs(kmat, print);

    // set the rhs for this problem
    double Q = 1.0; // load magnitude
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);

    // it's currently taking a really long time to make 
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads, true);
    
    // now do cusparse solve on linear static analysis
    CUSPARSE::direct_LU_solve_old<T>(kmat, loads, soln, true);
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");
    
    
    delete[] my_loads;

    return 0;
};

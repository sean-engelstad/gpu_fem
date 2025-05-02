#include "../../examples/plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "../test_commons.h"
#include <cassert>
#include <string>


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

/*
compare AMD reordered direct LU and GMRES solves on the plate case
*/

int main(int argc, char* argv[]) {
    // prelim command line inputs
    // --------------------------

    bool print = true;
    int nxe = 20;

    // ----------------------------------

    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bsr_data.AMD_reordering();

    // make copy of original bsr_data sparsity
    auto bsr_data2 = bsr_data.createDeviceBsrData().createHostBsrData();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // first solve with LU
    // -------------------

    // make full LU kmat
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // full LU solve
    CUSPARSE::direct_LU_solve<T>(kmat, loads, soln);

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    if (print) printf("resid_norm of direct LU = %.4e\n", resid_norm);

    // GMRES solve
    // -----------

    // make new kmat with ILU(k) sparsity pattern
    bsr_data2.compute_ILUk_pattern(3, fillin);
    auto d_bsr_dat2 = bsr_data2.createDeviceBsrData();
    assembler.setBsrData(d_bsr_dat2);
    auto kmat2 = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln2 = assembler.createVarsVec();

    // assemble the kmat with ILU(k) fillin
    assembler.add_jacobian(res, kmat2);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat2);

    // solve the linear system
    int n_iter = 100, max_iter = 200;
    T abs_tol = 1e-7, rel_tol = 1e-8;
    constexpr bool use_precond = true, debug = false;
    CUSPARSE::GMRES_solve<T, use_precond, debug>(kmat2, loads, soln2, n_iter, max_iter, abs_tol, rel_tol);

    // check residual of soln2
    assembler.set_variables(soln2);
    assembler.add_residual(res); // internal residual
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm2 = CUBLAS::get_vec_norm(rhs);
    if (print) printf("resid_norm of GMRES = %.4e\n", resid_norm2);

    // compare the two solutions
    // -------------------------

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate_LU.vtk");

    // print some of the data of host residual
    auto h_soln2 = soln2.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln2, "plate_GMRES.vtk");

    // also compute relative error and report test result
    double my_rel_err = rel_err(h_soln, h_soln2);  
    bool passed = abs(resid_norm) < 1e-6;
    printTestReport("AMD reordered plate linear solve - LU vs GMRES comparison", 
        passed, resid_norm);
};
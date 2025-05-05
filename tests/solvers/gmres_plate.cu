#include "../../examples/plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "../test_commons.h"
#include <cassert>
#include <string>
#include <list>

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

void test_GMRES_plate(std::string ordering, std::string fill_type, bool print = false, int nxe = 10) {

    int rcm_iters = 5;
    double p_factor = 1.0;
    int k = 5; // for ILU(k)
    // int nxe = 5;
    double fillin = 10.0;

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

    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, print);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    int n_iter = 200, max_iter = 400;
    T abs_tol = 1e-14, rel_tol = 1e-15;
    constexpr bool use_precond = true;
    bool debug = false; // print
    CUSPARSE::GMRES_solve<T, use_precond>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol,
        print, debug);

    // printf("debug running direct LU solve\n");
    // CUSPARSE::direct_LU_solve(kmat, loads, soln);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    if (print) printf("true resid_norm = %.4e\n", resid_norm);

    // report test result ------------------
    std::string testName = "GMRES plate solve, with ";

    testName += ordering;  // always include the ordering name

    if (fill_type == "ILUk") {
        testName += " ILU(" + std::to_string(k) + ")";
    } else if (fill_type == "nofill") {
        testName += " nofill";
    } else if (fill_type == "LU") {
        testName += " LU";
    }

    // need this test to eventually be more accurate than that..
    bool passed = abs(resid_norm) < 1e-4;
    printTestReport(testName, passed, resid_norm);
}

int main(int argc, char* argv[]) {
    // turn off test all for debugging
    bool test_all = false;

    // NOTE : the none, ILU(5) test may intermittently fail
    // also the preconditioned residual M^-1*(Ax-b) is about 6 orders of mag higher than 
    // the actual residual Ax-b so you need to converge to like 1e-15 the precond resid
    // to get good actual resid and this requires almost 200 iterations of GMRES for none reordering, less for others..

    int nxe = 20;
    bool print = false;
    if (test_all) {
        std::list<std::string> list1 = {"none", "RCM", "AMD", "qorder"};
        // don't expect nofill to actually solve it
        std::list<std::string> list2 = {"ILUk", "LU"};

        for (auto it2 = list2.begin(); it2 != list2.end(); ++it2) {
            for (auto it1 = list1.begin(); it1 != list1.end(); ++it1) {
                test_GMRES_plate(*it1, *it2, print, nxe);
            }
        }
    } else {
        // debug failing tests
        print = true;
        // test_GMRES_plate("none", "ILUk", print, nxe);
        test_GMRES_plate("qorder", "ILUk", print, nxe);
        // test_GMRES_plate("RCM", "LU", print, nxe);
    }  
};
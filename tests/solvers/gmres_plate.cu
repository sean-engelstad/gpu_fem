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

int main(int argc, char* argv[]) {
    // prelim command line inputs
    // --------------------------

    bool print = false;

    if (argc < 4) {
        std::cerr << "Usage: ./program <ordering> <fill> [p_factor or k] <nxe>\n";
        std::cerr << "Example: ./program qorder ILUk 3 100\n";
        return 1;
    }

    std::string ordering = argv[1];  // "none", "RCM", or "qorder"
    std::string fill_type = argv[2]; // "nofill", "ILUk", or "LU"

    double p_factor = 0;
    int k = 0;
    int nxe = 50;

    if (ordering == "qorder") {
        if (argc < 5) {
            std::cerr << "Error: qorder requires a p_factor\n";
            return 1;
        }
        p_factor = std::stod(argv[3]);
    }

    if (fill_type == "ILUk") {
        if ((ordering != "qorder" && argc < 5) || (ordering == "qorder" && argc < 6)) {
            std::cerr << "Error: ILUk requires a value for k\n";
            return 1;
        }
        // ILUk's k value is argv[3] if ordering ≠ qorder, argv[4] if ordering = qorder
        k = std::atoi(argv[ordering == "qorder" ? 4 : 3]);
        nxe = std::atoi(argv[ordering == "qorder" ? 5 : 4]);
    } else {
        nxe = std::atoi(argv[3]);
    }

    if (print) printf("nxe = %d\n", nxe);

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

    if (ordering == "RCM") {
        bsr_data.RCM_reordering();
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return 1;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return 1;
    }

    bsr_data.compute_full_LU_pattern(fillin, print);
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
    int n_iter = 100, max_iter = 200;
    T abs_tol = 1e-7, rel_tol = 1e-8;
    constexpr bool use_precond = true, debug = false;
    CUSPARSE::GMRES_solve<T, use_precond, debug>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol);

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
    if (print) printf("resid_norm = %.4e\n", resid_norm);

    bool passed = abs(resid_norm) < 1e-6;
    printTestReport("direct LU plate solve", passed, resid_norm);
};
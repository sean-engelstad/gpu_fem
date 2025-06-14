/**
 * This program runs a plate test case with various options:
 *
 * Command-line arguments:
 *  --case            : "plate" or "ucrm" (currently only "plate" is supported)
 *  --nxe             : number of elements in x-direction (plate only)
 *  --solve           : "LU" or "GMRES" to select the linear solver
 *  --just_assembly   : if true, assembles K only (no solve)
 *  --ILUk            : fill-in level for ILU(k)
 *  --ordering        : reordering type: "RCM", "AMD", "qorder", or "none"
 *  --test_type       : label for test, e.g. "assembly" or "solve"
 *  --debug           : enable debug prints
 *  --write_vtk       : write VTK output for visualization
 *
 * Example:
 * ./run_case --case plate --nxe 80 --solve GMRES --ILUk 3 --ordering qorder --test_type solve
 */

#include <getopt.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

#include "../plate/_src/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

void time_linear_static(int nxe, std::string ordering, std::string fill_type, bool LU_solve = true, int ILU_k = 5,
                        double p_factor = 1.0, bool print = true, bool write_vtk = false, bool debug = false,
                        bool just_assembly = false, std::string test_type = "solve") {
    using T = double;

    int rcm_iters = 5;
    auto start = std::chrono::high_resolution_clock::now();

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false, is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    auto &bsr_data = assembler.getBsrData();
    double fillin = 10.0;

    if (ordering == "RCM")
        bsr_data.RCM_reordering(rcm_iters);
    else if (ordering == "AMD")
        bsr_data.AMD_reordering();
    else if (ordering == "qorder")
        bsr_data.qorder_reordering(p_factor, rcm_iters, debug);
    else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill")
        bsr_data.compute_nofill_pattern();
    else if (fill_type == "ILUk")
        bsr_data.compute_ILUk_pattern(ILU_k, fillin, debug);
    else if (fill_type == "LU")
        bsr_data.compute_full_LU_pattern(fillin);
    else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    assembler.moveBsrDataToDevice();

    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, 1.0);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();

    assembler.apply_bcs(res); // warmup call
    assembler.add_residual(res, print);
    bool include_res = false; // false
    assembler.add_jacobian(res, kmat, print, include_res);
    assembler.apply_bcs(res); // very fast
    assembler.apply_bcs(kmat); // very fast

    if (!just_assembly) {
        if (LU_solve) {
            CUSPARSE::direct_LU_solve(kmat, loads, soln);
        } else {
            int n_iter = 200, max_iter = 400;
            T abs_tol = 1e-11, rel_tol = 1e-14;
            CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, debug);
        }

        if (write_vtk) {
            auto h_soln = soln.createHostVec();
            printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "plate.vtk");
        }

        assembler.set_variables(soln);
        assembler.add_residual(res);
        auto rhs = assembler.createVarsVec();
        CUBLAS::axpy(1.0, loads, rhs);
        CUBLAS::axpy(-1.0, res, rhs);
        assembler.apply_bcs(rhs);
        double resid_norm = CUBLAS::get_vec_norm(rhs);
        if (debug) printf("resid_norm = %.4e\n", resid_norm);
    }

    // temp print
    int ndof = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    int nelems = assembler.get_num_elements();
    int nnz = kmat.get_nnz();
    printf("ndof %d, nnodes %d, nelems %d, nnz %d\n", ndof, nnodes, nelems, nnz);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tot_time = end - start;

    std::string solve_str = LU_solve ? "LU" : "GMRES";
    std::string fill_str = (fill_type == "ILUk") ? "ILU(" + std::to_string(ILU_k) + ")" : fill_type;

    printf("%s: nxe=%d, ordering=%s, fill=%s, solver=%s, time=%.4f s\n",
           test_type.c_str(), nxe, ordering.c_str(), fill_str.c_str(), solve_str.c_str(), tot_time.count());
}

int main(int argc, char **argv) {
    // defaults
    std::string case_name = "plate";
    int nxe = 100;
    std::string solve_method = "LU";
    bool just_assembly = false;
    int ILU_k = 3;
    std::string ordering = "AMD";
    std::string test_type = "just_assembly"; // #solve
    bool debug = false, write_vtk = false;

    const struct option long_options[] = {
        {"case", required_argument, 0, 'c'},
        {"nxe", required_argument, 0, 'x'},
        {"solve", required_argument, 0, 's'},
        {"just_assembly", no_argument, 0, 'j'},
        {"ILUk", required_argument, 0, 'k'},
        {"ordering", required_argument, 0, 'o'},
        {"test_type", required_argument, 0, 't'},
        {"debug", no_argument, 0, 'd'},
        {"write_vtk", no_argument, 0, 'v'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "c:x:s:k:o:t:jdv", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'c': case_name = optarg; break;
            case 'x': nxe = std::stoi(optarg); break;
            case 's': solve_method = optarg; break;
            case 'j': just_assembly = true; break;
            case 'k': ILU_k = std::stoi(optarg); break;
            case 'o': ordering = optarg; break;
            case 't': test_type = optarg; break;
            case 'd': debug = true; break;
            case 'v': write_vtk = true; break;
            default:
                std::cerr << "Invalid argument or missing value.\n";
                return 1;
        }
    }

    if (case_name != "plate") {
        std::cerr << "Case type '" << case_name << "' not implemented yet.\n";
        return 1;
    }

    std::string fill_type = (solve_method == "LU") ? "LU" : "ILUk";
    bool LU = (solve_method == "LU");

    time_linear_static(nxe, ordering, fill_type, LU, ILU_k, 1.0, true, write_vtk, debug, just_assembly, test_type);
    return 0;
}

/* check ucrm preconditioner effect on vector (part of Kyle Anderson's paper) */

#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include "../../examples/uCRM/_src/crm_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

void compute_ucrm_reordering(MPI_Comm& comm, std::string ordering, std::string fill_type,
                             int rcm_iters, double p_factor, int ILUk) {
    using T = double;

    auto start0 = std::chrono::high_resolution_clock::now();

    // uCRM mesh files can be found at:
    // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    TACSMeshLoader mesh_loader{comm};
    mesh_loader.scanBDFFile("../../examples/uCRM/CRM_box_2nd.bdf");
    // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

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

    double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

    // make the assembler from the uCRM mesh
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // BSR factorization
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;

    // BSR reordering
    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters);
    } else if (ordering == "random") {
        bsr_data.random_reordering();
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    // check valid precond
    bsr_data.check_valid_perm();

    // BSR fillin
    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(ILUk);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    printf("nnodes %d, nnzb %d\n", bsr_data.nnodes, bsr_data.nnzb);

    // writeout to csv the sparsity
    write_to_csv<int>(bsr_data.nnodes + 1, bsr_data.rowp, "csv/rowp.csv");
    write_to_csv<int>(bsr_data.nnzb, bsr_data.cols, "csv/cols.csv");

    // get chain lengths and write that out
    double chain_lengths[bsr_data.nnodes];
    bsr_data.get_chain_lengths(chain_lengths);
    write_to_csv<double>(bsr_data.nnodes, chain_lengths, "csv/chain_lengths.csv");

    // now try applying chain lengths to the vector (with direct LU solve M^-1 v, becomes direct LU only if full LU pattern otherwise it's just M^-1 b)
    // -----------------------------

    assembler.moveBsrDataToDevice();

    // get the loads
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);
    double load_mag = 1e-5;
    double *h_loads_ptr = h_loads.getPtr();
    for (int idof = 0; idof < 6 * nnodes; idof++) {
        h_loads_ptr[idof] = load_mag + 100.0 * load_mag * ((double) rand() / RAND_MAX);
    }
    auto rhs = h_loads.createDeviceVec();
    assembler.apply_bcs(rhs);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.set_variables(vars);
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // direct LU solve
    CUSPARSE::direct_LU_solve<T>(kmat, rhs, soln);

    // copy solution back to host and writeout vectors
    auto h_soln = soln.createHostVec();
    write_to_csv<double>(6 * bsr_data.nnodes, h_loads.getPtr(), "csv/unprecond_vec.csv");
    write_to_csv<double>(6 * bsr_data.nnodes, h_soln.getPtr(), "csv/precond_vec.csv");

    // // assemble the kmat
    // assembler.set_variables(vars);
    // assembler.add_jacobian(res, kmat);
    // assembler.apply_bcs(res);
    // assembler.apply_bcs(kmat);

    load_mag = 1e2;
    memset(h_loads_ptr, 0.0, 6 * nnodes * sizeof(T));
    for (int inode = 0; inode < nnodes; inode++) {
        h_loads_ptr[6 * inode + 2] = load_mag;
    }
    auto d_loads = h_loads.createDeviceVec();
    assembler.apply_bcs(d_loads);

    // now try a GMRES solve (should be fine to restart since kmat values copied to do direct LU)
    soln.zeroValues();
    int n_iter = 200, max_iter = 200;
    T abs_tol = 1e-8, rel_tol = 1e-8;
    constexpr bool right = true, modifiedGS = true; // better with modifiedGS true, yeah it is..
    CUSPARSE::GMRES_solve<T, right, modifiedGS>(kmat, d_loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./program <ordering> <fill> [p_factor or k]\n";
        std::cerr << "Example: ./program qorder ILUk 3\n";
        return 1;
    }

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::string ordering = argv[1];   // "none", "RCM", or "qorder"
    std::string fill_type = argv[2];  // "nofill", "ILUk", or "LU"

    int rcm_iters = 5;
    double p_factor = 0;
    int k = 0;

    if (ordering == "qorder") {
        if (argc < 4) {
            std::cerr << "Error: qorder requires a p_factor\n";
            return 1;
        }
        p_factor = std::stod(argv[3]);
    }
    if (ordering == "RCM") {
        if (argc < 4) {
            std::cerr << "Error: RCM requires a number of iterations\n";
            return 1;
        }
        rcm_iters = std::atoi(argv[3]);
    }

    if (fill_type == "ILUk") {
        if ((ordering != "qorder" && argc < 4) || (ordering == "qorder" && argc < 5)) {
            std::cerr << "Error: ILUk requires a value for k\n";
            return 1;
        }
        // ILUk's k value is argv[3] if ordering ≠ qorder, argv[4] if ordering = qorder
        bool multi_input_order = ordering == "RCM" || ordering == "qorder";
        k = std::atoi(argv[multi_input_order ? 4 : 3]);
    }

    // now call unittests
    compute_ucrm_reordering(comm, ordering, fill_type, rcm_iters, p_factor, k);
}
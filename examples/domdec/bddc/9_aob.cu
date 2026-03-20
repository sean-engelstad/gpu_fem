// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"
#include "multigrid/utils/fea.h"
#include <string>
#include <chrono>
#include "multigrid/solvers/direct/cusp_directLU.h"

#include "domdec/bddc_assembler.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T>
T get_max_disp(DeviceVec<T> &d_soln, int idof = 2) {
    T *h_soln = d_soln.createHostVec().getPtr();
    int nvars = d_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}

template <typename T>
T get_max_disp(HostVec<T> h_soln, int idof = 2) {
    T *h_soln_ptr = h_soln.getPtr();
    int nvars = h_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln_ptr[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}


// declare a couple different types of loads..
template <typename T>
struct UniformPressure {
    T q0;

    __HOST_DEVICE__
    UniformPressure(T q0_) : q0(q0_) {}

    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        return q0;
    }
};

template <typename T>
struct ObliqueShearSineLoad {
    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        const T pi = T(3.14159265358979323846);

        T r = sqrt(x * x + y * y);
        T theta = atan2(y, x);

        return sin(T(5.0) * pi * r) * cos(T(4.0) * theta);
    }
};

int main(int argc, char **argv) {
    // NOTE : this version uses inner direct solvers

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;


    using T = double;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC4
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;

    // MITC9, doesn't help the global solution recovery be more accurate
    // using Quad = QuadQuadraticQuadrature<T>;
    // using Basis = LagrangeQuadBasis<T, Quad, 2>;

    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    using BDDC = BddcSolver<T, Assembler, VecType, BsrMat>;
    // const bool MULTI_SMOOTH = true;
    const bool MULTI_SMOOTH = false; // often not MULTI_SMOOTH is better..
    using InnerSolver = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH>;
    using InnerSolver_JUSTLU = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH, true>;
    using GamPCG = MatrixFreePCGSolver<T, BDDC>; // BDDC is the operator and preconditioner


    // int level = 0; // wing mesh level
    int level = 1;
    int nxe_subdomain_size = 4;
    // int nxe_subdomain_size = 8;
    T omega;
    int nsmooth, fill_level;
    T thick = 1e-3;
    // bool print_mem = false;
    bool print_mem = true;
    T mag = 1.0;
    
    // optional smoothing
    // 1) if ILU(k) here, ability to do multiple smoothing steps (Richardson)
    // omega = 0.5, nsmooth = 2, fill_level = 2;
    // or single-level (no multiple smoothing) and ILU(k)
    // omega = 1.0, nsmooth = 1, fill_level = 2;
    // 2) or if full LU fillin
    // think I actually need to run with full direct fillin (cause otherwise solution recovery is wrong)
    omega = 1.0, nsmooth = 1, fill_level = -1; // -1 indicates full LU fillin

    if (!MULTI_SMOOTH) {
        printf("NOTE: MULTI_SMOOTH is false, so omega, nsmooth inputs are ignored.\n");
    }

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--thick") == 0) {
            if (i + 1 < argc) {
                thick = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --thick\n";
                return 1;
            }
        } else if (strcmp(arg, "--mag") == 0) {
            if (i + 1 < argc) {
                mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --mag\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--subdomain") == 0) {
            if (i + 1 < argc) {
                nxe_subdomain_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --subdomain\n";
                return 1;
            }
        } else if (strcmp(arg, "--fill") == 0) {
            if (i + 1 < argc) {
                fill_level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --fill\n";
                return 1;
            }
        } else if (strcmp(arg, "--print_mem") == 0) {
            if (i + 1 < argc) {
                print_mem = (bool)std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --print_mem\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/krylov] [--nxe value] [--SR value] [--nsmooth int]" << std::endl;
            return 1;
        }
    }


    // =================

    // read the ESP/CAPS => nastran mesh for TACS
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
    mesh_loader.scanBDFFile(fname.c_str());
    double E = 70e9, nu = 0.3;  // material & thick properties (start thicker first try)
    printf("making assembler for mesh '%s'\n", fname.c_str());
    
    // create the TACS Assembler from the mesh loader
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto soln2 = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto res = assembler.createVarsVec();

    // kmat assembly (not actually needed here, but is needed for the nonlinear problem))
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_assembly = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_assembly = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end_assembly - start_assembly;

    cublasHandle_t cublasHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    bool print_timing = true; // profiling
    auto bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing);


    bddc->setup_wing_subdomains(nxe_subdomain_size, nxe_subdomain_size);
    printf("ONLY DEBUG : wing_setup_subdomains at the moment\n");
    return;

    // perform LU fillin and reordering (optional)
    auto &I_bsr_data = bddc->I_bsr_data;
    auto &IE_bsr_data = bddc->IE_bsr_data;
    I_bsr_data.AMD_reordering(); 
    if (fill_level != -1) { 
        I_bsr_data.compute_ILUk_pattern(fill_level, 10.0);
    } else {
        I_bsr_data.compute_full_LU_pattern(10.0);
    }

    IE_bsr_data.AMD_reordering(); 
    if (fill_level != -1) { 
        IE_bsr_data.compute_ILUk_pattern(fill_level, 10.0);
    } else {
        IE_bsr_data.compute_full_LU_pattern(10.0);
    }

    // now compute matrix sparsity, copy maps
    bddc->setup_matrix_sparsity();

    // then perform coarse matrix fillin and compute sparsity
    auto &Svv_bsr_data = bddc->Svv_bsr_data;
    Svv_bsr_data.AMD_reordering();
    if (fill_level != -1) { 
        Svv_bsr_data.compute_ILUk_pattern(fill_level, 10.0);
    } else {
        Svv_bsr_data.compute_full_LU_pattern(10.0);
    }

    bddc->setup_coarse_matrix_sparsity();

    // assemble local FETI-DP blocks
    bddc->assemble_subdomains();

    // external load (can add internally)
    ObliqueShearSineLoad<T> load;
    bddc->add_subdomain_fext(load, mag);

    bddc->set_IEV_residual(1.0, 0.0, vars);

    // ----------------------------------------
    // you still need actual solver objects here
    // ----------------------------------------
    //
    // Example sketch only; replace with your actual solver classes:
    
    // just LU allowed for IE and I solvers to reduce mem footprint (1/2 as much memory for them)
    //   means only the LU factor is stored, not original matrix as well
    auto *ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE, omega, nsmooth);
    auto *i_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->kmat_I, omega, nsmooth);
    // note assembler not really used in S_VV here or above classes either.. (and def not for size)
    auto *v_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV, omega, nsmooth);
    // auto *v_solver  = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->S_VV);
    
    
    bddc->set_inner_solvers(ie_solver, i_solver, v_solver);

    // if (nxe < 10) {
    //     // DEBUG small matrices
    //     bool print_IEV = true; // already verified
    //     bool print_IE = false; // already verified 
    //     bool print_I = false; // already verified now
    //     bddc->debug_IEV_matrices(print_IEV, print_IE, print_I);
    // }

    // factor each solver
    ie_solver->factor();
    i_solver->factor();

    // then assemble coarse problem (as it uses IE solver) before factoring v_solver
    bddc->assemble_coarse_problem();
    // bddc->debug_SVV_matrix();

    v_solver->factor();    

    // lambda rhs
    VecType<T> gam_rhs(bddc->getLambdaSize());
    VecType<T> gam(bddc->getLambdaSize());
    bddc->get_lam_rhs(gam_rhs);

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    opts.ncycles = 50;
    // opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 5;
    opts.debug = true;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;

    auto *gam_solver =
        new GamPCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);

    // optional: true initial residual before solve
    gam.zeroValues();
    T init_gam_resid = gam_solver->getResidualNorm(gam_rhs, gam);
    printf("initial gamma residual = %.8e\n", init_gam_resid);

    // ==============================================
    // SOLVE (TIMING)
    // ==============================================

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // run the assembly and factor (call from krylov solver, that also calls for preconditioner)
    gam_solver->update_after_assembly(vars);
        
    // end timing of the factor + assembly part
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> IEV_factor_time = start1 - start0;


    bool gam_fail = gam_solver->solve(gam_rhs, gam, true);
    bddc->get_global_soln(gam, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end - start1;
    std::chrono::duration<double> total_time = end - start0;

    T final_gam_resid = gam_solver->getResidualNorm(gam_rhs, gam);
    // printf("final lambda residual = %.8e in %.4e sec\n", final_gam_resid, solve_time.count());
    // printf("\nassembly in %.4e sec, IEV-assembly+factor in %.4e sec, PCG-solve in %.4e sec\n", assembly_time.count(), IEV_factor_time.count(), solve_time.count());
    // printf("\ttotal IEV-setup and PCG in %.4e sec\n", total_time.count());

    printf("\nBDDC solve summary:\n");
    printf("  final gamma residual : %.8e\n\n", final_gam_resid);

    printf("Timing breakdown:\n");
    printf("  assembly              : %.4e s\n", assembly_time.count());
    printf("  IEV assembly+factor   : %.4e s\n", IEV_factor_time.count());
    printf("  PCG solve             : %.4e s\n", solve_time.count());
    printf("  --------------------------------\n");
    printf("  total setup + solve   : %.4e s\n\n", total_time.count());

    if (print_mem) {
        // get memory usage
        size_t bytes_per_double = sizeof(double);
        double bytes_per_block = static_cast<double>(bytes_per_double) * 36.0;

        // nnzb counts
        int kmat_nnzb   = kmat.getBsrData().nnzb;
        int IEV_nnzb    = bddc->kmat_IEV->getBsrData().nnzb;
        int IE_nnzb     = IE_bsr_data.nnzb;
        int I_nnzb      = I_bsr_data.nnzb;
        int coarse_nnzb = Svv_bsr_data.nnzb;

        // no-fill counts (must already exist in your code)
        int IE_nofill_nnzb     = bddc->IE_nofill_nnzb;
        int I_nofill_nnzb      = bddc->I_nofill_nnzb;
        int coarse_nofill_nnzb = bddc->Svv_nofill_nnzb;   // or whatever your variable is called

        // memory in MB
        double kmat_mem_mb   = bytes_per_block * static_cast<double>(kmat_nnzb)   / 1024.0 / 1024.0;
        double IEV_mem_mb    = bytes_per_block * static_cast<double>(IEV_nnzb)    / 1024.0 / 1024.0;
        double IE_mem_mb     = bytes_per_block * static_cast<double>(IE_nnzb)     / 1024.0 / 1024.0;
        double I_mem_mb      = bytes_per_block * static_cast<double>(I_nnzb)      / 1024.0 / 1024.0;
        double coarse_mem_mb = bytes_per_block * static_cast<double>(coarse_nnzb) / 1024.0 / 1024.0;

        // total stored blocks:
        //   kmat * 1
        //   IEV  * 1
        //   IE   * 2
        //   I    * 1
        //   coarse * 1
        long long total_stored_nnzb =
            static_cast<long long>(kmat_nnzb) +
            static_cast<long long>(IEV_nnzb) +
            2LL * static_cast<long long>(IE_nnzb) +
            static_cast<long long>(I_nnzb) +
            static_cast<long long>(coarse_nnzb);

        double total_mem_mb =
            bytes_per_block * static_cast<double>(total_stored_nnzb) / 1024.0 / 1024.0;

        // fill ratios
        double IE_fill_ratio =
            (IE_nofill_nnzb > 0) ? static_cast<double>(IE_nnzb) / static_cast<double>(IE_nofill_nnzb) : 0.0;
        double I_fill_ratio =
            (I_nofill_nnzb > 0) ? static_cast<double>(I_nnzb) / static_cast<double>(I_nofill_nnzb) : 0.0;
        double coarse_fill_ratio =
            (coarse_nofill_nnzb > 0) ? static_cast<double>(coarse_nnzb) / static_cast<double>(coarse_nofill_nnzb) : 0.0;

        // optional: added fill blocks
        int IE_fill_added     = IE_nnzb - IE_nofill_nnzb;
        int I_fill_added      = I_nnzb - I_nofill_nnzb;
        int coarse_fill_added = coarse_nnzb - coarse_nofill_nnzb;
        T overall_fill_ratio = total_stored_nnzb * 1.0 / kmat_nnzb;

        printf("\nFETI-DP memory breakdown:\n");
        printf("  kmat                 : nnzb = %d, mem = %.4f MB\n", kmat_nnzb, kmat_mem_mb);
        printf("  IEV                  : nnzb = %d, mem = %.4f MB\n", IEV_nnzb, IEV_mem_mb);
        printf("  IE                   : nnzb = %d, mem = %.4f MB\n", IE_nnzb, IE_mem_mb);
        printf("    nofill             : %d\n", IE_nofill_nnzb);
        printf("    fill added         : %d\n", IE_fill_added);
        printf("    fill ratio         : %.4f\n", IE_fill_ratio);
        printf("  I                    : nnzb = %d, mem = %.4f MB\n", I_nnzb, I_mem_mb);
        printf("    nofill             : %d\n", I_nofill_nnzb);
        printf("    fill added         : %d\n", I_fill_added);
        printf("    fill ratio         : %.4f\n", I_fill_ratio);
        printf("  coarse S_VV          : nnzb = %d, mem = %.4f MB\n", coarse_nnzb, coarse_mem_mb);
        printf("    nofill             : %d\n", coarse_nofill_nnzb);
        printf("    fill added         : %d\n", coarse_fill_added);
        printf("    fill ratio         : %.4f\n", coarse_fill_ratio);
        printf("  --------------------------------\n");
        printf("  total stored nnzb    : %lld\n", total_stored_nnzb);
        printf("  total memory         : %.4f MB\n", total_mem_mb);
        printf("    overall fill ratio : %.4f\n\n", overall_fill_ratio);

    }


    if (gam_fail) {
        printf("BDDC lambda PCG failed\n");
    }

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/plate_bddc.vtk");

    // compare to direct solver (the solution)
    bool compare_direct = true;
    // bool compare_direct = false;
    if (compare_direct) {
        // and need it globally too..
        auto loads = assembler.createVarsVec();
        assembler.add_fext_fast(load, mag, loads);
        assembler.apply_bcs(loads);

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader2{comm};
        std::string fname2 = "../../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
        mesh_loader2.scanBDFFile(fname2.c_str());
        printf("making assembler for mesh '%s'\n", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler2 = Assembler::createFromBDF(mesh_loader2, Data(E, nu, thick));

        // BSR factorization (need to change it to )
        auto& bsr_data = assembler2.getBsrData();
        double fillin = 10.0;  // 10.0
        bool print = true;
        bsr_data.AMD_reordering();
        bsr_data.compute_full_LU_pattern(fillin, print);
        assembler2.moveBsrDataToDevice();


        auto kmat2 = createBsrMat<Assembler, VecType<T>>(assembler2);
        assembler2.add_jacobian_fast(kmat2);
        assembler2.apply_bcs(kmat2);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start3 = std::chrono::high_resolution_clock::now();

        CUSPARSE::direct_LU_solve(kmat2, loads, soln2);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solve_time_dir = end3 - start3;

        printf("BDDC solve time in %.4e sec, direct in %.4e sec\n", total_time.count(), solve_time_dir.count());

        // T lin_max_disp = get_max_disp(soln2);
        auto h_soln3 = soln2.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler2, h_soln3, "out/plate_lin.vtk");

        // now also compute solution error on host and print to VTK
        auto h_err = HostVec<T>(h_soln3.getSize());
        for (int i = 0; i < h_soln3.getSize(); i++) {
            h_err[i] = h_soln3[i] - h_soln[i];
        }
        printToVTK<Assembler,HostVec<T>>(assembler2, h_err, "out/plate_err.vtk");

        // for (int idof = 0; idof < 6; idof++) {
        //     T orig_nrm = get_max_disp(h_soln, idof);
        //     T err_nrm = get_max_disp(h_err, idof);
        //     T rel_nrm = err_nrm / (orig_nrm + 1e-30);
        //     printf("\tidof %d, orig |u|=%.4e, err nrm %.4e, rel err nrm %.4e to direct solve\n", idof, orig_nrm, err_nrm, rel_nrm);
        // }
        
        // now compute the residuals of each..
        int nvars = assembler2.get_num_vars();
        assembler2.set_variables(soln2);
        assembler2.add_residual_fast(res);
        T a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler2.apply_bcs(res);
        T load_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, loads.getPtr(), 1, &load_nrm));
        T res_norm_direct;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, res.getPtr(), 1, &res_norm_direct));
        T res_rel_nrm_direct = res_norm_direct / load_nrm;
        printf("\tres_nrm_direct %.4e, rel_nrm %.4e\n", res_norm_direct, res_rel_nrm_direct);

        assembler2.set_variables(soln);
        res.zeroValues();
        assembler2.add_residual_fast(res);
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler2.apply_bcs(res);
        T res_norm_feti;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, res.getPtr(), 1, &res_norm_feti));
        T res_rel_nrm_feti = res_norm_feti / load_nrm;
        printf("\tres_nrm_feti %.4e, rel_nrm %.4e\n", res_norm_feti, res_rel_nrm_feti);
        
    }

    delete gam_solver;
    delete ie_solver;
    delete i_solver;
    delete v_solver;
    delete bddc;


    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}

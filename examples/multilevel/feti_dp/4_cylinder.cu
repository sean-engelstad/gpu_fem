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

#include "domdec/fetidp_assembler.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
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
    using T = double;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    using FETIDP = FetidpSolver<T, Assembler, VecType, BsrMat>;
    // const bool MULTI_SMOOTH = true;
    const bool MULTI_SMOOTH = false; // often not MULTI_SMOOTH is better..
    using InnerSolver = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH>;
    using InnerSolver_JUSTLU = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH, true>;
    using LamPCG = MatrixFreePCGSolver<T, FETIDP>; // FETIDP is the operator and preconditioner

    // int nxe = 6, nxe_subdomain_size = 2;
    int nxe = 256, nxe_subdomain_size = 4;
    // NOTE : full fillin with fill_level = -1, but lower fill results in less ILU(k) factor time
    // for the coarse problem..
    T omega;
    int nsmooth, fill_level;
    T thick = 1e-3;
    bool print_mem = false;
    T mag = 1.0;
    
    // optional smoothing
    // omega = 0.5, nsmooth = 2, fill_level = 2;
    // or single-level (no multiple smoothing) and ILU(k)
    // omega = 1.0, nsmooth = 1, fill_level = 3;
    // 2) or if full LU fillin
    // think I actually need to run with full direct fillin (cause otherwise solution recovery is wrong)
    omega = 1.0, nsmooth = 1, fill_level = -1; // -1 indicates full LU fillin

    if (!MULTI_SMOOTH) {
        printf("NOTE: MULTI_SMOOTH is false, so omega, nsmooth inputs are ignored.\n");
    }

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--thick") == 0) {
            if (i + 1 < argc) {
                thick = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --thick\n";
                return 1;
            }
        }  else if (strcmp(arg, "--omega") == 0) {
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
        }  else if (strcmp(arg, "--mag") == 0) {
            if (i + 1 < argc) {
                mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --mag\n";
                return 1;
            }
        } else if (strcmp(arg, "--print_mem") == 0) {
            if (i + 1 < argc) {
                print_mem = (bool)std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --print_mem\n";
                return 1;
            }
        }  else if (strcmp(arg, "--nsmooth") == 0) {
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

    int nye = nxe;
    int nxs = nxe / nxe_subdomain_size;
    int nys = nxe / nxe_subdomain_size;

    // double SR = 1e1;
    double SR = 1e3;
    double Lx = 1.0;
    double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye;
    double R = 0.5;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto assembler = createCylinderAssembler<Assembler>(nxe, nye, Lx, R, E, nu, thick, 
        imperfection, imp_x, imp_hoop);


    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

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
    auto fetidp = new FETIDP(cublasHandle, cusparseHandle, assembler, kmat, print_timing);

    bool close_hoop = true; // true for cylinder case (not cylindrical panel)
    fetidp->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);

    // perform LU fillin and reordering (optional)
    auto &I_bsr_data = fetidp->I_bsr_data;
    auto &IE_bsr_data = fetidp->IE_bsr_data;
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
    fetidp->setup_matrix_sparsity();

    // then perform coarse matrix fillin and compute sparsity
    auto &Svv_bsr_data = fetidp->Svv_bsr_data;
    Svv_bsr_data.AMD_reordering();
    if (fill_level != -1) { 
        Svv_bsr_data.compute_ILUk_pattern(fill_level, 10.0);
    } else {
        Svv_bsr_data.compute_full_LU_pattern(10.0);
    }

    fetidp->setup_coarse_matrix_sparsity();

    // assemble local FETI-DP blocks
    fetidp->assemble_subdomains();

    // external load
    ObliqueShearSineLoad<T> load;
    fetidp->add_subdomain_fext(load, mag);

    // ----------------------------------------
    // you still need actual solver objects here
    // ----------------------------------------
    //
    // Example sketch only; replace with your actual solver classes:
    
    // just LU allowed for IE and I solvers to reduce mem footprint (1/2 as much memory for them)
    //   means only the LU factor is stored, not original matrix as well
    auto *ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *fetidp->kmat_IE, omega, nsmooth);
    auto *i_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *fetidp->kmat_I, omega, nsmooth);
    // note assembler not really used in S_VV here or above classes either.. (and def not for size)
    auto *v_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *fetidp->S_VV, omega, nsmooth);
    // auto *v_solver  = new InnerSolver(cublasHandle, cusparseHandle, assembler, *fetidp->S_VV);
    
    fetidp->set_inner_solvers(ie_solver, i_solver, v_solver);

    // if (nxe < 10) {
    //     // DEBUG small matrices
    //     bool print_IEV = true; // already verified
    //     bool print_IE = false; // already verified 
    //     bool print_I = false; // already verified now
    //     fetidp->debug_IEV_matrices(print_IEV, print_IE, print_I);
    // }

    // factor each solver
    ie_solver->factor();
    i_solver->factor();

    // then assemble coarse problem (as it uses IE solver) before factoring v_solver
    fetidp->assemble_coarse_problem();
    // fetidp->debug_SVV_matrix();

    v_solver->factor();    

    // lambda rhs
    VecType<T> lam_rhs(fetidp->getLambdaSize());
    VecType<T> lam(fetidp->getLambdaSize());
    fetidp->get_lam_rhs(lam_rhs);

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    opts.ncycles = 50;
    // opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 5;
    opts.debug = true;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;

    auto *lam_solver =
        new LamPCG(cublasHandle, fetidp, fetidp, opts, fetidp->getLambdaSize(), 0);

    // DEBUG:
    // lam.zeroValues();
    // fetidp->solve(lam_rhs, lam);
    // T *h_lam_debug = lam.createHostVec().getPtr();
    // for (int inode = 0; inode < lam.getSize() / 6; inode++) {
    //     printf("h_lam_debug\n");
    //     for (int idof = 2; idof < 5; idof++) {
    //         printf("%.6e,", h_lam_debug[6 * inode + idof]);
    //     }
    //     printf("\n");
    // }

    // optional: true initial residual before solve
    lam.zeroValues();
    T init_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("initial lambda residual = %.8e\n", init_lam_resid);

    // ==============================================
    // SOLVE (TIMING)
    // ==============================================

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // run the assembly and factor (call from krylov solver, that also calls for preconditioner)
    lam_solver->update_after_assembly(vars);
        
    // end timing of the factor + assembly part
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> IEV_factor_time = start1 - start0;


    bool lam_fail = lam_solver->solve(lam_rhs, lam, true);
    fetidp->get_global_soln(lam, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end - start1;
    std::chrono::duration<double> total_time = end - start0;

    T final_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    // printf("final lambda residual = %.8e in %.4e sec\n", final_lam_resid, solve_time.count());
    // printf("\nassembly in %.4e sec, IEV-assembly+factor in %.4e sec, PCG-solve in %.4e sec\n", assembly_time.count(), IEV_factor_time.count(), solve_time.count());
    // printf("\ttotal IEV-setup and PCG in %.4e sec\n", total_time.count());

    printf("\nFETI-DP solve summary:\n");
    printf("  final lambda residual : %.8e\n\n", final_lam_resid);

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
        int IEV_nnzb    = fetidp->kmat_IEV->getBsrData().nnzb;
        int IE_nnzb     = IE_bsr_data.nnzb;
        int I_nnzb      = I_bsr_data.nnzb;
        int coarse_nnzb = Svv_bsr_data.nnzb;

        // no-fill counts (must already exist in your code)
        int IE_nofill_nnzb     = fetidp->IE_nofill_nnzb;
        int I_nofill_nnzb      = fetidp->I_nofill_nnzb;
        int coarse_nofill_nnzb = fetidp->Svv_nofill_nnzb;   // or whatever your variable is called

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
        printf("  total memory         : %.4f MB\n\n", total_mem_mb);

    }


    if (lam_fail) {
        printf("FETI-DP lambda PCG failed\n");
    }

    // T *h_lam_soln = lam.createHostVec().getPtr();
    // for (int inode = 0; inode < lam.getSize() / 6; inode++) {
    //     printf("h_lam_soln\n");
    //     for (int idof = 2; idof < 5; idof++) {
    //         printf("%.6e,", h_lam_soln[6 * inode + idof]);
    //     }
    //     printf("\n");
    // }
    
    // done with Krylov solution, now report back


    // T *h_soln0 = soln.createHostVec().getPtr();
    // printf("\nh_glob_soln\n");
    // for (int inode = 0; inode < soln.getSize() / 6; inode++) {
    //     printf("glob soln node %d: ", inode);
    //     for (int idof = 2; idof < 5; idof++) {
    //         printf("%.6e,", h_soln0[6 * inode + idof]);
    //     }
    //     printf("\n");
    // }

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/cylinder_fetidp.vtk");

    delete lam_solver;
    delete ie_solver;
    delete i_solver;
    delete v_solver;
    delete fetidp;

    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}

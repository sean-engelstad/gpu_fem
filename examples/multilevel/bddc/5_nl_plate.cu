// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"



// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

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
#include "domdec/domdec_pcg_wrapper.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"


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


void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T>
void solve_bddc(int nxe, int nxe_subdomain_size, T omega, int nsmooth, int fill_level, 
    T thick, bool print_mem, T mag) {
    // =================

    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    // constexpr bool is_nonlinear = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    using BDDC = BddcSolver<T, Assembler, VecType, BsrMat>;
    // const bool MULTI_SMOOTH = true;
    const bool MULTI_SMOOTH = false; // often not MULTI_SMOOTH is better..
    using InnerSolver = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH>;
    using InnerSolver_JUSTLU = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH, true>;
    using PCG = MatrixFreePCGSolver<T, BDDC>; // BDDC is the operator and preconditioner


    if (!MULTI_SMOOTH) {
        printf("NOTE: MULTI_SMOOTH is false, so omega, nsmooth inputs are ignored.\n");
    }

    int nye = nxe;
    int nxs = nxe / nxe_subdomain_size;
    int nys = nxe / nxe_subdomain_size;
    double Lx = 1.0, Ly = 1.0;
    double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye;

    auto assembler = createPlateClampedAssembler<Assembler>(
        nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    auto &bsr_data = assembler.getBsrData();
    // can use nofill pattern in original kmat for FETI-DP
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto fine_vars = assembler.createVarsVec();

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

    // bool print_timing = true; // profiling
    bool print_timing = false;
    auto bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing);

    bool close_hoop = true; // true for cylinder case (not cylindrical panel)
    bddc->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);

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

    // // assemble local FETI-DP blocks
    // bddc->assemble_subdomains();

    // external load
    ObliqueShearSineLoad<T> load;
    bddc->add_subdomain_fext(load, mag);

    // and need it globally too..
    auto loads = assembler.createVarsVec();
    assembler.add_fext_fast(load, mag, loads);
    assembler.apply_bcs(loads);

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

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    opts.ncycles = 100;
    // opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 5;
    // opts.debug = true;
    opts.debug = false;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;

    auto *lam_solver =
        new PCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);

    // run the assembly and factor (call from krylov solver, that also calls for preconditioner)
    lam_solver->update_after_assembly(vars);

    // lambda rhs
    VecType<T> lam_rhs(bddc->getLambdaSize());
    VecType<T> lam(bddc->getLambdaSize());
    bddc->get_lam_rhs(lam_rhs);

    // optional: true initial residual before solve
    lam.zeroValues();
    T init_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("initial lambda residual = %.8e\n", init_lam_resid);

    // prelim linear solve
    lam_solver->solve(lam_rhs, lam, true);

    bddc->get_global_soln(lam, soln);

    T lin_disp = get_max_disp(soln);

    // build the BDDC-krylov wrapper
    using BDDC_WRAPPER = DomDecKrylovWrapper<T, BDDC, PCG>;
    auto wrapper = new BDDC_WRAPPER(bddc, lam_solver); 

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/plate_bddc.vtk");

    // -----------------------------------------------------------
    // 2) actually try Newton-mg solve here (this is just V1, later versions may use FMG cycle so less extra work needs to be done on fine grids)
    //     i.e. you can do most of hte nonlinear solves to get in basin of attraction on coarser grids first.. (then nonlinear fine grid at end only, or some FMG cycle)

    // new nonlinear solver
    // ======================

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    const bool LINE_SEARCH = true;
    // const bool LINE_SEARCH = false;
    const bool USE_FETI_IEV = true; // computes residual_IEV special way
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, BDDC_WRAPPER, LINE_SEARCH, USE_FETI_IEV>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    // need a bit lower base rtol for FETI-DP because some loss in error due to global solution recovery
    T base_rtol = 1e-4;
    // T base_rtol = 1e-10;
    // T base_rtol = 1e-8; // not enough for thinner shell.. (worse rtol requirements for FETI-DP)

    // T initLinSolveRtol = 1e-2;
    // T initLinSolveRtol = 1e-4;
    // T initLinSolveRtol = 1e-6;
    T initLinSolveRtol = base_rtol;
    T linSolveAtol = 1e-30;
    T minLinSolveTol = base_rtol * 1e-3;
    T maxLinSolveTol = base_rtol * 1;
    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads, wrapper, initLinSolveRtol, linSolveAtol, minLinSolveTol, maxLinSolveTol);

    // yeah fine to use predictor in FETI-DP, more robust / stable I think
    bool use_predictor = true, debug = false;
    // bool use_predictor = false, debug = false;
    NL *nl_solver = new NL(cublasHandle, assembler, inner_solver, use_predictor, debug);

    // now try calling it
    T lambda0 = 0.2;
    T inner_atol = 1e-6;

    lam_solver->set_print(false); // turn print off (only show NL outer results, not inner Krylov)

    // ==============================================
    // SOLVE (TIMING)
    // ==============================================

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();
        
    // end timing of the factor + assembly part
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> IEV_factor_time = start1 - start0;

    nl_solver->solve(fine_vars, lambda0, inner_atol);
    T nl_max_disp = get_max_disp(fine_vars);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end - start1;
    std::chrono::duration<double> total_time = end - start0;

    T lin_nl_disp = nl_max_disp / lin_disp;
    printf("NL/LIN disp ratio %.4e\n", lin_nl_disp);

    T final_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    // printf("final lambda residual = %.8e in %.4e sec\n", final_lam_resid, solve_time.count());
    // printf("\nassembly in %.4e sec, IEV-assembly+factor in %.4e sec, PCG-solve in %.4e sec\n", assembly_time.count(), IEV_factor_time.count(), solve_time.count());
    // printf("\ttotal IEV-setup and PCG in %.4e sec\n", total_time.count());

    printf("\nBDDC solve summary:\n");
    printf("  final lambda residual : %.8e\n\n", final_lam_resid);

    printf("Timing breakdown (TODO more breakdown here):\n");
    printf("  NL solve             : %.4e s\n", solve_time.count());
    printf("  --------------------------------\n");
    printf("  total setup + solve   : %.4e s\n\n", total_time.count());


    // if (lam_fail) {
    //     printf("FETI-DP lambda PCG failed\n");
    // }

    auto h_vars = fine_vars.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_vars, "out/plate_bddc_nl.vtk");

    delete lam_solver;
    delete ie_solver;
    delete i_solver;
    delete v_solver;
    delete bddc;

    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
}


template <typename T>
void solve_direct(int nxe, T thick, T mag) {
    // =================

    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    // constexpr bool is_nonlinear = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0;
    double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye;

    auto assembler = createPlateClampedAssembler<Assembler>(
        nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    auto &bsr_data = assembler.getBsrData();
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(10.0, true);
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto fine_vars = assembler.createVarsVec();

    // kmat assembly (not actually needed here, but is needed for the nonlinear problem))
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_assembly = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_assembly = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end_assembly - start_assembly;

    // internal loads
    ObliqueShearSineLoad<T> load;
    auto loads = assembler.createVarsVec();
    assembler.add_fext_fast(load, mag, loads);
    assembler.apply_bcs(loads);

    cublasHandle_t cublasHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // new nonlinear solver
    // ======================

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using LinearSolver = CusparseMGDirectLU<T, Assembler>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, LinearSolver>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    LinearSolver *solver = new LinearSolver(cublasHandle, cusparseHandle, assembler, kmat);
    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads, solver);
    NL *nl_solver = new NL(cublasHandle, assembler, inner_solver);


    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.05;
    nl_solver->solve(fine_vars, lambda0);
    T nl_max_disp = get_max_disp(vars);

    // important to know reduction for how NL regime we are
    // T ratio = nl_max_disp / lin_max_disp;
    // printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = assembly_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of plate geom, ndof %d : assembly time %.2e, solve time %.2e, total %.2e\n", ndof, assembly_time.count(), solve_time.count(), total);


    // print some of the data of host residual
    auto h_vars = fine_vars.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_vars, "out/plate_direct_nl.vtk");

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();


    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
}

int main(int argc, char **argv) {
    using T = double;

    // int nxe = 6, nxe_subdomain_size = 2;
    int nxe = 32, nxe_subdomain_size = 4;
    // int nxe = 256, nxe_subdomain_size = 4;
    // NOTE : full fillin with fill_level = -1, but lower fill results in less ILU(k) factor time
    // for the coarse problem..
    T omega;
    int nsmooth, fill_level;
    bool print_mem = false;

    // mid-thickness case
    // T thick = 1e-2, mag = 1e7; // leads to like 0.2 NL/LIN ratio
    
    // thinner shell case, can solve fairly stably in thin shell also
    T thick = 1e-3, mag = 1e4; // leads to 0.07 NL/LIN ratio

    std::string solver = "bddc";
    // std::string solver = "direct";
    
    // optional smoothing
    // 1) if ILU(k) here, ability to do multiple smoothing steps (Richardson)
    // omega = 0.5, nsmooth = 2, fill_level = 2;
    // or single-level (no multiple smoothing) and ILU(k)
    // omega = 1.0, nsmooth = 1, fill_level = 2;
    // 2) or if full LU fillin
    omega = 1.0, nsmooth = 1, fill_level = -1; // -1 indicates full LU fillin

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--solver") == 0) {
            if (i + 1 < argc) {
                solver = argv[++i];
            } else {
                std::cerr << "Missing value for --solver\n";
                return 1;
            }
        } else if (strcmp(arg, "--nxe") == 0) {
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
        } else if (strcmp(arg, "--mag") == 0) {
            if (i + 1 < argc) {
                mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --mag\n";
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

    if (solver == "bddc") {
        solve_bddc<T>(nxe, nxe_subdomain_size, omega, nsmooth, fill_level, thick, print_mem, mag);
    } else if (solver == "direct") {
        solve_direct<T>(nxe, thick, mag);
    }
    

    return 0;
}

#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "chrono"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "domdec/bddc_assembler.h"
#include "domdec/bddc_interface.h"
#include "domdec/domdec_pcg_wrapper.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/mitc_shell.h"
#include "element/shell/physics/isotropic_shell.h"

// multigrid / krylov imports used by BDDC interface
#include "multigrid/grid.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"
#include "multigrid/utils/fea.h"

template <typename T>
struct UniformPressure {
    __HOST_DEVICE__
    T operator()(T x, T y, T z) const { return 1.0; }
};

class LinearMITC_BDDCLU_WingSolver {
   public:
    using T = double;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    static constexpr bool has_ref_axis = false;
    static constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;

    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    using BDDC = BddcSolver<T, Assembler, VecType, BsrMat>;
    using InnerSolver = CusparseMGDirectLU<T, Assembler>;
    using InnerSolver_JUSTLU = CusparseMGDirectLU<T, Assembler, true>;
    using DUMMY = InnerSolver;
    using GRID = SingleGrid<Assembler, DUMMY, DUMMY, NONE>;
    using KIPCG = PCGSolver<T, GRID>;
    using GamPCG = MatrixFreePCGSolver<T, BDDC>;
    using BDDC_WRAPPER = DomDecKrylovWrapper<T, BDDC, GamPCG>;
    using StructSolver = TacsBDDCInterface<T, Assembler, BDDC_WRAPPER, BDDC>;

    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    using Vec = VecType<T>;
    using Mat = decltype(createBsrMat<Assembler, VecType<T>>(std::declval<Assembler &>()));

    LinearMITC_BDDCLU_WingSolver(double rhoKS = 100.0, double safety_factor = 1.5,
                                 double force = 684e3, double omega = 0.8, int level = 2,
                                 double rtol = 1e-6, int ORDER = 8, int nsmooth = 1,
                                 int ninnercyc = 1, bool print = false, int n_krylov = 50)
        : rhoKS_(rhoKS),
          safety_factor_(safety_factor),
          force_(force),
          level_(level),
          omega_(omega),
          rtol_(rtol),
          nsmooth_(nsmooth),
          print_(print) {
        printf("\n============================================================\n");
        printf("[SETUP] BEGIN LinearMITC_BDDCLU_WingSolver constructor\n");
        printf("============================================================\n");

        int already_init = 0;
        MPI_Initialized(&already_init);
        printf("[SETUP] MPI_Initialized = %d\n", already_init);
        if (!already_init) {
            printf("[SETUP] calling MPI_Init...\n");
            MPI_Init(NULL, NULL);
        }

        MPI_Comm comm = MPI_COMM_WORLD;

        num_lin_solves = 0;

        printf("[SETUP] creating cuBLAS / cuSPARSE handles...\n");
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
        print_gpu_mem("after handle creation");

        printf("[SETUP] config:\n");
        printf("         rhoKS_ = %.6f\n", rhoKS_);
        printf("         safety_factor_ = %.6f\n", safety_factor_);
        printf("         force_ = %.6e\n", force_);
        printf("         level_ = %d\n", level_);
        printf("         omega_ = %.6f\n", omega_);
        printf("         rtol_ = %.6e\n", rtol_);
        printf("         nsmooth_ = %d\n", nsmooth_);
        printf("         fill_level_ = %d\n", fill_level_);
        printf("         wraparound_ = %d\n", (int)wraparound_);
        printf("         wrapfrac_ = %.6f\n", wrapfrac_);
        printf("         nxe_subdomain_size_ = %d\n", nxe_subdomain_size_);

        // -----------------------------
        // Build finest-grid wing model
        // -----------------------------
        std::string fname = "../../examples/domdec/bddc/meshes/aob_wing_clamped_L" +
                            std::to_string(level_) + ".bdf";
        printf("[SETUP] BDF file = %s\n", fname.c_str());

        TACSMeshLoader mesh_loader{comm};
        printf("[SETUP] scanning BDF file...\n");
        mesh_loader.scanBDFFile(fname.c_str());
        printf("[SETUP] finished scanning BDF file\n");

        E_ = 70e9;
        nu_ = 0.3;
        rho_ = 2500.0;
        ys_ = 350e6;
        thick_ = 2.0 / 20.0;

        printf("[SETUP] material:\n");
        printf("         E_ = %.6e\n", E_);
        printf("         nu_ = %.6f\n", nu_);
        printf("         rho_ = %.6f\n", rho_);
        printf("         ys_ = %.6e\n", ys_);
        printf("         thick_ = %.6e\n", thick_);

        printf("[SETUP] creating assembler from BDF...\n");
        assembler = Assembler::createFromBDF(mesh_loader, Data(E_, nu_, thick_, rho_, ys_));
        printf("[SETUP] assembler created\n");

        nvars = assembler.get_num_vars();
        ndvs = assembler.get_num_dvs();
        printf("[SETUP] assembler sizes: nvars = %d, ndvs = %d\n", nvars, ndvs);

        auto &bsr_data = assembler.getBsrData();
        printf("[SETUP] computing nofill sparsity pattern...\n");
        bsr_data.compute_nofill_pattern();
        printf("[SETUP] moving BSR data to device...\n");
        assembler.moveBsrDataToDevice();
        print_gpu_mem("after assembler sparsity/device move");

        printf("[SETUP] creating kmat / vars / res / loads...\n");
        kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        loads = assembler.createVarsVec();

        printf("[SETUP] zeroing and assembling external loads...\n");
        UniformPressure<T> load;
        loads.zeroValues();
        assembler.add_fext_fast(load, force_, loads);
        assembler.apply_bcs(loads);
        print_gpu_mem("after external load assembly");

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_assembly = std::chrono::high_resolution_clock::now();
        printf("[SETUP] assembling Jacobian kmat...\n");
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_assembly = std::chrono::high_resolution_clock::now();
        assembly_time_ = std::chrono::duration<double>(end_assembly - start_assembly).count();
        printf("[SETUP] Jacobian assembly_time_ = %.6f sec\n", assembly_time_);
        print_gpu_mem("after Jacobian assembly");

        // -----------------------------
        // Build BDDC object
        // -----------------------------
        bool print_timing = false;
        bool warnings = false;
        printf("[SETUP] constructing BDDC object...\n");
        bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing, warnings);
        printf("[SETUP] BDDC object constructed\n");
        print_gpu_mem("after BDDC construction");

        int MOD_WRAPAROUND = -1;
        if (wraparound_) {
            MOD_WRAPAROUND = nxe_subdomain_size_ / 2;
            printf("[SETUP] BUILDING wraparound wing subdomains, modulo %d\n", MOD_WRAPAROUND);
        } else {
            printf("[SETUP] BUILDING non-wraparound wing subdomains\n");
        }

        printf("[SETUP] calling setup_tacs_component_subdomains...\n");
        printf("         args = (%d, %d, %d, %.6f)\n", nxe_subdomain_size_, nxe_subdomain_size_,
               MOD_WRAPAROUND, wrapfrac_);
        bddc->setup_tacs_component_subdomains(nxe_subdomain_size_, nxe_subdomain_size_,
                                              MOD_WRAPAROUND, wrapfrac_);
        printf("[SETUP] finished setup_tacs_component_subdomains\n");
        print_gpu_mem("after subdomain setup");

        // -----------------------------------
        // Factor patterns for I / IE / coarse
        // -----------------------------------
        auto &I_bsr_data = bddc->I_bsr_data;
        auto &IE_bsr_data = bddc->IE_bsr_data;
        auto &Svv_bsr_data = bddc->Svv_bsr_data;

        printf("[SETUP] starting factor-pattern setup for I / IE blocks\n");
        if (fill_level_ != -1) {
            printf("[SETUP] I block: RCM -> qorder -> ILU(%d)\n", fill_level_);
            I_bsr_data.RCM_reordering();
            I_bsr_data.qorder_reordering(0.5);
            I_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);

            printf("[SETUP] IE block: RCM -> qorder -> ILU(%d)\n", fill_level_);
            IE_bsr_data.RCM_reordering();
            IE_bsr_data.qorder_reordering(0.5);
            IE_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);
        } else {
            printf("[SETUP] I block: AMD -> full LU pattern\n");
            I_bsr_data.AMD_reordering();
            I_bsr_data.compute_full_LU_pattern(10.0);

            printf("[SETUP] IE block: AMD -> full LU pattern\n");
            IE_bsr_data.AMD_reordering();
            IE_bsr_data.compute_full_LU_pattern(10.0);
        }
        print_gpu_mem("after I / IE factor-pattern setup");

        printf("[SETUP] calling bddc->setup_matrix_sparsity()...\n");
        bddc->setup_matrix_sparsity();
        printf("[SETUP] finished bddc->setup_matrix_sparsity()\n");
        print_gpu_mem("after setup_matrix_sparsity");

        printf("[SETUP] starting coarse S_VV factor-pattern setup\n");
        if (fill_level_ != -1) {
            printf("[SETUP] S_VV block: RCM -> qorder -> ILU(%d)\n", fill_level_);
            Svv_bsr_data.RCM_reordering();
            Svv_bsr_data.qorder_reordering(0.5);
            Svv_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);
        } else {
            printf("[SETUP] S_VV block: AMD -> full LU pattern\n");
            Svv_bsr_data.AMD_reordering();
            Svv_bsr_data.compute_full_LU_pattern(10.0);
        }
        print_gpu_mem("after coarse factor-pattern setup");

        printf("[SETUP] calling bddc->setup_coarse_matrix_sparsity()...\n");
        bddc->setup_coarse_matrix_sparsity();
        printf("[SETUP] finished bddc->setup_coarse_matrix_sparsity()\n");
        print_gpu_mem("after setup_coarse_matrix_sparsity");

        printf("[SETUP] calling bddc->assemble_subdomains()...\n");
        bddc->assemble_subdomains();
        printf("[SETUP] finished bddc->assemble_subdomains()\n");
        print_gpu_mem("after assemble_subdomains");

        printf("[SETUP] adding BDDC split external load...\n");
        UniformPressure<T> press_load;
        bddc->add_subdomain_fext(press_load, force_);
        printf("[SETUP] finished add_subdomain_fext\n");
        print_gpu_mem("after add_subdomain_fext");

        printf("[SETUP] initializing IEV residual from vars...\n");
        bddc->set_IEV_residual(1.0, 0.0, vars);
        printf("[SETUP] finished set_IEV_residual\n");
        print_gpu_mem("after set_IEV_residual");

        // -----------------------------------
        // Inner solvers
        // -----------------------------------
        printf("[SETUP] constructing inner solvers...\n");
        if (fill_level_ == -1) {
            printf("[SETUP] direct/full-LU inner-solver path\n");

            ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE,
                                        omega_, nsmooth_);
            i_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler,
                                              *bddc->kmat_I, omega_, nsmooth_);
            v_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV,
                                              omega_, nsmooth_);
            print_gpu_mem("after direct inner solver allocation");

            printf("[SETUP] factoring ie_solver...\n");
            ie_solver->factor();
            print_gpu_mem("after ie_solver->factor");

            printf("[SETUP] factoring i_solver...\n");
            i_solver->factor();
            print_gpu_mem("after i_solver->factor");

            printf("[SETUP] binding direct inner solvers into BDDC...\n");
            bddc->set_inner_solvers(ie_solver, i_solver, v_solver);

            printf("[SETUP] assembling coarse problem...\n");
            bddc->assemble_coarse_problem();
            print_gpu_mem("after assemble_coarse_problem");

            printf("[SETUP] factoring v_solver...\n");
            v_solver->factor();
            print_gpu_mem("after v_solver->factor");
        } else {
            printf("[SETUP] incomplete/ILU inner-solver path\n");

            ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE,
                                        omega_, nsmooth_);
            i_solver_incomplete = new InnerSolver(cublasHandle, cusparseHandle, assembler,
                                                  *bddc->kmat_I, omega_, nsmooth_);
            v_solver_incomplete = new InnerSolver(cublasHandle, cusparseHandle, assembler,
                                                  *bddc->S_VV, omega_, nsmooth_);
            print_gpu_mem("after incomplete inner solver allocation");

            printf("[SETUP] factoring ie_solver...\n");
            ie_solver->factor();
            print_gpu_mem("after ie_solver->factor");

            printf("[SETUP] factoring i_solver_incomplete...\n");
            i_solver_incomplete->factor();
            print_gpu_mem("after i_solver_incomplete->factor");

            SolverOptions ki_opts;
            ki_opts.ncycles = 50;
            ki_opts.print = true;
            ki_opts.print_freq = 5;
            ki_opts.debug = true;
            ki_opts.rtol = 1e-15;
            ki_opts.atol = 1e-30;

            printf("[SETUP] constructing K_II grid / Krylov solver...\n");
            auto *grid = new GRID(assembler, nullptr, nullptr, *bddc->getKmatI(), loads,
                                  cublasHandle, cusparseHandle);
            i_krylov = new KIPCG(cublasHandle, cusparseHandle, grid, i_solver_incomplete, ki_opts,
                                 0, bddc->getInvars());
            print_gpu_mem("after i_krylov construction");

            printf("[SETUP] binding incomplete inner solvers into BDDC...\n");
            bddc->set_inner_solvers(ie_solver, i_solver_incomplete, v_solver_incomplete, i_krylov);

            printf("[SETUP] assembling coarse problem...\n");
            bddc->assemble_coarse_problem();
            print_gpu_mem("after assemble_coarse_problem");

            printf("[SETUP] factoring v_solver_incomplete...\n");
            v_solver_incomplete->factor();
            print_gpu_mem("after v_solver_incomplete->factor");
        }

        printf("[SETUP] allocating gamma vectors...\n");
        gam_rhs = Vec(bddc->getLambdaSize());
        gam = Vec(bddc->getLambdaSize());
        printf("[SETUP] lambda size = %d\n", bddc->getLambdaSize());

        printf("[SETUP] building lambda rhs...\n");
        bddc->get_lam_rhs(gam_rhs);
        print_gpu_mem("after get_lam_rhs");

        opts.ncycles = 200;
        opts.print = print_;
        opts.print_freq = 5;
        opts.debug = false;
        opts.rtol = rtol_;
        opts.atol = 1e-30;

        printf("[SETUP] constructing GamPCG...\n");
        gam_solver = new GamPCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);
        printf("[SETUP] constructing BDDC wrapper...\n");
        krylov_bddc_wrapper = new BDDC_WRAPPER(bddc, gam_solver);

        gam.zeroValues();
        init_gam_resid_ = gam_solver->getResidualNorm(gam_rhs, gam);
        printf("[SETUP] initial gamma residual = %.8e\n", init_gam_resid_);
        print_gpu_mem("after gamma solver setup");

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start0 = std::chrono::high_resolution_clock::now();

        printf("[SETUP] constructing StructSolver...\n");
        solver = new StructSolver(krylov_bddc_wrapper, bddc, assembler, kmat, print_);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto end0 = std::chrono::high_resolution_clock::now();
        setup_time_ = std::chrono::duration<double>(end0 - start0).count();
        printf("[SETUP] StructSolver setup_time_ = %.6f sec\n", setup_time_);
        print_gpu_mem("after StructSolver construction");

        // optimization state
        printf("[SETUP] allocating optimization state...\n");
        soln = DeviceVec<T>(nvars);

        d_loads = DeviceVec<T>(nvars);
        loads.copyValuesTo(d_loads);

        d_dvs = DeviceVec<T>(ndvs, 0.02);

        mass = std::make_unique<DMass>();
        ksfail = std::make_unique<DKSFail>(rhoKS_, safety_factor_);

        dvs_changed = true;
        first_solve = true;

        print_solver_state("end of constructor");
        print_gpu_mem("end of constructor");

        printf("============================================================\n");
        printf("[SETUP] END LinearMITC_BDDCLU_WingSolver constructor\n");
        printf("============================================================\n\n");
    }

    void set_design_variables(const std::vector<T> &dvs) {
        printf("\n[DVS] --------------------------------------------------\n");
        printf("[DVS] set_design_variables called\n");
        printf("[DVS] input size = %zu, prev size = %zu, ndvs = %d\n", dvs.size(), prev_dvs.size(),
               ndvs);
        print_gpu_mem("before set_design_variables");

        dvs_changed = (dvs.size() != prev_dvs.size());
        if (!dvs_changed) {
            for (int i = 0; i < dvs.size(); i++) {
                if (dvs[i] != prev_dvs[i]) {
                    dvs_changed = true;
                    printf("[DVS] change detected at i = %d, old = %.16e, new = %.16e\n", i,
                           prev_dvs[i], dvs[i]);
                    break;
                }
            }
        }

        printf("[DVS] dvs_changed before debug override = %d\n", (int)dvs_changed);

        dvs_changed = true;  // debug
        printf("[DVS] forcing dvs_changed = true for debug\n");

        if (dvs_changed) {
            prev_dvs = dvs;
            printf("[DVS] copying DVs to device...\n");
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            print_gpu_mem("after DV cudaMemcpy");

            printf("[DVS] calling solver->set_design_variables...\n");
            solver->set_design_variables(d_dvs);
            printf("[DVS] finished solver->set_design_variables\n");
            print_gpu_mem("after solver->set_design_variables");
        }

        print_solver_state("after set_design_variables");
        printf("[DVS] --------------------------------------------------\n\n");
    }

    void writeLoadsToVTK(const std::string &filename) {
        auto bsr_data = assembler.getBsrData();
        d_loads.permuteData(bsr_data.block_dim, bsr_data.perm);
        solver->copy_solution_in(d_loads);
        writeSolution(filename);
        d_loads.permuteData(bsr_data.block_dim, bsr_data.iperm);
    }

    void writeExplodedVTKs(const std::string int_struct_filename,
                           const std::string &upper_skin_filename,
                           const std::string lower_skin_filename) {
        auto bsr_data = assembler.getBsrData();
        soln.permuteData(bsr_data.block_dim, bsr_data.perm);
        auto h_soln = soln.createHostVec();

        explodedPrintToVTK<Assembler, HostVec<T>, INT_STRUCT>(assembler, h_soln,
                                                              int_struct_filename);
        explodedPrintToVTK<Assembler, HostVec<T>, UPPER_SKIN>(assembler, h_soln,
                                                              upper_skin_filename);
        explodedPrintToVTK<Assembler, HostVec<T>, LOWER_SKIN>(assembler, h_soln,
                                                              lower_skin_filename);
    }

    int get_num_vars() const { return nvars; }
    int get_num_dvs() const { return ndvs; }

    void writeSolution(const std::string &filename) const { solver->writeSoln(filename); }

    void solve() {
        printf("\n[SOLVE] ==================================================\n");
        printf("[SOLVE] entering solve()\n");
        print_solver_state("before solve");
        print_gpu_mem("before solve");

        if (dvs_changed) {
            printf("[SOLVE] design changed -> calling solver->solve(d_loads)\n");
            solver->solve(d_loads);
            num_lin_solves++;
            printf("[SOLVE] solver->solve finished, num_lin_solves = %d\n", num_lin_solves);
            print_gpu_mem("after solver->solve");

            printf("[SOLVE] copying solution out to cached soln\n");
            solver->copy_solution_out(soln);
            print_gpu_mem("after copy_solution_out");
        } else {
            printf("[SOLVE] design unchanged -> reloading cached solution\n");
            solver->copy_solution_in(soln);
            print_gpu_mem("after copy_solution_in");
        }

        first_solve = false;
        print_solver_state("after solve");
        printf("[SOLVE] ==================================================\n\n");
    }

    T evalFunction(const std::string &name) {
        printf("\n[FUNC] --------------------------------------------------\n");
        printf("[FUNC] evalFunction(%s)\n", name.c_str());
        print_gpu_mem("before evalFunction");

        T val = 0.0;
        if (name == "mass") {
            val = solver->evalFunction(*mass);
        } else if (name == "ksfailure") {
            val = solver->evalFunction(*ksfail);
        } else {
            throw std::invalid_argument("Unknown func");
        }

        printf("[FUNC] %s = %.16e\n", name.c_str(), val);
        print_gpu_mem("after evalFunction");
        printf("[FUNC] --------------------------------------------------\n\n");
        return val;
    }

    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        printf("\n[SENS] **************************************************\n");
        printf("[SENS] evalFunctionSens(%s)\n", name.c_str());
        print_solver_state("before evalFunctionSens");
        print_gpu_mem("before evalFunctionSens");

        double *dptr = nullptr;
        if (name == "mass") {
            printf("[SENS] calling solver->solve_adjoint(mass)\n");
            solver->solve_adjoint(*mass);
            printf("[SENS] finished solve_adjoint(mass)\n");
            dptr = mass->dv_sens.getPtr();
        } else if (name == "ksfailure") {
            printf("[SENS] calling solver->solve_adjoint(ksfailure)\n");
            solver->solve_adjoint(*ksfail);
            printf("[SENS] finished solve_adjoint(ksfailure)\n");
            num_lin_solves++;
            printf("[SENS] incremented num_lin_solves to %d\n", num_lin_solves);
            dptr = ksfail->dv_sens.getPtr();
        } else {
            throw std::invalid_argument("Unknown func");
        }

        print_gpu_mem("after solve_adjoint");

        printf("[SENS] copying DV sensitivities back to host, ndvs = %d\n", ndvs);
        CHECK_CUDA(cudaMemcpy(out_h_sens, dptr, ndvs * sizeof(T), cudaMemcpyDeviceToHost));
        print_gpu_mem("after sens cudaMemcpy");

        if (ndvs > 0) {
            int nprint = (ndvs < 5 ? ndvs : 5);
            printf("[SENS] first %d sensitivities: ", nprint);
            for (int i = 0; i < nprint; i++) {
                printf("%.6e ", out_h_sens[i]);
            }
            printf("\n");
        }

        print_solver_state("after evalFunctionSens");
        printf("[SENS] **************************************************\n\n");
    }

    int get_num_lin_solves() { return num_lin_solves; }

    void free() {
        printf("\n[FREE] ===================================================\n");
        printf("[FREE] entering free()\n");
        print_gpu_mem("at start of free");

        if (solver) {
            printf("[FREE] solver->free()\n");
            solver->free();
        }

        printf("[FREE] freeing device vectors d_loads / d_dvs\n");
        d_loads.free();
        d_dvs.free();
        print_gpu_mem("after d_loads/d_dvs free");

        printf("[FREE] deleting solver\n");
        delete solver;
        solver = nullptr;

        printf("[FREE] deleting krylov_bddc_wrapper\n");
        delete krylov_bddc_wrapper;
        krylov_bddc_wrapper = nullptr;

        printf("[FREE] deleting gam_solver\n");
        delete gam_solver;
        gam_solver = nullptr;

        printf("[FREE] deleting bddc\n");
        delete bddc;
        bddc = nullptr;

        printf("[FREE] deleting ie_solver\n");
        delete ie_solver;
        ie_solver = nullptr;

        printf("[FREE] deleting i_solver\n");
        delete i_solver;
        i_solver = nullptr;

        printf("[FREE] deleting v_solver\n");
        delete v_solver;
        v_solver = nullptr;

        printf("[FREE] deleting i_solver_incomplete\n");
        delete i_solver_incomplete;
        i_solver_incomplete = nullptr;

        printf("[FREE] deleting v_solver_incomplete\n");
        delete v_solver_incomplete;
        v_solver_incomplete = nullptr;

        printf("[FREE] deleting i_krylov\n");
        delete i_krylov;
        i_krylov = nullptr;

        print_gpu_mem("after deleting solver objects");

        if (cublasHandle) {
            printf("[FREE] destroying cublasHandle\n");
            cublasDestroy(cublasHandle);
            cublasHandle = nullptr;
        }
        if (cusparseHandle) {
            printf("[FREE] destroying cusparseHandle\n");
            cusparseDestroy(cusparseHandle);
            cusparseHandle = nullptr;
        }

        print_gpu_mem("after destroying library handles");

        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        printf("[FREE] MPI_Finalized = %d\n", mpi_finalized);
        if (!mpi_finalized) {
            printf("[FREE] calling MPI_Finalize()\n");
            MPI_Finalize();
        }

        printf("[FREE] done with free()\n");
        printf("[FREE] ===================================================\n\n");
    }

   private:
    static void print_gpu_mem(const char *label) {
        size_t free_b = 0, total_b = 0;
        auto err = cudaMemGetInfo(&free_b, &total_b);
        if (err == cudaSuccess) {
            double free_mb = static_cast<double>(free_b) / 1024.0 / 1024.0;
            double total_mb = static_cast<double>(total_b) / 1024.0 / 1024.0;
            double used_mb = total_mb - free_mb;
            printf("[GPU MEM] %s | used = %.2f MB, free = %.2f MB, total = %.2f MB\n", label,
                   used_mb, free_mb, total_mb);
        } else {
            printf("[GPU MEM] %s | cudaMemGetInfo failed\n", label);
        }
    }

    static void print_vec_info(const char *label, int n) {
        printf("[DBG] %s size = %d\n", label, n);
    }

    void print_solver_state(const char *label) {
        printf("[DBG] %s | num_lin_solves=%d, ndvs=%d, nvars=%d, dvs_changed=%d, first_solve=%d\n",
               label, num_lin_solves, ndvs, nvars, (int)dvs_changed, (int)first_solve);
    }

   private:
    // config
    double rhoKS_ = 100.0;
    double safety_factor_ = 1.5;
    double force_ = 30e3;
    int level_ = 2;
    int nxe_subdomain_size_ = 8;
    T omega_ = 1.0;
    T rtol_ = 1e-6;
    int nsmooth_ = 1;
    bool print_ = false;
    int fill_level_ = -1;
    bool wraparound_ = true;
    T wrapfrac_ = 1.0;

    // material
    double E_ = 70e9, nu_ = 0.3, rho_ = 2500.0, ys_ = 350e6, thick_ = 0.1;

    // timing / diagnostics
    double assembly_time_ = 0.0;
    double setup_time_ = 0.0;
    T init_gam_resid_ = 0.0;

    // persistent FEA objects
    Assembler assembler;
    Mat kmat;
    Vec vars, res, loads;
    Vec gam_rhs, gam;

    // handles / solvers
    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;

    BDDC *bddc = nullptr;
    InnerSolver *ie_solver = nullptr;
    InnerSolver_JUSTLU *i_solver = nullptr;
    InnerSolver_JUSTLU *v_solver = nullptr;

    InnerSolver *i_solver_incomplete = nullptr;
    InnerSolver *v_solver_incomplete = nullptr;
    KIPCG *i_krylov = nullptr;

    GamPCG *gam_solver = nullptr;
    BDDC_WRAPPER *krylov_bddc_wrapper = nullptr;
    StructSolver *solver = nullptr;

    // optimization state
    SolverOptions opts;
    std::unique_ptr<DMass> mass;
    std::unique_ptr<DKSFail> ksfail;

    int num_lin_solves = 0;
    int ndvs = 0, nvars = 0;

    DeviceVec<T> d_loads, d_dvs;
    DeviceVec<T> soln;
    std::vector<T> prev_dvs;
    bool dvs_changed = true;
    bool first_solve = true;
};
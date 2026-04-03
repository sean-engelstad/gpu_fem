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

// wing load utility from wing optimizer
// #include "loads_util.h"

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
                                 double force = 30e3, int level = 2, int nxe_subdomain_size = 8,
                                 T omega = 1.0, T rtol = 1e-6, int nsmooth = 1, bool print = false,
                                 int fill_level = -1, bool wraparound = true, T wrapfrac = 1.0)
        : rhoKS_(rhoKS),
          safety_factor_(safety_factor),
          force_(force),
          level_(level),
          nxe_subdomain_size_(nxe_subdomain_size),
          omega_(omega),
          rtol_(rtol),
          nsmooth_(nsmooth),
          print_(print),
          fill_level_(fill_level),
          wraparound_(wraparound),
          wrapfrac_(wrapfrac) {
        int already_init = 0;
        MPI_Initialized(&already_init);
        if (!already_init) {
            MPI_Init(NULL, NULL);
        }

        MPI_Comm comm = MPI_COMM_WORLD;

        num_lin_solves = 0;

        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // -----------------------------
        // Build finest-grid wing model
        // -----------------------------
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "../../examples/domdec/bddc/meshes/aob_wing_clamped_L" +
                            std::to_string(level_) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());

        E_ = 70e9;
        nu_ = 0.3;
        rho_ = 2500.0;
        ys_ = 350e6;
        thick_ = 2.0 / 20.0;  // same default as wing GMG optimizer

        assembler = Assembler::createFromBDF(mesh_loader, Data(E_, nu_, thick_, rho_, ys_));

        auto &bsr_data = assembler.getBsrData();
        bsr_data.compute_nofill_pattern();
        assembler.moveBsrDataToDevice();

        kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        loads = assembler.createVarsVec();

        // wing optimizer used lower-skin loads through addSkinLoadsToWing(...)
        int nvars_local = assembler.get_num_vars();
        // T *wing_loads = nullptr;
        // addSkinLoadsToWing<T, Basis, Assembler>(assembler, wing_loads, force_);
        // loads = assembler.createVarsVec(wing_loads);
        // assembler.apply_bcs(loads);

        // global external load vector from uniform pressure
        UniformPressure<T> load;
        loads.zeroValues();
        assembler.add_fext_fast(load, force_, loads);
        assembler.apply_bcs(loads);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_assembly = std::chrono::high_resolution_clock::now();
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_assembly = std::chrono::high_resolution_clock::now();
        assembly_time_ = std::chrono::duration<double>(end_assembly - start_assembly).count();

        // -----------------------------
        // Build BDDC object
        // -----------------------------
        bool print_timing = false;
        bool warnings = false;
        bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing, warnings);

        int MOD_WRAPAROUND = -1;
        if (wraparound_) {
            MOD_WRAPAROUND = nxe_subdomain_size_ / 2;
            printf("BUILDING wraparound wing subdomains, modulo %d\n", MOD_WRAPAROUND);
        } else {
            printf("BUILDING non-wraparound wing subdomains\n");
        }

        bddc->setup_tacs_component_subdomains(nxe_subdomain_size_, nxe_subdomain_size_,
                                              MOD_WRAPAROUND, wrapfrac_);

        // -----------------------------------
        // Factor patterns for I / IE / coarse
        // -----------------------------------
        auto &I_bsr_data = bddc->I_bsr_data;
        auto &IE_bsr_data = bddc->IE_bsr_data;
        auto &Svv_bsr_data = bddc->Svv_bsr_data;

        if (fill_level_ != -1) {
            I_bsr_data.RCM_reordering();
            I_bsr_data.qorder_reordering(0.5);
            I_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);

            IE_bsr_data.RCM_reordering();
            IE_bsr_data.qorder_reordering(0.5);
            IE_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);
        } else {
            I_bsr_data.AMD_reordering();
            I_bsr_data.compute_full_LU_pattern(10.0);

            IE_bsr_data.AMD_reordering();
            IE_bsr_data.compute_full_LU_pattern(10.0);
        }

        bddc->setup_matrix_sparsity();

        if (fill_level_ != -1) {
            Svv_bsr_data.RCM_reordering();
            Svv_bsr_data.qorder_reordering(0.5);
            Svv_bsr_data.compute_ILUk_pattern(fill_level_, 10.0);
        } else {
            Svv_bsr_data.AMD_reordering();
            Svv_bsr_data.compute_full_LU_pattern(10.0);
        }

        bddc->setup_coarse_matrix_sparsity();
        bddc->assemble_subdomains();

        // add matching external load into the IEV-split BDDC rhs
        UniformPressure<T> press_load;
        bddc->add_subdomain_fext(press_load, force_);

        // initialize IEV residual with current state
        bddc->set_IEV_residual(1.0, 0.0, vars);

        // -----------------------------------
        // Inner solvers
        // -----------------------------------
        if (fill_level_ == -1) {
            ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE,
                                        omega_, nsmooth_);
            i_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler,
                                              *bddc->kmat_I, omega_, nsmooth_);
            v_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV,
                                              omega_, nsmooth_);

            ie_solver->factor();
            i_solver->factor();

            bddc->set_inner_solvers(ie_solver, i_solver, v_solver);
            bddc->assemble_coarse_problem();
            v_solver->factor();
        } else {
            ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE,
                                        omega_, nsmooth_);
            i_solver_incomplete = new InnerSolver(cublasHandle, cusparseHandle, assembler,
                                                  *bddc->kmat_I, omega_, nsmooth_);
            v_solver_incomplete = new InnerSolver(cublasHandle, cusparseHandle, assembler,
                                                  *bddc->S_VV, omega_, nsmooth_);

            ie_solver->factor();
            i_solver_incomplete->factor();

            SolverOptions ki_opts;
            ki_opts.ncycles = 50;
            ki_opts.print = true;
            ki_opts.print_freq = 5;
            ki_opts.debug = true;
            ki_opts.rtol = 1e-15;
            ki_opts.atol = 1e-30;

            auto *grid = new GRID(assembler, nullptr, nullptr, *bddc->getKmatI(), loads,
                                  cublasHandle, cusparseHandle);
            i_krylov = new KIPCG(cublasHandle, cusparseHandle, grid, i_solver_incomplete, ki_opts,
                                 0, bddc->getInvars());

            bddc->set_inner_solvers(ie_solver, i_solver_incomplete, v_solver_incomplete, i_krylov);

            bddc->assemble_coarse_problem();
            v_solver_incomplete->factor();
        }

        gam_rhs = Vec(bddc->getLambdaSize());
        gam = Vec(bddc->getLambdaSize());
        bddc->get_lam_rhs(gam_rhs);

        opts.ncycles = 200;
        opts.print = print_;
        opts.print_freq = 5;
        opts.debug = false;
        opts.rtol = rtol_;
        opts.atol = 1e-30;

        gam_solver = new GamPCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);
        krylov_bddc_wrapper = new BDDC_WRAPPER(bddc, gam_solver);

        gam.zeroValues();
        init_gam_resid_ = gam_solver->getResidualNorm(gam_rhs, gam);
        printf("initial gamma residual = %.8e\n", init_gam_resid_);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start0 = std::chrono::high_resolution_clock::now();

        solver = new StructSolver(krylov_bddc_wrapper, bddc, assembler, kmat, print_);

        auto end0 = std::chrono::high_resolution_clock::now();
        setup_time_ = std::chrono::duration<double>(end0 - start0).count();

        // optimization state, same pattern as wing GMG optimizer
        nvars = assembler.get_num_vars();
        soln = DeviceVec<T>(nvars);

        d_loads = DeviceVec<T>(nvars);
        loads.copyValuesTo(d_loads);

        ndvs = assembler.get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, 0.02);

        mass = std::make_unique<DMass>();
        ksfail = std::make_unique<DKSFail>(rhoKS_, safety_factor_);

        dvs_changed = true;
        first_solve = true;
    }

    void set_design_variables(const std::vector<T> &dvs) {
        dvs_changed = (dvs.size() != prev_dvs.size());
        if (!dvs_changed) {
            for (int i = 0; i < dvs.size(); i++) {
                if (dvs[i] != prev_dvs[i]) {
                    dvs_changed = true;
                    break;
                }
            }
        }

        dvs_changed = true;  // debug, same as your other classes

        if (dvs_changed) {
            prev_dvs = dvs;
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            solver->set_design_variables(d_dvs);
        }
    }

    void writeLoadsToVTK(const std::string &filename) {
        // permute defect loads to VIS order
        auto bsr_data = assembler.getBsrData();
        d_loads.permuteData(bsr_data.block_dim, bsr_data.perm);
        solver->copy_solution_in(d_loads);
        writeSolution(filename);
        d_loads.permuteData(bsr_data.block_dim, bsr_data.iperm);  // un-permute
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
        if (dvs_changed) {
            printf("design changed, new solve\n");
            solver->solve(d_loads);
            num_lin_solves++;
            solver->copy_solution_out(soln);
        } else {
            printf("design didn't change, reload vals\n");
            solver->copy_solution_in(soln);
        }
    }

    T evalFunction(const std::string &name) {
        if (name == "mass") {
            return solver->evalFunction(*mass);
        } else if (name == "ksfailure") {
            return solver->evalFunction(*ksfail);
        }
        throw std::invalid_argument("Unknown func");
    }

    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        double *dptr = nullptr;
        if (name == "mass") {
            solver->solve_adjoint(*mass);
            dptr = mass->dv_sens.getPtr();
        } else if (name == "ksfailure") {
            solver->solve_adjoint(*ksfail);
            num_lin_solves++;
            dptr = ksfail->dv_sens.getPtr();
        } else {
            throw std::invalid_argument("Unknown func");
        }

        CHECK_CUDA(cudaMemcpy(out_h_sens, dptr, ndvs * sizeof(T), cudaMemcpyDeviceToHost));
    }

    int get_num_lin_solves() { return num_lin_solves; }

    void free() {
        if (solver) {
            solver->free();
        }

        d_loads.free();
        d_dvs.free();

        delete solver;
        solver = nullptr;

        delete krylov_bddc_wrapper;
        krylov_bddc_wrapper = nullptr;

        delete gam_solver;
        gam_solver = nullptr;

        delete bddc;
        bddc = nullptr;

        delete ie_solver;
        ie_solver = nullptr;

        delete i_solver;
        i_solver = nullptr;

        delete v_solver;
        v_solver = nullptr;

        delete i_solver_incomplete;
        i_solver_incomplete = nullptr;

        delete v_solver_incomplete;
        v_solver_incomplete = nullptr;

        delete i_krylov;
        i_krylov = nullptr;

        if (cublasHandle) {
            cublasDestroy(cublasHandle);
            cublasHandle = nullptr;
        }
        if (cusparseHandle) {
            cusparseDestroy(cusparseHandle);
            cusparseHandle = nullptr;
        }

        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
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
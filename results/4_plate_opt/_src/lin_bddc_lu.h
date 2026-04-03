#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
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
#include "multigrid/grid.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"
#include "multigrid/utils/fea.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

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

class Linear_BDDCLU_PlateSolver {
   public:
    using T = double;
    using Director = LinearizedRotation<T>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, false>;

    // MITC4
    using Quad = QuadLinearQuadrature<T>;
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

    Linear_BDDCLU_PlateSolver(double rhoKS = 100.0, double safety_factor = 1.5,
                              double load_mag = 100.0, T omega = 1.0, int nxe = 100,
                              int nx_comp = 5, int ny_comp = 5, double SR = 50.0, T rtol = 1e-6,
                              int ORDER = 8, double Lx = 1.0, int nsmooth = 1, int ninnercyc = 1,
                              double in_plane_frac = 0.1, bool print = false)
        : rhoKS_(rhoKS),
          safety_factor_(safety_factor),
          load_mag_(load_mag),
          omega_(omega),
          nxe_(nxe),
          nx_comp_(nx_comp),
          ny_comp_(ny_comp),
          SR_(SR),
          rtol_(rtol),
          ORDER_(ORDER),
          Lx_(Lx),
          nsmooth_(nsmooth),
          ninnercyc_(ninnercyc),
          in_plane_frac_(in_plane_frac),
          print_(print) {
        // 1) Build mesh & assembler
        assert(nxe_ % nx_comp_ == 0);
        int nye = nxe_;
        assert(nye % ny_comp_ == 0);

        num_lin_solves = 0;

        nxe_subdomain_size_ = 4;
        nxs_ = nxe_ / nxe_subdomain_size_;
        nys_ = nxe_ / nxe_subdomain_size_;

        Ly_ = Lx_;
        E_ = 70e9;
        nu_ = 0.3;
        rho_ = 2500.0;
        ys_ = 350e6;
        thick_ = 1.0 / SR_;
        mag_ = load_mag_;
        nxe_per_comp_ = nxe_ / nx_comp;
        nye_per_comp_ = nye / ny_comp;

        assembler = createPlateClampedAssembler<Assembler>(nxe_, nye, Lx_, Ly_, E_, nu_, thick_,
                                                           rho_, ys_, nxe_per_comp_, nye_per_comp_);

        auto &bsr_data = assembler.getBsrData();
        bsr_data.compute_nofill_pattern();
        assembler.moveBsrDataToDevice();

        kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        soln_host = assembler.createVarsVec();
        soln2 = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        loads = assembler.createVarsVec();

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_assembly = std::chrono::high_resolution_clock::now();
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_assembly = std::chrono::high_resolution_clock::now();
        assembly_time_ = std::chrono::duration<double>(end_assembly - start_assembly).count();

        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        print_timing_ = false;
        bool warnings_ = false;
        bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing_, warnings_);

        close_hoop_ = false;
        printf("setup structured subdomains\n");
        bddc->setup_structured_subdomains(nxe_, nye, nxs_, nys_, close_hoop_);
        printf("\tdone with setup structured subdomains\n");

        auto &I_bsr_data = bddc->I_bsr_data;
        auto &IE_bsr_data = bddc->IE_bsr_data;
        I_bsr_data.AMD_reordering();
        I_bsr_data.compute_full_LU_pattern(10.0);
        IE_bsr_data.AMD_reordering();
        IE_bsr_data.compute_full_LU_pattern(10.0);

        printf("setup matrix sparsity\n");
        bddc->setup_matrix_sparsity();
        printf("\tdone with setup matrix sparsity\n");

        auto &Svv_bsr_data = bddc->Svv_bsr_data;
        Svv_bsr_data.AMD_reordering();
        Svv_bsr_data.compute_full_LU_pattern(10.0);

        printf("setup coarse matrix subdomains\n");
        bddc->setup_coarse_matrix_sparsity();
        printf("\tdone with setup coarse matrix subdomains\n");

        printf("assemble subdomains\n");
        bddc->assemble_subdomains();
        printf("\tdone with assemble subdomains\n");

        printf("add subdomain fext\n");
        bddc->add_subdomain_fext(load_functor, mag_, in_plane_frac);
        printf("\tdone with add subdomain fext\n");

        printf("add glob vec fext\n");
        assembler.add_fext_fast(load_functor, mag_, loads, in_plane_frac);
        assembler.apply_bcs(loads);
        printf("\tdone with add glob vec fext\n");

        printf("set IEV residual\n");
        bddc->set_IEV_residual(1.0, 0.0, vars);
        printf("\tdone with set IEV residual\n");

        ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE, omega_,
                                    nsmooth_);
        i_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->kmat_I,
                                          omega_, nsmooth_);
        v_solver = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV,
                                          omega_, nsmooth_);

        ie_solver->factor();
        i_solver->factor();

        printf("set inner solvers\n");
        bddc->set_inner_solvers(ie_solver, i_solver, v_solver);
        printf("\tdone with set inner solvers\n");

        printf("assemble coarse problem\n");
        bddc->assemble_coarse_problem();
        printf("\tdone with assemble coarse problem\n");
        v_solver->factor();

        gam_rhs = Vec(bddc->getLambdaSize());
        gam = Vec(bddc->getLambdaSize());

        printf("get lam rhs\n");
        bddc->get_lam_rhs(gam_rhs);
        printf("\tdone with get lam rhs\n");

        opts.ncycles = 200;
        // opts.print = true;
        opts.print = print;
        opts.print_freq = 5;
        // opts.debug = true;
        opts.debug = false;
        opts.rtol = rtol;
        opts.atol = 1e-30;

        gam_solver = new GamPCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);
        krylov_bddc_wrapper = new BDDC_WRAPPER(bddc, gam_solver);

        gam.zeroValues();
        init_gam_resid_ = gam_solver->getResidualNorm(gam_rhs, gam);
        printf("initial gamma residual = %.8e\n", init_gam_resid_);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start0 = std::chrono::high_resolution_clock::now();

        printf("build BDDC struct solver\n");
        solver = new StructSolver(krylov_bddc_wrapper, bddc, assembler, kmat, print_);
        printf("\tdone with build BDDC struct solver\n");

        auto end0 = std::chrono::high_resolution_clock::now();
        setup_time_ = std::chrono::duration<double>(end0 - start0).count();

        nvars = assembler.get_num_vars();
        d_loads = DeviceVec<T>(nvars);
        loads.copyValuesTo(d_loads);

        int nn = assembler.get_num_nodes();
        (void)nn;  // if unused

        soln = DeviceVec<T>(nvars);
        ndvs = assembler.get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, /*initial=*/0.02);

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

        dvs_changed = true;  // debug

        if (dvs_changed) {
            prev_dvs = dvs;
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            solver->set_design_variables(d_dvs);
        }
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
        if (name == "mass")
            return solver->evalFunction(*mass);
        else if (name == "ksfailure")
            return solver->evalFunction(*ksfail);
        throw std::invalid_argument("Unknown func");
    }

    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        double *dptr;
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
        if (solver) solver->free();

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
    // user/config inputs
    double rhoKS_ = 100.0;
    double safety_factor_ = 1.5;
    double load_mag_ = 100.0;
    T omega_ = 1.0;
    int nxe_ = 100;
    int nx_comp_ = 5;
    int ny_comp_ = 5;
    double SR_ = 50.0;
    T rtol_ = 1e-6;
    int ORDER_ = 8;
    double Lx_ = 1.0;
    int nsmooth_ = 1;
    int ninnercyc_ = 1;
    double in_plane_frac_ = 0.1;
    bool print_ = false;

    // derived mesh/material/setup state
    int nxe_subdomain_size_ = 4;
    int nxs_ = 0, nys_ = 0;
    int nxe_per_comp_ = 0, nye_per_comp_ = 0;
    double Ly_ = 0.0;
    double E_ = 0.0, nu_ = 0.0, rho_ = 0.0, ys_ = 0.0;
    double thick_ = 0.0;
    double mag_ = 0.0;
    bool close_hoop_ = false;
    bool print_timing_ = false;

    // timing / diagnostics
    double assembly_time_ = 0.0;
    double setup_time_ = 0.0;
    T init_gam_resid_ = 0.0;

    // persistent FEA objects
    Assembler assembler;
    Mat kmat;
    Vec soln_host, soln2, vars, res, loads;
    Vec gam_rhs, gam;

    // handles / solvers
    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;

    BDDC *bddc = nullptr;
    InnerSolver *ie_solver = nullptr;
    InnerSolver_JUSTLU *i_solver = nullptr;
    InnerSolver_JUSTLU *v_solver = nullptr;
    GamPCG *gam_solver = nullptr;
    BDDC_WRAPPER *krylov_bddc_wrapper = nullptr;
    StructSolver *solver = nullptr;

    // loads/functions/state
    ObliqueShearSineLoad<T> load_functor;
    SolverOptions opts;

    std::unique_ptr<DMass> mass;
    std::unique_ptr<DKSFail> ksfail;

    int num_lin_solves = 0;
    int ndvs = 0, nvars = 0;

    DeviceVec<T> d_loads, d_dvs;
    std::vector<T> prev_dvs;
    bool dvs_changed = true;
    bool first_solve = true;
    DeviceVec<T> soln;
};
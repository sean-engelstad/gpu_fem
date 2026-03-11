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

#include "include/fetidp_assembler.h"
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

        return T(100.0) * sin(T(5.0) * pi * r) * cos(T(4.0) * theta);
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
    using InnerSolver = CusparseMGDirectLU<T, Assembler>;
    using LamPCG = MatrixFreePCGSolver<T, FETIDP>; // FETIDP is the operator and preconditioner

    int nxe = 4;
    int nye = nxe;
    int nxs = 2;
    int nys = 2;

    double SR = 1e3;
    double Lx = 1.0, Ly = 1.0;
    double E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye;

    auto assembler = createPlateClampedAssembler<Assembler>(
        nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    cublasHandle_t cublasHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    auto fetidp = new FETIDP(cublasHandle, cusparseHandle, assembler, kmat);

    bool close_hoop = false;
    fetidp->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);
    fetidp->setup_matrix_sparsity();

    // assemble local FETI-DP blocks
    fetidp->assemble_subdomains();

    // external load
    ObliqueShearSineLoad<T> load;
    fetidp->add_subdomain_fext(load);

    // ----------------------------------------
    // you still need actual solver objects here
    // ----------------------------------------
    //
    // Example sketch only; replace with your actual solver classes:
    
    auto *ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *fetidp->kmat_IE);
    auto *i_solver  = new InnerSolver(cublasHandle, cusparseHandle, assembler, *fetidp->kmat_I);
    // note assembler not really used in S_VV here or above classes either.. (and def not for size)
    auto *v_solver  = new InnerSolver(cublasHandle, cusparseHandle, assembler, *fetidp->S_VV);
    
    fetidp->set_inner_solvers(ie_solver, i_solver, v_solver);

    // factor each solver
    ie_solver->factor();
    i_solver->factor();
    v_solver->factor();

    // lambda rhs
    VecType<T> lam_rhs(fetidp->getLambdaSize());
    VecType<T> lam(fetidp->getLambdaSize());
    fetidp->get_lam_rhs(lam_rhs);

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 1;
    opts.debug = true;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;

    auto *lam_solver =
        new LamPCG(cublasHandle, fetidp, fetidp, opts, fetidp->getLambdaSize(), 0);

    // optional: true initial residual before solve
    lam.zeroValues();
    T init_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("initial lambda residual = %.8e\n", init_lam_resid);

    bool lam_fail = lam_solver->solve(lam_rhs, lam, true);

    T final_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("final lambda residual = %.8e\n", final_lam_resid);

    if (lam_fail) {
        printf("FETI-DP lambda PCG failed\n");
    }
    
    // done with Krylov solution, now report back

    auto soln = assembler.createVarsVec();
    fetidp->get_global_soln(lam, soln);

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/plate_fetidp.vtk");

    delete lam_solver;
    delete ie_solver;
    delete i_solver;
    delete v_solver;
    delete fetidp;

    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}

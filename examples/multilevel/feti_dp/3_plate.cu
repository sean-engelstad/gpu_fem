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

    // =====================
    // INPUTS
    // =====================

    // options for problem size and SD sizes

    // int nxe = 4, nxe_subdomain_size = 2;
    // int nxe = 16, nxe_subdomain_size = 4;
    // int nxe = 32, nxe_subdomain_size = 4;
    // int nxe = 64, nxe_subdomain_size = 4;
    int nxe = 64, nxe_subdomain_size = 8;
    // int nxe = 128, nxe_subdomain_size = 4;
    // int nxe = 128, nxe_subdomain_size = 8;
    // int nxe = 256, nxe_subdomain_size = 4;

    // =================

    int nye = nxe;
    int nxs = nxe / nxe_subdomain_size;
    int nys = nxe / nxe_subdomain_size;

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

    // // check matrices
    // auto kmat_IEV = fetidp->kmat_IEV;
    // auto kmat_IEV_vec = kmat_IEV->getVec().createHostVec();
    // T *h_kmat_IEV = kmat_IEV_vec.getPtr();
    // int IEV_nvals = kmat_IEV_vec.getSize();
    // int IEV_nnodes = IEV_nvals / 36;
    // printf("\n\nh_kmat_IEV %d\n", IEV_nvals);
    // for (int iblock = 0; iblock < IEV_nnodes; iblock++) {
    //     T *IEV_block = &h_kmat_IEV[36 * iblock];
    //     printf("block %d\n", iblock);
    //     for (int i = 0; i < 9; i++) {
    //         int ix = 2 + i % 3;
    //         int iy = 2 + i / 3;
    //         T val = IEV_block[6 * ix + iy];
    //         printf("%.4e,", val);
    //         if (i % 3 == 2) printf("\n");
    //     }
    //     printf("\n");
    // }

    // auto kmat_IE = fetidp->kmat_IE;
    // auto kmat_IE_vec = kmat_IE->getVec().createHostVec();
    // T *h_kmat_IE = kmat_IE_vec.getPtr();
    // int IE_nvals = kmat_IE_vec.getSize();
    // int IE_nnodes = IE_nvals / 36;
    // printf("\n\nh_kmat_IE %d\n", IE_nvals);
    // for (int iblock = 0; iblock < IE_nnodes; iblock++) {
    //     T *IE_block = &h_kmat_IE[36 * iblock];
    //     printf("block %d\n", iblock);
    //     for (int i = 0; i < 9; i++) {
    //         int ix = 2 + i % 3;
    //         int iy = 2 + i / 3;
    //         T val = IE_block[6 * ix + iy];
    //         printf("%.4e,", val);
    //         if (i % 3 == 2) printf("\n");
    //     }
    //     printf("\n");
    // }

    // factor each solver
    ie_solver->factor();
    i_solver->factor();

    // then assemble coarse problem (as it uses IE solver) before factoring v_solver
    fetidp->assemble_coarse_problem();
    v_solver->factor();

    // auto S_VV = fetidp->S_VV;
    // auto S_VV_vec = S_VV->getVec().createHostVec();
    // T *h_S_VV = S_VV_vec.getPtr();
    // int SVV_nvals = S_VV_vec.getSize();
    // int SVV_nnodes = SVV_nvals / 36;
    // printf("\n\nh_S_VV %d\n", SVV_nvals);
    // for (int iblock = 0; iblock < SVV_nnodes; iblock++) {
    //     T *SVV_block = &h_S_VV[36 * iblock];
    //     printf("block %d\n", iblock);
    //     for (int i = 0; i < 9; i++) {
    //         int ix = 2 + i % 3;
    //         int iy = 2 + i / 3;
    //         T val = SVV_block[6 * ix + iy];
    //         printf("%.4e,", val);
    //         if (i % 3 == 2) printf("\n");
    //     }
    //     printf("\n");
    // }
    

    // lambda rhs
    VecType<T> lam_rhs(fetidp->getLambdaSize());
    VecType<T> lam(fetidp->getLambdaSize());
    fetidp->get_lam_rhs(lam_rhs);

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    // opts.ncycles = 20;
    opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 5;
    opts.debug = true;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;

    auto *lam_solver =
        new LamPCG(cublasHandle, fetidp, fetidp, opts, fetidp->getLambdaSize(), 0);

    // optional: true initial residual before solve
    lam.zeroValues();
    T init_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("initial lambda residual = %.8e\n", init_lam_resid);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    bool lam_fail = lam_solver->solve(lam_rhs, lam, true);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end - start;

    T final_lam_resid = lam_solver->getResidualNorm(lam_rhs, lam);
    printf("final lambda residual = %.8e in %.4e sec\n", final_lam_resid, solve_time.count());

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

#pragma once
/*
CUDA BSR double-precision solver for Linear elastic unsteady solve (with Gen-alpha method)

Generalized-alpha method for structural dynamics solutions
    "A Time Integration Algorithm for Structural Dynamics With Improved Numerical Dissipation: The Generalized-α Method" by J. Chung, G.M. Hulbert 1993
    https://asmedigitalcollection.asme.org/appliedmechanics/article/60/2/371/423023/A-Time-Integration-Algorithm-for-Structural

Choosing to use the HHT-alpha variant of gen-alpha such that:
    alpha_f = 1/3 (from choice rho_infty = 1/2)
    alpha_m = 0
    beta = 0.25 * (1 - alpha_m + alpha_f)^2 = 4/9
    gamma = 1/2 - alpha_m + alpha_f = 2/3
*/

#include "linalg/vec.h"
#include "linalg/bsr_mat.h"

class LinearGenAlphaIntegrator {
  public:
    // could go back and template this later, we'll see
    using T = double;
    using Vec = DeviceVec<T>;
    using Mat = BsrMat<DeviceVec<T>>;

    LinearGenAlphaIntegrator(Mat &mass_mat, Mat &kmat, Vec &forces, int num_dof, int num_timesteps, double dt) : ndof(num_dof), num_timesteps(num_timesteps), dt(dt) {
        // intialize the disps, vel and accelerations
        assert(forces.getSize() == (num_timesteps * ndof));

        // only storing final displacements for each timestep
        // vel and accel are stored in circular buffer (cycles in mod 2 where each one is)
        disp = Vec(num_timesteps * ndof).getPtr();
        vel = Vec(2 * ndof).getPtr():
        accel = Vec(2 * ndof).getPtr();

        // util vectors
        update = Vec(ndof).getPtr();
        temp = Vec(ndof).getPtr();

        // initialize constants for HHT-alpha variant
        alpha_f = 1.0 / 3.0;
        alpha_m = 0.0;
        beta = 0.25 * (1.0 - alpha_m + alpha_f) * (1.0 - alpha_m + alpha_f);
        gamma = 0.5 - alpha_m + alpha_f;

        _initialize_LU(mass_mat, kmat, forces);
    } 

    void iterate() {
        /* first compute the residual as the rhs for new update */
        

        /* then compute the update to accel */


        /* and update the disps */
    }

    void writeToVTK() {
        // TBD
    }

    void free() {
        // TBD
    }

private:
    void _initialize_LU(Mat &mass_mat, Mat &kmat, Vec &forces) {
        // initialize CUDA LU data so we can hold the LU factor in place
        // assumes kmat and mass_mat have full LU sparsity
        
        // get perm, iperm, block_data out of the matrix
        perm = mass_mat.getPerm();
        iperm = mass_mat.getIPerm();
        block_dim = mass_mat.getBlockDim();
        mat_nnz = mass_mat.get_nnz();
        BsrData bsr_data = mass_mat.getBsrData();
        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;

        // copy device pointers out of the matrices
        Mvals = mass_mat.getPtr();
        Kvals = kmat.getPtr();

        // create handles for CUDA - cusparse and cublas
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // compute the l.c. of M and K matrix for new accel updates
        // TBD : later include C matrix for damping?
        // l.c. is MK = (1-alpha_m) * M + (1-alpha_f) * dt^2 * beta
        LUvals = Vec(mat_nnz);
        double a = 1 - alpha_m; // first we add (1-alpha_m) * M
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, Mvals, 1, LUvals, 1));
        a = (1 - alpha_f) * dt * dt * beta; // first we add (1-alpha_f) * dt * dt * beta * K
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, Kvals, 1, LUvals, 1));

        // compute the LU factor of the MK matrix stored in LUvals
        CUSPARSE::perform_ilu0_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb,
                                         nnzb, block_dim, LUvals, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

        // create a matrix descriptors for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M)); // first for mass matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_K)); // second for stiffness matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_K, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_K, CUSPARSE_INDEX_BASE_ZERO));
    }

    // timestep and gen-alpha info
    double dt;
    int num_timesteps;
    T alpha_f, alpha_m, beta, gamma;

    // general vec data
    int block_dim, mat_nnz, ndof;
    int *perm, *iperm;
    int *d_rowp, *d_cols;
    T *disp, *vel, *accel, *forces;
    T *update, *temp;

    // matrix data
    T *Mvals, *Kvals, *LUvals;

    // general CUDA data
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

    // CUDA data for LU factor of (alpha * M + beta * K)^-1 approx U^-1 L^-1
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // CUDA data for M and K matrices for mat-mul and residual estimates
    cusparseMatDescr_t descr_M = 0; // mass matrix descriptor for SpMV
    cusparseMatDescr_t descr_K = 0; // stiffness matrix descriptor for SpMV
};
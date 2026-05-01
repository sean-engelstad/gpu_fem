#include "include/poisson.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include <chrono>

int main() {
    // double BSR mv routine doesn't work (see archive)
    // so need to use float instead for BSR matrix
    using T = double;

    // case inputs
    // -----------

    constexpr bool test_mult = false;
    constexpr bool check_mat_data = false;
    int N = 16384; // 16384
    // int N = 16;
    // int n_iter = min(N, 200);
    int n_iter = 500;
    constexpr bool use_precond = true;
    constexpr bool debug = false;
    int checkpoint = -1;
    T abs_tol = 1e-8, rel_tol = 1e-8;

    // NOTE : starting with BSR matrix of block size 1 (just to demonstrate the correct cusparse methods for BSR)

    // initialize data
    // ---------------
    
    int *csr_rowp, *csr_cols;
    // int M = N;
    T *csr_vals, *rhs, *x;
    int nz = 5 * N - 4 * (int)sqrt((double)N);

    // allocate rowp, cols on host
    csr_rowp = (int*)malloc(sizeof(int) * (N + 1));
    csr_cols = (int*)malloc(sizeof(int) * nz);
    csr_vals = (T*)malloc(sizeof(T) * nz);
    x = (T*)malloc(sizeof(T) * N);
    rhs = (T*)malloc(sizeof(T) * N);

    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
        rhs[i] = 0.0;    
    }

    // initialize data
    genLaplaceCSR<T>(csr_rowp, csr_cols, csr_vals, N, nz, rhs);
    // now rhs is not zero

    // convert to BSR
    int *rowp, *cols, nnzb;
    T *vals;
    int mb = N / 2;
    int block_dim = 2;
    int block_dim2 = block_dim * block_dim;
    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals, &rowp, &cols, &vals, &nnzb);


    int nnodes = N / block_dim;
    T *diag_vals = new T[block_dim * block_dim * nnodes]; // bsr format
    getDiagValsBSR(rowp, cols, vals, block_dim, nnodes, diag_vals);

    int *diag_rowp = new int[nnodes + 1];
    int *diag_cols = new int[nnodes];
    int diag_nnzb = nnodes;
    memset(diag_rowp, 0, (nnodes+1) * sizeof(int));
    memset(diag_cols, 0, nnodes * sizeof(int));
    for (int i = 0; i < nnodes; i++) {
        diag_rowp[i+1] = i+1;
        diag_cols[i] = i;
    }

    // transfer data to the device
    int *d_rowp, *d_cols;
    int *d_diag_rowp, *d_diag_cols;
    T *d_vals, *d_diag_vals, *d_x, *d_rhs, *d_vals_ILU0;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (nnodes+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_diag_rowp, (nnodes+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_diag_cols, nnodes * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, 4 * nnzb * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_diag_vals, block_dim2 * nnodes * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));

    // copy data for the matrix over to device
    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (nnodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag_rowp, diag_rowp, (nnodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag_cols, diag_cols, nnodes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, 4 * nnzb * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag_vals, diag_vals, block_dim2 * nnodes * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs, N * sizeof(T), cudaMemcpyHostToDevice));

    // create temp vec objects
    // -----------------------

    double *d_resid, *d_tmp, *d_w;
    CHECK_CUDA(cudaMalloc((void **)&d_resid, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_tmp, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_w, N * sizeof(T)));

    auto start = std::chrono::high_resolution_clock::now();

    int nb = nnodes;
    // int max_iter = _max_iter == -1 ? n_iter : _max_iter;  // no restarts if -1 input
    int max_iter = n_iter;

    // create initial cusparse and cublas handles --------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    T a = 1.0, b = 0.0;

    // create diag inverse in LU factor form (ILU0)
    // ----------------------------------------

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                         mb, nnodes, block_dim, d_diag_vals, d_diag_rowp, d_diag_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // prelim vectors and data for PCG
    // ------------------------------------

    // make temp vecs
    // T *d_tmp = DeviceVec<T>(N).getPtr();
    // T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_p = DeviceVec<T>(N).getPtr();
    // T *d_w = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();
    bool can_print = true;
    int print_freq = 10;

    int nrestarts = max_iter / n_iter;
    int total_iter = 0;
    bool converged = false;
    for (int irestart = 0; irestart < nrestarts; irestart++) {
        // compute r_0 = b - Ax
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        // compute |r_0|
        T init_resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
        if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

        // compute z = D^-1 r_0
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnodes, &a, descr_L,
                                             d_diag_vals, d_diag_rowp, d_diag_cols, block_dim, info_L,
                                             d_resid, d_tmp, policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnodes, &a, descr_U,
                                             d_diag_vals, d_diag_rowp, d_diag_cols, block_dim, info_U, d_tmp,
                                             d_z, policy_U, pBuffer));

        // copy z => p
        CHECK_CUDA(cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // inner loop
        for (int j = 0; j < n_iter; j++, total_iter++) {
            // w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                          descrA, d_vals, d_rowp, d_cols, block_dim, d_p, &b, d_w));

            // alpha = <r,z> / <w,p>, with dot products in rz0, wp0
            T rz0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rz0));
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            T alpha = rz0 / wp0;

            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

            // r -= alpha * w
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));

            // z = M^-1 * r
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnodes, &a, descr_L,
                                                d_diag_vals, d_diag_rowp, d_diag_cols, block_dim, info_L,
                                                d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnodes, &a, descr_U,
                                                d_diag_vals, d_diag_rowp, d_diag_cols, block_dim, info_U, d_tmp,
                                                d_z, policy_U, pBuffer));

            // beta = <r_new,z_new> / <r_old,z_old> = rz1 / rz0
            T rz1;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rz1));
            T beta = rz1 / rz0;

            // p = z + beta * p
            a = beta;  // p *= beta scalar
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
            a = 1.0;  // p += z
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));

            // check for convergence
            T resid_norm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
            if (can_print && (j % print_freq == 0)) printf("PCG [%d] = %.8e\n", j, resid_norm);

            if (abs(resid_norm) < (abs_tol + init_resid_norm * rel_tol)) {
                converged = true;
                // printf("in convergence\n");
                if (can_print)
                    printf("\tPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
                break;
            }
        }                      // end of inner loop
        if (converged) break;  // exit outer loop if converged
    }

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished PCG with BSR ILU in %.4e sec\n", dt);
    }


    // -------------------------------------------
    // CHECK solution
    // -------------------------------------------

    // now copy solution back to host
    CHECK_CUDA(cudaMemcpy(x, d_x, N * sizeof(T), cudaMemcpyDeviceToHost));

    // now print solution
    if (N <= 16) {
        printf("GMRES no precond soln:");
        printVec<T>(N, x);
    }

    if (N == 16) {
        // compare against truth from python solver
        T ref[N] = {0.45454545, 0.59469697, 0.59469697, 0.45454545, 0.22348485,
            0.32954545, 0.32954545, 0.22348485, 0.10984848, 0.17045455,
            0.17045455, 0.10984848, 0.04545455, 0.0719697 , 0.0719697 ,
            0.04545455};
        T abs_diff[N], tot_abs_diff = 0.0;
        for (int i = 0; i < N; i++) {
            abs_diff[i] = abs(ref[i] - x[i]);
            tot_abs_diff += abs_diff[i];
        }
        printf("tot diff against truth = %.9e\n", tot_abs_diff);
    }

};
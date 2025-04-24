#include "_gmres_utils.h"
#include "_mat_utils.h"

int main() {
    using T = double;

    // case inputs
    // -----------

    constexpr bool test_mult = true;
    constexpr bool check_mat_data = true;
    int N = 4; // 16384
    int n_iter = min(N, 200);
    constexpr bool use_precond = true;
    constexpr bool test_precond = false;
    bool debug = false;
    T abs_tol = 1e-8, rel_tol = 1e-8;

    // NOTE : starting with BSR matrix of block size 1 (just to demonstrate the correct cusparse methods for BSR)

    // initialize data
    // ---------------
    
    int *rowp, *cols;
    // int M = N;
    T *vals, *rhs, *x;
    int nz = 5 * N - 4 * (int)sqrt((double)N);

    // allocate rowp, cols on host
    rowp = (int*)malloc(sizeof(int) * (N + 1));
    cols = (int*)malloc(sizeof(int) * nz);
    vals = (T*)malloc(sizeof(T) * nz);
    x = (T*)malloc(sizeof(T) * N);
    rhs = (T*)malloc(sizeof(T) * N);

    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
        rhs[i] = 0.0;    
    }

    // initialize data
    genLaplaceCSR<T>(rowp, cols, vals, N, nz, rhs);
    // now rhs is not zero

    // transfer data to the device
    int *d_rowp, *d_cols;
    T *d_vals, *d_x, *d_rhs, *d_vals_ILU0;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, nz * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, nz * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));

    // copy data for the matrix over to device
    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, nz * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (check_mat_data) {
        // also print out d_rowp, d_cols, d_vals to double check their values on host
        int *h_rowp = new int[5];
        int *h_cols = new int[12];
        T *h_vals = new T[12];
        
        CHECK_CUDA(cudaMemcpy(h_rowp, d_rowp, 5 * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_cols, d_cols, 12 * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_vals, d_vals, 12 * sizeof(T), cudaMemcpyDeviceToHost));
        printf("d_rowp:");
        printVec<int>(5, h_rowp);
        printf("d_cols:");
        printVec<int>(12, h_cols);
        printf("d_vals:");
        printVec<T>(12, h_vals);
    }

    // create temp vec objects
    // -----------------------

    double *d_resid, *d_tmp, *d_w;
    CHECK_CUDA(cudaMalloc((void **)&d_resid, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_tmp, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_w, N * sizeof(T)));

    // create initial cusparse and cublas objects
    // ------------------------------------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // wrap dense vectors into cusparse dense vector objects
    // -----------------------------------------------------

    cusparseDnVecDescr_t vec_rhs, vec_tmp, vec_w;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_rhs, N, d_rhs, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_tmp, N, d_tmp, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w, N, d_w, CUDA_R_64F));
    
    // create the matrix BSR objects
    // -----------------------------

    // bsr data
    int mb = N;
    int block_dim = 1;
    int nnzb = nz;


    // create ILU(0) preconditioner
    // ----------------------------


    // first test the matrix-vec product on GPU
    // ----------------------------------------
    
    if constexpr (test_mult) {
        assert(4 == N);
        T *my_vec = new T[4];
        for (int i = 0; i < 4; i++) {
            my_vec[i] = i+1;
        }

        printf("mb = %d, nnzb = %d, block_dim = %d\n", mb, nnzb, block_dim);

        T *d_my_vec;
        CHECK_CUDA(cudaMalloc((void **)&d_my_vec, 4 * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_my_vec, my_vec, 4 * sizeof(T), cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t vec_my_vec = NULL;
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_my_vec, N, d_my_vec, CUDA_R_64F));

        printf("here1\n");

        assert(block_dim > 0);
        assert(mb > 0);
        assert(nnzb > 0);
        assert(d_vals != nullptr);
        assert(d_rowp != nullptr);
        assert(d_cols != nullptr);
        assert(d_my_vec != nullptr);
        assert(d_tmp != nullptr);       


        printf("mb = %d, nnzb = %d, block_dim = %d\n", mb, nnzb, block_dim);
        printf("d_vals = %p, d_rowp = %p, d_cols = %p, d_my_vec = %p, d_tmp = %p\n",
            d_vals, d_rowp, d_cols, d_my_vec, d_tmp);


        // A * my_vec => tmp1
        // TODO: BSR MV here
        T a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, 
            CUSPARSE_DIRECTION_ROW,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            mb, mb, nnzb,
            &a, descrA,
            d_vals, d_rowp, d_cols,
            block_dim,
            d_my_vec,
            &b,
            d_tmp
        ));
        
        printf("here2\n");

        // should get this result in tmp1
        // t1=array([  1,  -3,  -7, -11])

        // copy data back to host x vec (just a test of matmult here)
        CHECK_CUDA(cudaMemcpy(x, d_tmp, N * sizeof(T), cudaMemcpyDeviceToHost));
        if (debug && N == 4) {
            printf("A*my_vec=");
            printVec<T>(4, x);
        }

        return 0;
    }


    // GMRES solve now with CSR matrix
    // -------------------------------

    printf("checkpt\n");

    // initialize GMRES data, some on host, some on GPU
    // host GMRES data
    T g[n_iter+1], cs[n_iter], ss[n_iter];
    T H[(n_iter+1)*(n_iter)];

    // GMRES device data
    T *d_Vmat, *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter+1) * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    // use single cusparseDnVecDescr_t of size N and just update it's values occasionally
    cusparseDnVecDescr_t vec_V;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_64F));
    // other strategy is to make pointer array of cusparseDnVecDescr_t vecs
    // update with void *col_ptr = static_cast<void*>(&d_Vmat[k * N]);
    //             cusparseDnVecSetValues(vec_V, col_ptr)

    // setup the ILU(0) preconditioner (if in use)
    // -------------------------------------------

    if constexpr (use_precond) {
        // setup ILU(0) preconditioner here..
    }

    if constexpr (test_precond && use_precond) {
        // print A matrix values::
        T *h_A = new T[nz];
        T *h_M = new T[nz];
        CHECK_CUDA(cudaMemcpy(h_A, d_vals, nz * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_M, d_vals_ILU0, nz * sizeof(T), cudaMemcpyDeviceToHost));
        
        printf("h_A:");
        printVec<T>(nz, h_A);
        printf("h_M:");
        printVec<T>(nz, h_M);

        // test LU solve on each unit vector..
    }


    // GMRES algorithm
    // ----------------------------

    int jj = n_iter - 1;

    // apply precond to rhs if in use
    if constexpr (use_precond) {
        // print part of initial vec_rhs
        int NPRINT = N;
        T *h_rhs = new T[NPRINT];
        CHECK_CUDA(cudaMemcpy(h_rhs, d_rhs, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
        // printf("init vec_rhs:");
        // printVec<T>(NPRINT, h_rhs);

        // zero vec_tmp
        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));
        
        // TODO : ILU solve U^-1 L^-1 * b

        // CHECK_CUDA(cudaMemcpy(h_rhs, d_rhs, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
        // printf("precond vec_rhs:");
        // printVec<T>(NPRINT, h_rhs);
    }

    // GMRES initial residual
    // assumes here d_X is 0 initially => so r0 = b - Ax = b
    T beta;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_rhs, 1, &beta));
    printf("GMRES init resid = %.5e\n", beta);
    g[0] = beta;

    // set v0 = r0 / beta (unit vec)
    T a = 1.0 / beta;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_rhs, 1, &d_Vmat[0], 1));

    T *h_v0 = new T[N];
    CHECK_CUDA(cudaMemcpy(h_v0, d_Vmat, N * sizeof(T), cudaMemcpyDeviceToHost));
    // print vec
    if (debug) {
        printf("r0:");
        printVec<T>(4, h_v0);
    }

    // then begin main GMRES iteration loop!
    for (int j = 0; j < n_iter; j++) {
        // zero this vec
        // CHECK_CUDA(cudaMemset(&d_Vmat[j * N], 0.0, N * sizeof(T)));

        // get vj and copy it into the cusparseDnVec_t
        void *vj_col = static_cast<void*>(&d_Vmat[j * N]);
        CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vj_col));

        // w = A * vj + 0 * w
        // TODO : BSR matrix multiply here MV
        // CHECK_CUSPARSE(cusparseDbsrmv(
        //     cusparseHandle, 
        //     CUSPARSE_DIRECTION_ROW,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     mb, mb, nnzb,
        //     &a, descrA,
        //     kmat_vals, dk_rowp, dk_cols,
        //     block_dim,
        //     soln_ptr,
        //     &b,
        //     resid_ptr
        // ));

        if constexpr (use_precond) {
            // TODO : U^-1 L^-1 * w => w precond solve here
        }

        // double check and print the value of 
        if (debug && N <= 16) {
            T *h_w = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("h_w[%d] pre-GS:", j);
            printVec<T>(N, h_w);

            // if (j == 0) return 0;
        }

        // now update householder matrix
        for (int i = 0 ; i < j+1; i++) {
            // get vi column
            void *vi_col = static_cast<void*>(&d_Vmat[i * N]);
            CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vi_col));

            T w_vi_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &w_vi_dot));

            // H_ij = vi dot w
            H[n_iter * i + j] = w_vi_dot;

            if (debug) printf("H[%d,%d] = %.4e\n", i, j, H[n_iter * i + j]);
            
            // w -= Hij * vi
            a = -H[n_iter * i + j];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
        }

        // double check and print the value of 
        if (debug && N <= 16) {
            T *h_w = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("h_w[%d] post GS:", j);
            printVec<T>(N, h_w);

            // if (j == 0) return 0;
        }

        // norm of w
        T nrm_w;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &nrm_w));

        // H_{j+1,j}
        H[n_iter * (j+1) + j] = nrm_w;

        // v_{j+1} column unit vec = w / H_{j+1,j}
        a = 1.0 / H[n_iter * (j+1) + j];
        CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j+1) * N], 1));
        CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j+1) * N], 1));

        if (debug && N <= 16) {
            T *h_tmp = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_tmp, &d_Vmat[(j+1) * N], N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("next V:");
            printVec<T>(N, h_tmp);

            // if (j == 0) return 0;
        }

        // then givens rotations to elim householder matrix
        for (int i = 0; i < j; i++) {
            T temp = H[i * n_iter + j];
            H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i+1) + j];
            H[n_iter * (i+1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i+1) + j];
        }

        T Hjj = H[n_iter * j + j], Hj1j = H[n_iter * (j+1) + j];
        cs[j] = Hjj / sqrt(Hjj * Hjj + Hj1j * Hj1j);
        ss[j] = cs[j] * Hj1j / Hjj;

        T g_temp = g[j];
        g[j] *= cs[j];
        g[j+1] = -ss[j] * g_temp;

        // printf("GMRES iter %d : resid %.4e\n", j, nrm_w);
        printf("GMRES iter %d : resid %.4e\n", j, abs(g[j+1]));

        if (debug) printf("j=%d, g[j]=%.4e, g[j+1]=%.4e\n", j, g[j], g[j+1]);

        H[n_iter * j + j] = cs[j] * H[n_iter * j + j] + ss[j] * H[n_iter * (j+1) + j];
        H[n_iter * (j+1) + j] = 0.0;

        if (abs(g[j+1]) < (abs_tol + beta * rel_tol)) {
            printf("GMRES converged in %d iterations to %.4e resid\n", j+1, g[j+1]);
            jj = j;
            break;
        }

        // TODO : should I use givens rotations or nrm_w for convergence? I think givens rotations
        // if (abs(nrm_w) < (abs_tol + beta * rel_tol)) {
        //     printf("GMRES converged in %d iterations to %.4e resid\n", j+1, nrm_w);
        //     jj = j;
        //     break;
        // }

    }

    // now solve Householder triangular system
    // only up to size jj+1 x jj+1 where we exited on iteration jj
    T *Hred = new T[(jj+1) * (jj+1)];
    for (int i = 0; i < jj+1; i++) {
        for (int j = 0; j < jj+1; j++) {
            // in-place transpose to be compatible with column-major cublasDtrsv later on
            Hred[(jj+1) * i + j] = H[n_iter * j + i];

            // Hred[(jj+1) * i + j] = H[n_iter * i + j];
        }
    }

    // now print out Hred
    if (debug) {
        printf("Hred:");
        printVec<T>((jj+1) * (jj+1), Hred);
        printf("gred:");
        printVec<T>((jj+1), g);
    }

    // now copy data from Hred host to device
    T *d_Hred;
    CHECK_CUDA(cudaMalloc(&d_Hred, (jj+1) * (jj+1) * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_Hred, Hred, (jj+1) * (jj+1) * sizeof(T), cudaMemcpyHostToDevice));

    // also create gred vector on the device
    T *d_gred;
    CHECK_CUDA(cudaMalloc(&d_gred, (jj+1) * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_gred, g, (jj+1) * sizeof(T), cudaMemcpyHostToDevice));

    // now solve Householder system H * y = g
    // T *d_y;
    // CHECK_CUDA(cudaMalloc(&d_y, (jj+1) * sizeof(T)));
    CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
        CUBLAS_DIAG_NON_UNIT, jj+1, d_Hred, jj+1, d_gred, 1));
    // writes g => y inplace

    // now copy back to the host
    T *h_y = new T[jj+1];
    CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj+1) * sizeof(T), cudaMemcpyDeviceToHost));

    if (debug && N <= 16) {
        printf("yred:");
        printVec<T>((jj+1), h_y);
    }

    // now compute the matrix product soln = V * y one column at a time
    // zero solution (d_x is already zero)
    for (int j = 0; j < jj+1; j++) {
        a = h_y[j];
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_x, 1));
    }

    // TODO : now compute the residual again
    // resid = b - A * x
    // resid = -1 * A * vj + 1 * w
    // T float_neg_one = -1.0;
    // CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &float_neg_one, matA,
    //     vec_rhs, &floatone, vec_tmp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
    //     d_bufferMV));

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
        printf("tot diff against truth = %.4e\n", tot_abs_diff);
    }

};
#include "include/poisson.h"
#include "include/gpuvec.h"
#include "include/gpumat.h"
#include "include/gpudiagmat.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include <chrono>

int main() {
    using T = double;

    // case inputs
    // ----------------------------

    constexpr bool test_mult = false;
    constexpr bool check_mat_data = false;
    int N = 16384; // 16384
    // int N = 64;
    // int N = 16;
    // int n_iter = min(N, 200);
    int n_iter = 500;
    constexpr bool use_precond = true;
    constexpr bool debug = false;
    int checkpoint = -1;
    T abs_tol = 1e-8, rel_tol = 1e-8;
    
    // initialize data on host
    // -----------------------------
    
    int *csr_rowp, *csr_cols;
    // int M = N;
    T *csr_vals, *h_rhs, *h_x;
    int nz = 5 * N - 4 * (int)sqrt((double)N);

    // allocate rowp, cols on host
    csr_rowp = (int*)malloc(sizeof(int) * (N + 1));
    csr_cols = (int*)malloc(sizeof(int) * nz);
    csr_vals = (T*)malloc(sizeof(T) * nz);
    h_x = (T*)malloc(sizeof(T) * N);
    h_rhs = (T*)malloc(sizeof(T) * N);

    for (int i = 0; i < N; i++) {
        h_x[i] = 0.0;
        h_rhs[i] = 0.0;    
    }

    genLaplaceCSR<T>(csr_rowp, csr_cols, csr_vals, N, nz, h_rhs);
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

    // create data on each GPU
    // ---------------------------------

    CHECK_CUDA(cudaSetDevice(0));
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    bool debug_gpu = true && N < 100; // meaning we can build it like multi-GPU (on a single GPU)
    if (debug_gpu) {
        device_count = 5;
    }
    auto x = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    auto kmat = new GPUbsrmat<T>(cublasHandle, cusparseHandle, rowp, cols, vals, 
        device_count, N, block_dim, debug_gpu);

    // setup rhs vec (TBD)
    auto rhs = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    rhs->setFromHost(h_rhs);

    // TODO : setup diagonal matrix on GPU also
    auto Dinv_mat = new GPUdiagmat<T>(cublasHandle, cusparseHandle, rowp, cols,
        vals, device_count, N, block_dim, debug_gpu);
    Dinv_mat->factor();

    // prelim vectors for PCG
    // -------------------------
    int max_iter = n_iter;
    T a = 1.0, b = 0.0; // util scalars

    auto resid = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    auto tmp = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    auto w = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    auto p = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    auto z = new GPUvec<T>(cublasHandle, device_count, N, block_dim, debug_gpu);
    bool can_print = true;
    int print_freq = 10;


    auto start = std::chrono::high_resolution_clock::now();

    int nrestarts = max_iter / n_iter;
    int total_iter = 0;
    bool converged = false;

    // compute r_0 = b - A * x
    rhs->copyTo(resid);
    a = -1.0, b = 1.0;
    kmat->mult(a, x, b, resid);

    // compute residual norm
    T init_resid_norm = resid->getResidual();
    if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

    // compute z = Dinv * r_0
    Dinv_mat->solve(resid, z);
    z->copyTo(p);

    // inner loop
    for (int j = 0; j < n_iter; j++, total_iter++) {
        // w = A * p
        a = 1.0, b = 0.0;
        kmat->mult(a, p, b, w);
        
        // alpha = <resid,z> / <w,p>, with dot products in rz0, wp0
        T rz0 = resid->dotProd(z);
        T wp0 = w->dotProd(p);
        T alpha = rz0 / wp0;

        // update x += alpha * p and resid -= alpha * w
        x->axpy(alpha, p);
        a = -alpha;
        resid->axpy(a, w);

        // z = Dinv * resid
        Dinv_mat->solve(resid, z);

        // beta = <resid, z> / rz0
        T rz1 = resid->dotProd(z);
        T beta = rz1 / rz0;

        // p = z + beta * p
        p->scale(beta);
        a = 1.0;
        p->axpy(a, z);

        // check for convergence
        T resid_norm = resid->getResidual();
        if (can_print && (j % print_freq == 0)) printf("PCG [%d] = %.8e\n", j, resid_norm);
        if (abs(resid_norm) < (abs_tol + init_resid_norm * rel_tol)) {
            converged = true;
            // printf("in convergence\n");
            if (can_print)
                printf("\tPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
            break;
        }
    }


    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished PCG with BSR ILU in %.4e sec\n", dt);
    }

    // CHECK solution (TODO)..

}
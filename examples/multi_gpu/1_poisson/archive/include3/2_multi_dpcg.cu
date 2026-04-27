#include "include/poisson.h"
#include "include/gpuvec.h"
#include "include/gpumat.h"
#include "include/gpudiagmat.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include "include/multigpu_context.h"
#include <chrono>

int main(int argc, char *argv[]) {
    using T = double;

    // case inputs
    // ----------------------------

    constexpr bool test_mult = false;
    constexpr bool check_mat_data = false;

    int N = 16384; // default

    // just type ./2_multi_dpcg.out 1048576 >> out.txt for instance
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // int N = 16777216; // need larger problem size to see good speedup of multi GPU to single GPU (for poisson)
    // int N = 1048576;
    // int N = 16384; // 16384
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

    bool debug_gpu = true && N < 100;
    auto ctx = new MultiGPUContext(debug_gpu);

    int device_count = ctx->ngpus;
    printf("#GPUs = %d\n", device_count);

    auto x = new GPUvec<T>(ctx, N, block_dim);

    auto kmat = new GPUbsrmat<T>(ctx, rowp, cols, vals, N, block_dim);

    auto rhs = new GPUvec<T>(ctx, N, block_dim);
    rhs->setFromHost(h_rhs);

    auto Dinv_mat = new GPUdiagmat<T>(ctx, rowp, cols, vals, N, block_dim);
    Dinv_mat->factor();

    auto resid = new GPUvec<T>(ctx, N, block_dim);
    auto tmp   = new GPUvec<T>(ctx, N, block_dim);
    auto w     = new GPUvec<T>(ctx, N, block_dim);
    auto p     = new GPUvec<T>(ctx, N, block_dim);
    auto z     = new GPUvec<T>(ctx, N, block_dim);
    bool can_print = true;
    // int print_freq = 10;
    int print_freq = 50;

    int max_iter = n_iter;
    T a = 1.0, b = 0.0; // util scalars

    ctx->sync();
    auto start = std::chrono::high_resolution_clock::now();

    int nrestarts = max_iter / n_iter;
    int total_iter = 0;
    bool converged = false;

    // compute r_0 = b - A * x
    rhs->copyTo(resid);
    a = -1.0, b = 1.0;
    printf("kmat mult startup\n");
    kmat->mult(a, x, b, resid);

    // compute residual norm
    printf("get resid startup\n");
    T init_resid_norm = resid->getResidual();
    if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

    // compute z = Dinv * r_0
    printf("Dinvmat solve startup\n");
    Dinv_mat->solve(resid, z);
    printf("vec copy startup\n");
    z->copyTo(p);

    // inner loop
    for (int j = 0; j < n_iter; j++, total_iter++) {
        // w = A * p
        a = 1.0, b = 0.0;
	    // printf("kmat mult on iter %d\n", j);
        kmat->mult(a, p, b, w);
        
        // alpha = <resid,z> / <w,p>, with dot products in rz0, wp0
        T rz0 = resid->dotProd(z);
        T wp0 = w->dotProd(p);
        T alpha = rz0 / wp0;
        // printf("rz0 %.2e, wp0 %.2e, alpha %.2e\n", rz0, wp0, alpha);

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
    ctx->sync();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished PCG with BSR ILU in %.4e sec\n", dt);
    }

    // CHECK solution (TODO)..


    delete x;
    delete kmat;
    delete rhs;
    delete Dinv_mat;
    delete resid;
    delete tmp;
    delete w;
    delete p;
    delete z;

    delete ctx;
}

#include "include/poisson.h"
#include "include/gpuvec.h"
#include "include/gpumat.h"
#include "include/gpudiagmat.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include "include/multigpu_context.h"
#include "include/gpu_pcg.h"
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

    bool can_print = true;
    int print_freq = 50;

    auto pcg = new GPU_PCG<T, GPUdiagmat<T>>(ctx, kmat, Dinv_mat, N, block_dim);

    int pcg_iters = pcg->solve(rhs, x,
                            n_iter, abs_tol, rel_tol,
                            print_freq, can_print);

    if (pcg_iters < 0) {
        printf("PCG did not converge in %d iterations\n", -pcg_iters);
    }

    // CHECK solution (TODO)..

    delete x;
    delete kmat;
    delete rhs;
    delete Dinv_mat;
    delete ctx;
}

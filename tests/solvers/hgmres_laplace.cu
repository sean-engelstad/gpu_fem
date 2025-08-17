#include "utils/_laplace_utils.h"
#include "linalg/_linalg.h"
#include "solvers/linear_static_cusparse.h"
#include "../test_commons.h"

int main() {
    // double BSR mv routine doesn't work (see archive)
    // so need to use float instead for BSR matrix
    using T = double;

    // true solution for N=64 from python solver
    T true_soln[] = {0.48629073, 0.66913706, 0.74475527, 0.77346185, 0.77346185,
        0.74475527, 0.66913706, 0.48629073, 0.27602587, 0.44550224,
        0.53642216, 0.57563028, 0.57563028, 0.53642216, 0.44550224,
        0.27602587, 0.17231052, 0.30042386, 0.37980085, 0.41700683,
        0.41700683, 0.37980085, 0.30042386, 0.17231052, 0.11279234,
        0.20408183, 0.26535057, 0.29558935, 0.29558935, 0.26535057,
        0.20408183, 0.11279234, 0.07477702, 0.13776056, 0.18193023,
        0.20441065, 0.20441065, 0.18193024, 0.13776056, 0.07477702,
        0.04855519, 0.09025316, 0.12019915, 0.13571237, 0.13571237,
        0.12019915, 0.09025316, 0.04855519, 0.02919055, 0.05449776,
        0.07290081, 0.08252732, 0.08252732, 0.07290082, 0.05449776,
        0.02919055, 0.01370927, 0.02564651, 0.03437903, 0.03896878,
        0.03896878, 0.03437903, 0.02564651, 0.01370927};

    // case inputs
    // -----------
    // int N = 16384;
    int N = 64; // 16384
    int n_iter = min(N, 200);
    int max_iter = min(N, 200);
    constexpr bool use_precond = true;
    T abs_tol = 1e-7, rel_tol = 1e-8;
    bool print = true;

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
    int mb = N /2;
    int block_dim = 2;
    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals, &rowp, &cols, &vals, &nnzb);

    // transfer data to the device
    int *d_rowp, *d_cols;
    T *d_vals, *d_x, *d_rhs;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N/2+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, 4 * nnzb * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));

    // copy data for the matrix over to device
    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (N/2+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, 4 * nnzb * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs, N * sizeof(T), cudaMemcpyHostToDevice));

    // make perm and iperm on device
    int *perm, *iperm;
    perm = new int[mb];
    iperm = new int[mb];
    for (int i = 0; i < mb; i++) {
        perm[i] = i;
        iperm[i] = i;
    }
    int *d_perm = HostVec<int>(mb, perm).createDeviceVec().getPtr();
    int *d_iperm = HostVec<int>(mb, iperm).createDeviceVec().getPtr();

    // compare these results with cuda_examples/gmres/3_gmres_Dbsr.cu
    // now make BSRData and BSRMat objects here
    auto bsr_data = BsrData(mb, block_dim, nnzb, d_rowp, d_cols, d_perm, d_iperm, false);
    auto vals_vec = DeviceVec<T>(4 * nnzb, d_vals);
    auto mat = BsrMat<DeviceVec<T>>(bsr_data, vals_vec);
    auto rhs_vec = DeviceVec<T>(N, d_rhs);
    auto soln_vec = DeviceVec<T>(N, d_x);

    bool debug = true;
    int print_freq = 3;

    // now call GMRES algorithm
    CUSPARSE::HGMRES_solve<T, use_precond>(mat, rhs_vec, soln_vec, n_iter, max_iter, abs_tol, rel_tol, print, debug, print_freq);

    // now check soln error?
    T max_rel_err = rel_err(soln_vec, true_soln);
    bool passed = EXPECT_VEC_NEAR(soln_vec, true_soln, 1e-6);
    printTestReport("GMRES N=16 Laplace test", passed, max_rel_err);
}
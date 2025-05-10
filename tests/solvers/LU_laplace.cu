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
    int N = 64; // 16384
    int n_iter = min(N, 200);

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
    T *d_x, *d_rhs;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N/2+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nnzb * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void **)&d_vals, 4 * nnzb * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));

    // copy data for the matrix over to device
    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (N/2+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nnzb * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_vals, vals, 4 * nnzb * sizeof(T), cudaMemcpyHostToDevice));
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

    // deep copy rowp, cols first
    int *orig_rowp = new int[mb + 1];
    int *orig_cols = new int[nnzb];
    for (int i = 0; i < mb + 1; i++) {
        orig_rowp[i] = rowp[i];
    }
    for (int i = 0; i < nnzb; i++) {
        orig_cols[i] = cols[i];
    }

    // compare these results with cuda_examples/gmres/3_gmres_Dbsr.cu
    // now make BSRData and BSRMat objects here
    auto bsr_data = BsrData(mb, block_dim, nnzb, rowp, cols, perm, iperm, false);
    double fill_factor = 10.0;
    bool print = false;
    // printf("here\n");
    bsr_data.compute_full_LU_pattern(fill_factor, print);
    // printf("here2\n");
    HostVec<T> filled_vals(bsr_data.nnzb * 4);

    // printf("orig rowp:");
    // printVec<int>(mb + 1, orig_rowp);
    // printf("orig cols:");
    // printVec<int>(nnzb, orig_cols);

    // printf("filled in rowp:");
    // printVec<int>(bsr_data.nnodes + 1, bsr_data.rowp);
    // printf("filled in cols:");
    // printVec<int>(bsr_data.nnzb, bsr_data.cols);

    // now need to copy over the d_vals to the new filled in pattern..
    for (int i = 0; i < bsr_data.nnodes; i++) {
        // loop over current connectivity
        // printf("i = %d\n", i);
        for (int j = bsr_data.rowp[i]; j < bsr_data.rowp[i+1]; j++) {
            int bcol = bsr_data.cols[j];
            // check if in the other connectivity
            int bcol2 = -1;
            int jj2 = -1;
            for (int j2 = orig_rowp[i]; j2 < orig_rowp[i+1]; j2++) {
                if (orig_cols[j2] == bcol) {
                    bcol2 = orig_cols[j2];
                    jj2 = j2;
                    break;
                }
            }
            if (bcol2 != -1) {
                for (int ii = 0; ii < 4; ii++) {
                    // loop inside the 2x2 block
                    // printf("matching vals at bcol %d with %d and val %.4e\n", j, jj2, vals[4 * jj2 + ii]);
                    filled_vals[4 * j + ii] = vals[4 * jj2 + ii];
                }
            }
        }
    }
    // printf("filled_vals:");
    // printVec<T>(filled_vals.getSize(), filled_vals.getPtr());
    auto d_vals = filled_vals.createDeviceVec();
    auto d_bsr_data = bsr_data.createDeviceBsrData();
    auto mat = BsrMat<DeviceVec<T>>(d_bsr_data, d_vals);
    auto rhs_vec = DeviceVec<T>(N, d_rhs);
    auto soln_vec = DeviceVec<T>(N, d_x);

    // now call direct LU solve algorithm
    CUSPARSE::direct_LU_solve<T>(mat, rhs_vec, soln_vec);

    // auto h_soln = soln_vec.createHostVec();
    // printf("h_soln:");
    // printVec<T>(h_soln.getSize(), h_soln.getPtr());

    // now check soln error?
    T max_rel_err = rel_err(soln_vec, true_soln);
    bool passed = EXPECT_VEC_NEAR(soln_vec, true_soln);
    printTestReport("Direct-LU N=16 Laplace test", passed, max_rel_err);
}
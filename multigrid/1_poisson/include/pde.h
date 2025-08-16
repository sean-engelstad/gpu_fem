// gpu_fem utils
#include <thrust/device_vector.h>

#include "linalg/vec.h"
#include "pde.cuh"

template <typename T>
class PoissonSolver {
    /* a CUDA GPU solver for the 2D poisson problem with multigrid capabilities as well.. */
    using I = unsigned long long int;  // large int datatype
    static constexpr int nodes_per_elem = 4;

   public:
    PoissonSolver(int nxe_) : nxe(nxe_) {
        nx = nxe + 1;
        N = nx * nx;
        nelems = nxe * nxe;
        dx = 1.0 / nx;  // element dx or mesh size metric

        // compute the nnz pattern of the LHS
        buildCSRPattern();
    }

    void buildCSRPattern() {
        /* compute the CSR matrix nnz pattern on the GPU (with nofill) */
        // TODO : can lattern extend some of these methods to compute nnz patterns on GPU with TACS

        // 1) get total row counts (non-unique)
        auto d_csr_NUNIQ_row_counts = DeviceVec<int>(N);
        int nblocks = (nelems + 31) / 32;
        dim3 block(32);
        dim3 grid(nblocks, 16);
        k_csrNUNIQRowCounts<I><<<grid, block>>>(nxe, nelems, d_csr_NUNIQ_row_counts.getPtr());

        // DEBUG
        // printf("h_rowCounts (not unique): ");
        // int *h_rowCounts1 = d_csr_NUNIQ_row_counts.createHostVec().getPtr();
        // printVec<int>(N, h_rowCounts1);

        // 2) get max num row counts (non-unique)
        auto d_maxRowCount1 = DeviceVec<int>(1);
        dim3 grid3((N + 31) / 32);
        k_csrMaxRowCount<<<grid3, block>>>(N, d_csr_NUNIQ_row_counts.getPtr(),
                                           d_maxRowCount1.getPtr());
        int maxRowCount1 = d_maxRowCount1.createHostVec().getPtr()[0];
        d_maxRowCount1.free();
        d_csr_NUNIQ_row_counts.free();

        // printf("max row count (non-unique) = %d\n", maxRowCount1);

        // 3) get row counts (unique)
        auto d_csr_row_counts = DeviceVec<int>(N);
        auto d_minConnectedNode_in = DeviceVec<int>(N);
        auto d_minConnectedNode_out = DeviceVec<int>(N);
        cudaMemset(d_minConnectedNode_in.getPtr(), -1, N * sizeof(int));
        auto d_csr_nnz = DeviceVec<I>(1);
        for (int i_minNode = 0; i_minNode < maxRowCount1; i_minNode++) {
            // reset to N so min actually works
            cudaMemset(d_minConnectedNode_out.getPtr(), N, N * sizeof(int));

            // get the ith min node connected to this nodal row (loops over each element..)
            k_getMinConnectedNode<<<grid, block>>>(nxe, N, nelems, d_minConnectedNode_in.getPtr(),
                                                   d_minConnectedNode_out.getPtr());

            // add this ith min connected node to the row (until no more to add)
            k_uniqueRowCount<I><<<grid3, block>>>(N, d_minConnectedNode_out.getPtr(),
                                                  d_csr_row_counts.getPtr(), d_csr_nnz.getPtr());

            // eliminates race conditiions with two arrays
            cudaMemcpy(d_minConnectedNode_in.getPtr(), d_minConnectedNode_out.getPtr(),
                       N * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        csr_nnz = d_csr_nnz.createHostVec().getPtr()[0];
        d_csr_nnz.free();

        // // DEBUG
        // printf("h_rowCounts (unique): ");
        // int *h_rowCounts2 = d_csr_row_counts.createHostVec().getPtr();
        // printVec<int>(N, h_rowCounts2);

        // 4) go from row counts to row ptr
        d_csr_rowp = DeviceVec<int>(N + 1);
        thrustPrefixScan(d_csr_row_counts.getPtr());
        d_csr_row_counts.free();

        // // DEBUG
        // printf("h_rowp: ");
        // int *h_rowp = d_csr_rowp.createHostVec().getPtr();
        // printVec<int>(N + 1, h_rowp);

        // 5) get max row count (unique row counts)
        auto d_maxRowCount2 = DeviceVec<int>(1);
        k_csrMaxRowCount<<<grid3, block>>>(N, d_csr_row_counts.getPtr(), d_maxRowCount2.getPtr());
        int maxRowCount2 = d_maxRowCount2.createHostVec().getPtr()[0];
        d_maxRowCount2.free();
        // printf("max row count unique = %d\n", maxRowCount2);
        // printf("csr nnz = %d\n", (int)csr_nnz);

        // 6) get colptr
        d_csr_cols = DeviceVec<int>(csr_nnz);
        cudaMemset(d_minConnectedNode_in.getPtr(), -1, N * sizeof(int));  // reset
        for (int i_minNode = 0; i_minNode < maxRowCount2; i_minNode++) {
            // reset to N so min actually works
            cudaMemset(d_minConnectedNode_out.getPtr(), N, N * sizeof(int));

            // get the ith min node connected to this nodal row (loops over each element..)
            k_getMinConnectedNode<<<grid, block>>>(nxe, N, nelems, d_minConnectedNode_in.getPtr(),
                                                   d_minConnectedNode_out.getPtr());

            // add this ith min connected node to the row (until no more to add)
            k_addMinConnectedNode<<<grid3, block>>>(i_minNode, N, d_minConnectedNode_out.getPtr(),
                                                    d_csr_rowp.getPtr(), d_csr_cols.getPtr());

            // eliminates race conditiions with two arrays
            cudaMemcpy(d_minConnectedNode_in.getPtr(), d_minConnectedNode_out.getPtr(),
                       N * sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // // DEBUG
        // printf("h_cols: ");
        // int *h_cols = d_csr_cols.createHostVec().getPtr();
        // printVec<int>(csr_nnz, h_cols);
    }

    // public data
    int nxe, nx, N, nelems;
    I csr_nnz;
    DeviceVec<int> d_csr_rowp, d_csr_cols, d_node_min_elems;
    T dx;
    DeviceVec<T> d_vals;

   private:
    /* other helper methods (non device) */
    void thrustPrefixScan(int *_d_csr_row_counts) {
        // use thrust here to do prefix scan, there's also some interesting steps to get efficient
        // Bleloch Scan kernel on GPU (with shared mem + bank conflicts)
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // see this on how to handle CUDA shared mem banks,
        // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/

        thrust::device_vector<int> d_counts(N);
        thrust::copy_n(thrust::device_pointer_cast(_d_csr_row_counts), N, d_counts.begin());
        thrust::device_vector<int> d_rowp2(N + 1);
        d_rowp2[0] = 0;
        thrust::inclusive_scan(d_counts.begin(), d_counts.end(), d_rowp2.begin() + 1);

        cudaMemcpy(d_csr_rowp.getPtr(), thrust::raw_pointer_cast(d_rowp2.data()),
                   (N + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    }
};
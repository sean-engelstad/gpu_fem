#include "include/poisson.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <algorithm>

template <typename T>
struct GhostCopy {
    int src_gpu;
    int src_local_node;
    int dst_ext_node;
};

template <typename T>
struct GPUData {
    int dev = 0;

    int row_start_node = 0;
    int row_end_node = 0;
    int local_nnodes = 0;
    int nghost = 0;
    int local_N = 0;
    int ext_N = 0;
    int nnzb_local = 0;

    std::vector<int> ghost_global_nodes;
    std::vector<GhostCopy<T>> ghost_copies;

    int *h_rowp = nullptr;
    int *h_cols = nullptr;
    T *h_vals = nullptr;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    T *d_vals = nullptr;

    T *d_x_owned = nullptr;
    T *d_x_ext = nullptr;
    T *d_y_owned = nullptr;

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseMatDescr_t descrA = nullptr;
};

int ownerOfNode(int node, const std::vector<int> &starts, const std::vector<int> &ends) {
    for (int g = 0; g < (int)starts.size(); g++) {
        if (node >= starts[g] && node < ends[g]) return g;
    }
    return -1;
}

template <typename T>
void extractLocalBSRRowsWithGhosts(
    GPUData<T> &gd,
    const std::vector<int> &starts,
    const std::vector<int> &ends,
    const int *rowp,
    const int *cols,
    const T *vals,
    int block_dim
) {
    int block_dim2 = block_dim * block_dim;

    int row_start = gd.row_start_node;
    int row_end = gd.row_end_node;
    int local_nrows = row_end - row_start;

    int start_nnz = rowp[row_start];
    int end_nnz = rowp[row_end];
    int nnzb_local = end_nnz - start_nnz;

    gd.h_rowp = (int*)malloc((local_nrows + 1) * sizeof(int));
    gd.h_cols = (int*)malloc(nnzb_local * sizeof(int));
    gd.h_vals = (T*)malloc(nnzb_local * block_dim2 * sizeof(T));

    gd.nnzb_local = nnzb_local;

    std::unordered_map<int, int> ghost_map;

    gd.h_rowp[0] = 0;
    for (int i = 0; i < local_nrows; i++) {
        gd.h_rowp[i + 1] = rowp[row_start + i + 1] - start_nnz;
    }

    for (int k = 0; k < nnzb_local; k++) {
        int global_col = cols[start_nnz + k];

        if (global_col >= row_start && global_col < row_end) {
            gd.h_cols[k] = global_col - row_start;
        } else {
            auto it = ghost_map.find(global_col);

            if (it == ghost_map.end()) {
                int ghost_id = (int)gd.ghost_global_nodes.size();
                ghost_map[global_col] = ghost_id;
                gd.ghost_global_nodes.push_back(global_col);

                int src_gpu = ownerOfNode(global_col, starts, ends);
                if (src_gpu < 0) {
                    printf("ERROR: could not find owner for node %d\n", global_col);
                    exit(1);
                }

                GhostCopy<T> cp;
                cp.src_gpu = src_gpu;
                cp.src_local_node = global_col - starts[src_gpu];
                cp.dst_ext_node = local_nrows + ghost_id;
                gd.ghost_copies.push_back(cp);

                gd.h_cols[k] = local_nrows + ghost_id;
            } else {
                gd.h_cols[k] = local_nrows + it->second;
            }
        }
    }

    for (int k = 0; k < nnzb_local * block_dim2; k++) {
        gd.h_vals[k] = vals[start_nnz * block_dim2 + k];
    }

    gd.nghost = (int)gd.ghost_global_nodes.size();
    gd.local_nnodes = local_nrows;
    gd.local_N = gd.local_nnodes * block_dim;
    gd.ext_N = (gd.local_nnodes + gd.nghost) * block_dim;
}

template <typename T>
void setupGhostedMultiGPU(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int N,
    int nnodes,
    int block_dim,
    int block_dim2,
    const int *rowp,
    const int *cols,
    const T *vals
) {
    gpus.resize(ngpu);

    std::vector<int> starts(ngpu), ends(ngpu);
    for (int g = 0; g < ngpu; g++) {
        starts[g] = (g * nnodes) / ngpu;
        ends[g] = ((g + 1) * nnodes) / ngpu;
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(g));

        gpus[g].dev = g;
        gpus[g].row_start_node = starts[g];
        gpus[g].row_end_node = ends[g];

        extractLocalBSRRowsWithGhosts<T>(
            gpus[g], starts, ends, rowp, cols, vals, block_dim
        );

        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_rowp,
                              (gpus[g].local_nnodes + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_cols,
                              gpus[g].nnzb_local * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_vals,
                              gpus[g].nnzb_local * block_dim2 * sizeof(T)));

        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_x_owned,
                              gpus[g].local_N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_x_ext,
                              gpus[g].ext_N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_y_owned,
                              gpus[g].local_N * sizeof(T)));

        CHECK_CUDA(cudaMemcpy(gpus[g].d_rowp, gpus[g].h_rowp,
                              (gpus[g].local_nnodes + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpus[g].d_cols, gpus[g].h_cols,
                              gpus[g].nnzb_local * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpus[g].d_vals, gpus[g].h_vals,
                              gpus[g].nnzb_local * block_dim2 * sizeof(T),
                              cudaMemcpyHostToDevice));

        CHECK_CUSPARSE(cusparseCreate(&gpus[g].cusparseHandle));
        CHECK_CUSPARSE(cusparseCreateMatDescr(&gpus[g].descrA));
        CHECK_CUSPARSE(cusparseSetMatType(gpus[g].descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(gpus[g].descrA, CUSPARSE_INDEX_BASE_ZERO));

        printf("GPU %d owns block rows [%d, %d), local = %d, ghosts = %d, local nnzb = %d\n",
               g,
               gpus[g].row_start_node,
               gpus[g].row_end_node,
               gpus[g].local_nnodes,
               gpus[g].nghost,
               gpus[g].nnzb_local);
    }
}

template <typename T>
void scatterOwnedXToGPUs(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int block_dim,
    const T *h_x
) {
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        int scalar_start = gpus[g].row_start_node * block_dim;

        CHECK_CUDA(cudaMemcpy(
            gpus[g].d_x_owned,
            &h_x[scalar_start],
            gpus[g].local_N * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }
}

template <typename T>
void gatherOwnedYFromGPUs(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int block_dim,
    T *h_y
) {
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        int scalar_start = gpus[g].row_start_node * block_dim;

        CHECK_CUDA(cudaMemcpy(
            &h_y[scalar_start],
            gpus[g].d_y_owned,
            gpus[g].local_N * sizeof(T),
            cudaMemcpyDeviceToHost
        ));
    }
}

template <typename T>
void exchangeGhosts(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int block_dim
) {
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        CHECK_CUDA(cudaMemcpy(
            gpus[g].d_x_ext,
            gpus[g].d_x_owned,
            gpus[g].local_N * sizeof(T),
            cudaMemcpyDeviceToDevice
        ));
    }

    for (int dst = 0; dst < ngpu; dst++) {
        for (auto &cp : gpus[dst].ghost_copies) {
            int src = cp.src_gpu;

            CHECK_CUDA(cudaMemcpyPeer(
                gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                gpus[dst].dev,
                gpus[src].d_x_owned + cp.src_local_node * block_dim,
                gpus[src].dev,
                block_dim * sizeof(T)
            ));
        }
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

template <typename T>
void multiGpuGhostedBSRMatVec(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int block_dim
) {
    T alpha = 1.0;
    T beta = 0.0;

    exchangeGhosts<T>(gpus, ngpu, block_dim);

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        CHECK_CUSPARSE(cusparseDbsrmv(
            gpus[g].cusparseHandle,
            CUSPARSE_DIRECTION_ROW,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            gpus[g].local_nnodes,
            gpus[g].local_nnodes + gpus[g].nghost,
            gpus[g].nnzb_local,
            &alpha,
            gpus[g].descrA,
            gpus[g].d_vals,
            gpus[g].d_rowp,
            gpus[g].d_cols,
            block_dim,
            gpus[g].d_x_ext,
            &beta,
            gpus[g].d_y_owned
        ));
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

template <typename T>
void cleanupMultiGPU(std::vector<GPUData<T>> &gpus) {
    for (auto &gd : gpus) {
        CHECK_CUDA(cudaSetDevice(gd.dev));

        if (gd.d_rowp) cudaFree(gd.d_rowp);
        if (gd.d_cols) cudaFree(gd.d_cols);
        if (gd.d_vals) cudaFree(gd.d_vals);
        if (gd.d_x_owned) cudaFree(gd.d_x_owned);
        if (gd.d_x_ext) cudaFree(gd.d_x_ext);
        if (gd.d_y_owned) cudaFree(gd.d_y_owned);

        if (gd.descrA) cusparseDestroyMatDescr(gd.descrA);
        if (gd.cusparseHandle) cusparseDestroy(gd.cusparseHandle);

        if (gd.h_rowp) free(gd.h_rowp);
        if (gd.h_cols) free(gd.h_cols);
        if (gd.h_vals) free(gd.h_vals);
    }
}

int main() {
    using T = double;

    int N = 16384;
    int block_dim = 2;
    int block_dim2 = block_dim * block_dim;

    int nz = 5 * N - 4 * (int)sqrt((double)N);

    int *csr_rowp = (int*)malloc(sizeof(int) * (N + 1));
    int *csr_cols = (int*)malloc(sizeof(int) * nz);
    T *csr_vals = (T*)malloc(sizeof(T) * nz);
    T *rhs = (T*)malloc(sizeof(T) * N);
    T *x = (T*)malloc(sizeof(T) * N);

    for (int i = 0; i < N; i++) {
        rhs[i] = 0.0;
        x[i] = sin(0.001 * i) + 0.01 * cos(0.07 * i);
    }

    genLaplaceCSR<T>(csr_rowp, csr_cols, csr_vals, N, nz, rhs);

    int *rowp = nullptr;
    int *cols = nullptr;
    T *vals = nullptr;
    int nnzb = 0;

    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals,
                &rowp, &cols, &vals, &nnzb);

    int nnodes = N / block_dim;
    int mb = nnodes;

    printf("N = %d, nnodes = %d, CSR nz = %d, BSR nnzb = %d\n",
           N, nnodes, nz, nnzb);

    // -------------------------------------------------------
    // Single GPU reference
    // -------------------------------------------------------

    CHECK_CUDA(cudaSetDevice(0));

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    T *d_vals = nullptr;
    T *d_x = nullptr;
    T *d_y = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowp, (nnodes + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_cols, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnzb * block_dim2 * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (nnodes + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nnzb * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, nnzb * block_dim2 * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T),
                          cudaMemcpyHostToDevice));

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseMatDescr_t descrA = nullptr;

    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    T alpha = 1.0;
    T beta = 0.0;

    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle,
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        mb,
        mb,
        nnzb,
        &alpha,
        descrA,
        d_vals,
        d_rowp,
        d_cols,
        block_dim,
        d_x,
        &beta,
        d_y
    ));

    T *y_single = (T*)malloc(sizeof(T) * N);
    T *y_multi = (T*)malloc(sizeof(T) * N);

    CHECK_CUDA(cudaMemcpy(y_single, d_y, N * sizeof(T),
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------
    // Ghosted multi GPU
    // -------------------------------------------------------

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    int requested_gpus = 4;
    int ngpu = std::min(requested_gpus, device_count);

    printf("Using %d GPU(s) out of %d available.\n", ngpu, device_count);

    std::vector<GPUData<T>> gpus;

    setupGhostedMultiGPU<T>(
        gpus,
        ngpu,
        N,
        nnodes,
        block_dim,
        block_dim2,
        rowp,
        cols,
        vals
    );

    scatterOwnedXToGPUs<T>(gpus, ngpu, block_dim, x);

    multiGpuGhostedBSRMatVec<T>(gpus, ngpu, block_dim);

    gatherOwnedYFromGPUs<T>(gpus, ngpu, block_dim, y_multi);

    // -------------------------------------------------------
    // Error check
    // -------------------------------------------------------

    double diff2 = 0.0;
    double norm2 = 0.0;
    double max_abs = 0.0;

    for (int i = 0; i < N; i++) {
        double diff = (double)y_multi[i] - (double)y_single[i];
        diff2 += diff * diff;
        norm2 += (double)y_single[i] * (double)y_single[i];
        max_abs = std::max(max_abs, std::abs(diff));
    }

    double rel_err = sqrt(diff2 / norm2);

    printf("\nGhosted multi-GPU BSR SpMV check:\n");
    printf("  rel L2 error = %.15e\n", rel_err);
    printf("  max abs err  = %.15e\n", max_abs);

    if (rel_err < 1e-12) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    cleanupMultiGPU<T>(gpus);

    CHECK_CUDA(cudaSetDevice(0));
    cudaFree(d_rowp);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);

    free(csr_rowp);
    free(csr_cols);
    free(csr_vals);
    free(rhs);
    free(x);
    free(y_single);
    free(y_multi);

    return 0;
}
#include "_utils.h"
#include "../../../include/linalg/vec.h"
#include "../../../include/solvers/linear_static/_cusparse_utils.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

template <typename T>
struct GPUData {
    int dev = 0;

    int row_start_node = 0;
    int row_end_node = 0;
    int local_nnodes = 0;
    int local_N = 0;
    int nnzb_local = 0;

    int *h_rowp = nullptr;
    int *h_cols = nullptr;
    T *h_vals = nullptr;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    T *d_vals = nullptr;

    T *d_x_full = nullptr;
    T *d_y_local = nullptr;

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseMatDescr_t descrA = nullptr;
};

template <typename T>
void extractLocalBSRRows(
    int row_start,
    int row_end,
    const int *rowp,
    const int *cols,
    const T *vals,
    int block_dim,
    int **local_rowp_out,
    int **local_cols_out,
    T **local_vals_out,
    int *local_nnzb_out
) {
    int local_nrows = row_end - row_start;
    int block_dim2 = block_dim * block_dim;

    int start_nnz = rowp[row_start];
    int end_nnz = rowp[row_end];
    int nnzb_local = end_nnz - start_nnz;

    int *local_rowp = (int*)malloc((local_nrows + 1) * sizeof(int));
    int *local_cols = (int*)malloc(nnzb_local * sizeof(int));
    T *local_vals = (T*)malloc(nnzb_local * block_dim2 * sizeof(T));

    local_rowp[0] = 0;
    for (int i = 0; i < local_nrows; i++) {
        local_rowp[i + 1] = rowp[row_start + i + 1] - start_nnz;
    }

    for (int k = 0; k < nnzb_local; k++) {
        local_cols[k] = cols[start_nnz + k]; // keep GLOBAL block columns
    }

    for (int k = 0; k < nnzb_local * block_dim2; k++) {
        local_vals[k] = vals[start_nnz * block_dim2 + k];
    }

    *local_rowp_out = local_rowp;
    *local_cols_out = local_cols;
    *local_vals_out = local_vals;
    *local_nnzb_out = nnzb_local;
}

template <typename T>
void setupMultiGPU(
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

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(g));

        int row_start = (g * nnodes) / ngpu;
        int row_end = ((g + 1) * nnodes) / ngpu;

        gpus[g].dev = g;
        gpus[g].row_start_node = row_start;
        gpus[g].row_end_node = row_end;
        gpus[g].local_nnodes = row_end - row_start;
        gpus[g].local_N = gpus[g].local_nnodes * block_dim;

        extractLocalBSRRows<T>(
            row_start, row_end,
            rowp, cols, vals,
            block_dim,
            &gpus[g].h_rowp,
            &gpus[g].h_cols,
            &gpus[g].h_vals,
            &gpus[g].nnzb_local
        );

        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_rowp,
                              (gpus[g].local_nnodes + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_cols,
                              gpus[g].nnzb_local * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_vals,
                              gpus[g].nnzb_local * block_dim2 * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_x_full,
                              N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&gpus[g].d_y_local,
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

        printf("GPU %d owns block rows [%d, %d), local_nnodes = %d, local nnzb = %d\n",
               g, row_start, row_end, gpus[g].local_nnodes, gpus[g].nnzb_local);
    }
}

template <typename T>
void multiGpuBSRMatVec(
    std::vector<GPUData<T>> &gpus,
    int ngpu,
    int N,
    int nnodes,
    int block_dim,
    const T *h_x,
    T *h_y
) {
    T alpha = 1.0;
    T beta = 0.0;

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        CHECK_CUDA(cudaMemcpy(gpus[g].d_x_full, h_x,
                              N * sizeof(T),
                              cudaMemcpyHostToDevice));

        CHECK_CUSPARSE(cusparseDbsrmv(
            gpus[g].cusparseHandle,
            CUSPARSE_DIRECTION_ROW,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            gpus[g].local_nnodes,  // local block rows
            nnodes,                // global block cols because x is full
            gpus[g].nnzb_local,
            &alpha,
            gpus[g].descrA,
            gpus[g].d_vals,
            gpus[g].d_rowp,
            gpus[g].d_cols,
            block_dim,
            gpus[g].d_x_full,
            &beta,
            gpus[g].d_y_local
        ));
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));

        int scalar_start = gpus[g].row_start_node * block_dim;

        CHECK_CUDA(cudaMemcpy(&h_y[scalar_start],
                              gpus[g].d_y_local,
                              gpus[g].local_N * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void cleanupMultiGPU(std::vector<GPUData<T>> &gpus) {
    for (auto &gd : gpus) {
        CHECK_CUDA(cudaSetDevice(gd.dev));

        if (gd.d_rowp) cudaFree(gd.d_rowp);
        if (gd.d_cols) cudaFree(gd.d_cols);
        if (gd.d_vals) cudaFree(gd.d_vals);
        if (gd.d_x_full) cudaFree(gd.d_x_full);
        if (gd.d_y_local) cudaFree(gd.d_y_local);

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
    // Single-GPU reference y_single = A*x on GPU 0
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
    // Multi-GPU row-split SpMV with replicated x
    // -------------------------------------------------------

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count <= 0) {
        printf("No CUDA GPUs found.\n");
        return 1;
    }

    int requested_gpus = 4;
    int ngpu = std::min(requested_gpus, device_count);

    printf("Using %d GPU(s) out of %d available.\n", ngpu, device_count);

    std::vector<GPUData<T>> gpus;

    setupMultiGPU<T>(
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

    multiGpuBSRMatVec<T>(
        gpus,
        ngpu,
        N,
        nnodes,
        block_dim,
        x,
        y_multi
    );

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

    printf("\nSingle vs multi-GPU BSR SpMV check:\n");
    printf("  rel L2 error = %.15e\n", rel_err);
    printf("  max abs err  = %.15e\n", max_abs);

    if (rel_err < 1e-12) {
        printf("  PASS\n");
    } else {
        printf("  FAIL / investigate BSR row split or block ordering\n");
    }

    // cleanup
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
#pragma once

#include <cstdio>
#include <cstring>

#include "cuda_utils.h"
#include "gpuvec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include "utils.h"

template <typename T>
class GPUdiagmat {
   public:
    GPUdiagmat(cublasHandle_t *cublasHandles_, cusparseHandle_t *cusparseHandles_, int *h_rowp_,
               int *h_cols_, T *h_vals_, int ngpus_, int N_, int block_dim_ = 6,
               bool debug_ = false)
        : cublasHandles(cublasHandles_), cusparseHandles(cusparseHandles_) {
        h_rowp = h_rowp_;
        h_cols = h_cols_;
        h_vals = h_vals_;
        ngpus = ngpus_;
        N = N_;
        block_dim = block_dim_;
        nnodes = N / block_dim;
        debug = debug_;
        block_dim2 = block_dim * block_dim;

        tmp = new GPUvec<T>(cublasHandles, ngpus, N, block_dim, debug);

        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];

        diag_nnzb = new int[ngpus];

        h_diag_rowp = new int *[ngpus];
        h_diag_cols = new int *[ngpus];
        h_diag_vals = new T *[ngpus];

        d_diag_rowp = new int *[ngpus];
        d_diag_cols = new int *[ngpus];
        d_diag_vals = new T *[ngpus];

        pBuffer = new void *[ngpus];

        memset(h_diag_rowp, 0, ngpus * sizeof(int *));
        memset(h_diag_cols, 0, ngpus * sizeof(int *));
        memset(h_diag_vals, 0, ngpus * sizeof(T *));

        memset(d_diag_rowp, 0, ngpus * sizeof(int *));
        memset(d_diag_cols, 0, ngpus * sizeof(int *));
        memset(d_diag_vals, 0, ngpus * sizeof(T *));

        memset(pBuffer, 0, ngpus * sizeof(void *));

        if (debug) printf("GPUdiagmat with nnodes %d, ngpus %d\n", nnodes, ngpus);

        for (int g = 0; g < ngpus; g++) {
            start_node[g] = nnodes * g / ngpus;
            end_node[g] = nnodes * (g + 1) / ngpus;
            local_nnodes[g] = end_node[g] - start_node[g];
            local_N[g] = local_nnodes[g] * block_dim;

            if (debug) printf("\tgpu[%d] nodes [%d,%d)\n", g, start_node[g], end_node[g]);

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            int mb = local_nnodes[g];
            diag_nnzb[g] = mb;

            h_diag_rowp[g] = new int[mb + 1];
            h_diag_cols[g] = new int[mb];

            memset(h_diag_rowp[g], 0, (mb + 1) * sizeof(int));
            memset(h_diag_cols[g], 0, mb * sizeof(int));

            for (int i = 0; i < mb; i++) {
                h_diag_rowp[g][i + 1] = i + 1;
                h_diag_cols[g][i] = i;
            }

            CHECK_CUDA(cudaMalloc((void **)&d_diag_rowp[g], (mb + 1) * sizeof(int)));

            CHECK_CUDA(cudaMalloc((void **)&d_diag_cols[g], mb * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(d_diag_rowp[g], h_diag_rowp[g], (mb + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMemcpy(d_diag_cols[g], h_diag_cols[g], mb * sizeof(int),
                                  cudaMemcpyHostToDevice));

            h_diag_vals[g] = new T[block_dim2 * mb];
            memset(h_diag_vals[g], 0, block_dim2 * mb * sizeof(T));

            for (int i = start_node[g]; i < end_node[g]; i++) {
                int iloc = i - start_node[g];

                for (int jp = h_rowp[i]; jp < h_rowp[i + 1]; jp++) {
                    int j = h_cols[jp];

                    if (i == j) {
                        for (int ii = 0; ii < block_dim2; ii++) {
                            h_diag_vals[g][block_dim2 * iloc + ii] = h_vals[block_dim2 * jp + ii];
                        }
                        break;
                    }
                }
            }

            CHECK_CUDA(cudaMalloc((void **)&d_diag_vals[g], mb * block_dim2 * sizeof(T)));

            CHECK_CUDA(cudaMemcpy(d_diag_vals[g], h_diag_vals[g], mb * block_dim2 * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    ~GPUdiagmat() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            if (d_diag_rowp && d_diag_rowp[g]) cudaFree(d_diag_rowp[g]);
            if (d_diag_cols && d_diag_cols[g]) cudaFree(d_diag_cols[g]);
            if (d_diag_vals && d_diag_vals[g]) cudaFree(d_diag_vals[g]);
            if (pBuffer && pBuffer[g]) cudaFree(pBuffer[g]);

            if (h_diag_rowp && h_diag_rowp[g]) delete[] h_diag_rowp[g];
            if (h_diag_cols && h_diag_cols[g]) delete[] h_diag_cols[g];
            if (h_diag_vals && h_diag_vals[g]) delete[] h_diag_vals[g];
        }

        delete[] start_node;
        delete[] end_node;
        delete[] local_nnodes;
        delete[] local_N;

        delete[] diag_nnzb;

        delete[] h_diag_rowp;
        delete[] h_diag_cols;
        delete[] h_diag_vals;

        delete[] d_diag_rowp;
        delete[] d_diag_cols;
        delete[] d_diag_vals;

        delete[] pBuffer;

        delete tmp;
    }

    void factor() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            int mb = local_nnodes[g];
            int nnzb = diag_nnzb[g];

            CUSPARSE::perform_ilu0_factorization(cusparseHandles[g], descr_L, descr_U, info_L,
                                                 info_U, &pBuffer[g], mb, nnzb, block_dim,
                                                 d_diag_vals[g], d_diag_rowp[g], d_diag_cols[g],
                                                 trans_L, trans_U, policy_L, policy_U, dir);
        }
    }

    void solve(GPUvec<T> *x, GPUvec<T> *y) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            int mb = local_nnodes[g];
            int nnzb = diag_nnzb[g];

            T a = 1.0;

            T *loc_x = x->getPtr(g);
            T *loc_tmp = tmp->getPtr(g);
            T *loc_y = y->getPtr(g);

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandles[g], dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_diag_vals[g], d_diag_rowp[g],
                                                 d_diag_cols[g], block_dim, info_L, loc_x, loc_tmp,
                                                 policy_L, pBuffer[g]));

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandles[g], dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_diag_vals[g], d_diag_rowp[g],
                                                 d_diag_cols[g], block_dim, info_U, loc_tmp, loc_y,
                                                 policy_U, pBuffer[g]));
        }
    }

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;

    int nnodes = 0;
    int block_dim = 0;
    int N = 0;
    int ngpus = 0;
    int block_dim2 = 0;

    int *start_node = nullptr;
    int *end_node = nullptr;
    int *local_nnodes = nullptr;
    int *local_N = nullptr;

    int *h_rowp = nullptr;
    int *h_cols = nullptr;
    T *h_vals = nullptr;

    bool debug = false;

    int *diag_nnzb = nullptr;

    int **h_diag_rowp = nullptr;
    int **h_diag_cols = nullptr;
    T **h_diag_vals = nullptr;

    int **d_diag_rowp = nullptr;
    int **d_diag_cols = nullptr;
    T **d_diag_vals = nullptr;

    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;

    bsrsv2Info_t info_L = 0;
    bsrsv2Info_t info_U = 0;

    void **pBuffer = nullptr;

    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    GPUvec<T> *tmp = nullptr;
};
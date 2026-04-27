#pragma once

#include "cuda_utils.h"
#include "gpuvec.h"
#include "solvers/linear_static/_cusparse_utils.h"
#include "utils.h"

template <typename T>
class GPUdiagmat {
    // a matrix class for multi-GPU parallelism

   public:
    GPUdiagmat(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, int *h_rowp_,
               int *h_cols_, T *h_vals_, int ngpus_, int N_, int block_dim_ = 6,
               bool debug_ = false)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        // a constructor from a host data form of the matrix
        // need alternate form from all on GPU later

        // set the host version of matrix
        h_rowp = h_rowp_, h_cols = h_cols_, h_vals = h_vals_;
        ngpus = ngpus_, N = N_, block_dim = block_dim_;
        nnodes = N / block_dim;
        debug = debug_;
        block_dim2 = block_dim * block_dim;

        tmp = new GPUvec<T>(cublasHandle, ngpus, N);

        // setup standard vector storage (for ghost copies)
        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        h_diag_rowp = new int *[ngpus];
        h_diag_cols = new int *[ngpus];
        d_diag_rowp = new int *[ngpus];
        d_diag_cols = new int *[ngpus];
        h_diag_vals = new T *[ngpus];
        d_diag_vals = new T *[ngpus];
        // d_vals_owned = new T *[ngpus]; // change this to d_vals_ext size only
        // NOTE: ghost nodes handled in mat-vec products not in vectors
        printf("GPUdiagmat with nnodes %d, ngpus %d\n", nnodes, ngpus);
        for (int g = 0; g < ngpus; g++) {
            start_node[g] = nnodes * g / ngpus;
            end_node[g] = nnodes * (g + 1) / ngpus;
            local_nnodes[g] = end_node[g] - start_node[g];
            local_N[g] = local_nnodes[g] * block_dim;
            printf("\tgpu[%d] nodes [%d,%d)\n", g, start_node[g], end_node[g]);

            if (!debug) CHECK_CUDA(cudaSetDevice(g));

            // get diagonal matrix pattern owned by this GPU
            int mb = local_nnodes[g];
            h_diag_rowp[g] = new int[mb + 1];
            h_diag_cols[g] = new int[mb];
            diag_nnzb[g] = nnodes;
            memset(h_diag_rowp[g], 0, (mb + 1) * sizeof(int));
            memset(h_diag_cols[g], 0, mb * sizeof(int));
            for (int i = 0; i < mb; i++) {
                h_diag_rowp[g][i + 1] = i + 1;
                h_diag_cols[g][i] = i;
            }

            CHECK_CUDA(cudaMalloc((void **)&d_diag_rowp[g], (mb + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_diag_cols[g], mb * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_diag_rowp[g], h_diag_rowp, (mb + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(
                cudaMemcpy(d_diag_cols[g], h_diag_cols, mb * sizeof(int), cudaMemcpyHostToDevice));

            // now get the values
            h_diag_vals[g] = new T[block_dim2 * mb];
            for (int i = 0; i < nnodes; i++) {
                int row_on_proc = start_node[g] <= i && i < end_node[g];
                if (!row_on_proc) continue;
                for (int jp = h_rowp[i]; jp < h_rowp[i + 1]; jp++) {
                    int j = h_cols[jp];
                    if (i == j) {
                        for (int ii = 0; ii < block_dim2; ii++) {
                            h_diag_vals[g][block_dim2 * i + ii] = h_vals[block_dim2 * jp + ii];
                        }
                    }
                }
            }

            CHECK_CUDA(cudaMalloc((void **)d_diag_vals[g], mb * block_dim2 * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(d_diag_vals[g], h_diag_vals[g], mb * block_dim2 * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    void factor() {
        // perform the symbolic and numeric factorization of LU on given sparsity pattern
        // on each GPU g
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int mb = local_nnodes[g];
            int nnzb = mb;
            CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, mb, nnzb, block_dim, d_diag_vals[g],
                                                 d_diag_rowp[g], d_diag_cols[g], trans_L, trans_U,
                                                 policy_L, policy_U, dir);
        }
    }

    void solve(GPUvec<T> *x, GPUvec<T> *y) {
        // Dinv * x => y

        for (int g = 0; g < ngpus; g++) {
            int mb = local_nnodes[g];
            int nnzb = mb;
            T a = 1.0;
            T *loc_x = x->getPtr(g);
            T *loc_tmp = tmp->getPtr(g);
            T *loc_y = y->getPtr(g);
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_diag_vals[g], d_diag_rowp[g],
                d_diag_cols[g], block_dim, info_L, loc_x, loc_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_diag_vals[g], d_diag_rowp[g],
                d_diag_cols[g], block_dim, info_U, loc_tmp, loc_y, policy_U, pBuffer))
        }
    }

    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrA = nullptr;
    int nnodes, block_dim, N, ngpus, block_dim2;
    int *start_node = nullptr, *end_node = nullptr;
    int *local_nnodes = nullptr, *local_N = nullptr;

    // host version of matrix
    int *h_rowp = nullptr, *h_cols = nullptr;
    T *h_vals = nullptr;
    bool debug;

    // local diagonal matrices on each GPU
    int *diag_nnzb;
    int **h_diag_rowp, **h_diag_cols;
    T **h_diag_vals;
    int **d_diag_rowp, **d_diag_cols;
    T **d_diag_vals;

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // temp vec
    GPUvec<T> *tmp;
};
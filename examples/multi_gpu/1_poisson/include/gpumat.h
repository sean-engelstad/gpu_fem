#pragma once
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "cuda_utils.h"
#include "gpumat.cuh"
#include "gpuvec.h"
#include "utils.h"

template <typename T>
class GPUbsrmat {
    // Multi-GPU BSR matrix.
    // Rows are owned by dst GPU. Matvec builds x_wghost[dst] = owned x[dst] + ghost x from src
    // GPUs. NOTE: x_wghost is still a GPUvec over ngpus. Pairwise reduced ghost buffers are raw
    // arrays, not GPUvec over ngpus*ngpus.

   public:
    GPUbsrmat(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, int *h_rowp_,
              int *h_cols_, T *h_vals_, int ngpus_, int N_, int block_dim_ = 6, bool debug_ = false)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        h_rowp = h_rowp_;
        h_cols = h_cols_;
        h_vals = h_vals_;
        ngpus = ngpus_;
        N = N_;
        block_dim = block_dim_;
        nnodes = N / block_dim;
        debug = debug_;
        block_dim2 = block_dim * block_dim;

        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];

        if (debug) printf("GPUbsrmat-internal vec with nnodes %d, ngpus %d\n", nnodes, ngpus);
        for (int g = 0; g < ngpus; g++) {
            start_node[g] = nnodes * g / ngpus;
            end_node[g] = nnodes * (g + 1) / ngpus;
            local_nnodes[g] = end_node[g] - start_node[g];
            local_N[g] = local_nnodes[g] * block_dim;
            if (debug) printf("\tgpu[%d] nodes [%d,%d)\n", g, start_node[g], end_node[g]);
        }

        setup_local_gpu_matrices();
        setup_ghost_nodes();
    }

    ~GPUbsrmat() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            if (d_loc_rowp && d_loc_rowp[g]) cudaFree(d_loc_rowp[g]);
            if (d_loc_cols && d_loc_cols[g]) cudaFree(d_loc_cols[g]);
            if (d_loc_vals && d_loc_vals[g]) cudaFree(d_loc_vals[g]);

            if (h_loc_rowp && h_loc_rowp[g]) delete[] h_loc_rowp[g];
            if (h_loc_cols && h_loc_cols[g]) delete[] h_loc_cols[g];
            if (h_loc_vals && h_loc_vals[g]) delete[] h_loc_vals[g];
            if (h_cols_map && h_cols_map[g]) delete[] h_cols_map[g];

            if (ghost_nnodes && ghost_nnodes[g]) delete[] ghost_nnodes[g];
            if (h_ghost_nodes && h_ghost_nodes[g]) {
                for (int src = 0; src < ngpus; src++) {
                    if (h_ghost_nodes[g][src]) delete[] h_ghost_nodes[g][src];
                }
                delete[] h_ghost_nodes[g];
            }
        }

        if (d_xred) {
            for (int dst = 0; dst < ngpus; dst++) {
                for (int src = 0; src < ngpus; src++) {
                    int idx = pair_index(dst, src);
                    if (d_xred[idx]) {
                        CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                        cudaFree(d_xred[idx]);
                    }
                }
            }
        }

        if (d_srcred_map) {
            for (int dst = 0; dst < ngpus; dst++) {
                for (int src = 0; src < ngpus; src++) {
                    int idx = pair_index(dst, src);
                    if (d_srcred_map[idx]) {
                        CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                        cudaFree(d_srcred_map[idx]);
                    }
                }
            }
        }

        if (h_srcred_map) {
            for (int i = 0; i < ngpus * ngpus; i++) {
                if (h_srcred_map[i]) delete[] h_srcred_map[i];
            }
        }

        delete x_wghost;
        if (descrA) cusparseDestroyMatDescr(descrA);

        delete[] start_node;
        delete[] end_node;
        delete[] local_nnodes;
        delete[] local_N;

        delete[] loc_mb;
        delete[] loc_nb;
        delete[] loc_nnzb;
        delete[] loc_nnz;
        delete[] h_loc_rowp;
        delete[] h_loc_cols;
        delete[] h_loc_vals;
        delete[] d_loc_rowp;
        delete[] d_loc_cols;
        delete[] d_loc_vals;
        delete[] h_cols_map;
        delete[] ghost_nnodes;
        delete[] h_ghost_nodes;

        delete[] dst_offset_nnodes;
        delete[] srcdest_nnodes;
        delete[] temp_ct;
        delete[] h_srcred_map;
        delete[] d_srcred_map;
        delete[] d_xred;
        delete[] xred_N;
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    void setup_local_gpu_matrices() {
        printf("create local gpu matrices\n");
        loc_mb = new int[ngpus];
        loc_nb = new int[ngpus];
        loc_nnzb = new int[ngpus];
        loc_nnz = new int[ngpus];
        h_loc_rowp = new int *[ngpus];
        h_loc_cols = new int *[ngpus];
        h_loc_vals = new T *[ngpus];
        d_loc_vals = new T *[ngpus];
        d_loc_rowp = new int *[ngpus];
        d_loc_cols = new int *[ngpus];
        h_cols_map = new int *[ngpus];
        ghost_nnodes = new int *[ngpus];
        h_ghost_nodes = new int **[ngpus];

        memset(h_loc_rowp, 0, ngpus * sizeof(int *));
        memset(h_loc_cols, 0, ngpus * sizeof(int *));
        memset(h_loc_vals, 0, ngpus * sizeof(T *));
        memset(d_loc_vals, 0, ngpus * sizeof(T *));
        memset(d_loc_rowp, 0, ngpus * sizeof(int *));
        memset(d_loc_cols, 0, ngpus * sizeof(int *));
        memset(h_cols_map, 0, ngpus * sizeof(int *));
        memset(ghost_nnodes, 0, ngpus * sizeof(int *));
        memset(h_ghost_nodes, 0, ngpus * sizeof(int **));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
        CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            int _loc_nnodes = local_nnodes[g];
            int _start = start_node[g], _end = end_node[g];

            int *h_loc_row_cts = new int[_loc_nnodes];
            memset(h_loc_row_cts, 0, _loc_nnodes * sizeof(int));

            std::unordered_set<int> unique_cols_set;
            loc_nnzb[g] = 0;
            for (int row = _start; row < _end; row++) {
                int loc_row = row - _start;
                for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                    int j = h_cols[jp];
                    unique_cols_set.insert(j);
                    h_loc_row_cts[loc_row]++;
                    loc_nnzb[g]++;
                }
            }

            std::vector<int> unique_cols(unique_cols_set.begin(), unique_cols_set.end());
            std::sort(unique_cols.begin(), unique_cols.end());
            loc_mb[g] = _loc_nnodes;
            loc_nb[g] = static_cast<int>(unique_cols.size());
            loc_nnz[g] = loc_nnzb[g] * block_dim2;

            h_loc_rowp[g] = new int[_loc_nnodes + 1];
            h_cols_map[g] = new int[loc_nb[g]];
            h_loc_cols[g] = new int[loc_nnzb[g]];
            h_loc_vals[g] = new T[loc_nnz[g]];
            memset(h_loc_rowp[g], 0, (_loc_nnodes + 1) * sizeof(int));
            memset(h_cols_map[g], -1, loc_nb[g] * sizeof(int));
            memset(h_loc_cols[g], 0, loc_nnzb[g] * sizeof(int));
            memset(h_loc_vals[g], 0, loc_nnz[g] * sizeof(T));

            for (int loc_row = 0; loc_row < _loc_nnodes; loc_row++) {
                h_loc_rowp[g][loc_row + 1] = h_loc_rowp[g][loc_row] + h_loc_row_cts[loc_row];
            }

            int *cols_imap = new int[nnodes];
            for (int i = 0; i < nnodes; i++) cols_imap[i] = -1;

            // Reduced local column numbering: owned columns [0, loc_mb), then ghosts [loc_mb,
            // loc_nb).
            int ghost_offset = _loc_nnodes;
            for (int m = 0; m < loc_nb[g]; m++) {
                int col = unique_cols[m];
                if (_start <= col && col < _end) {
                    int colred = col - _start;
                    cols_imap[col] = colred;
                    h_cols_map[g][colred] = col;
                } else {
                    int colred = ghost_offset++;
                    cols_imap[col] = colred;
                    h_cols_map[g][colred] = col;
                }
            }

            memset(h_loc_row_cts, 0, _loc_nnodes * sizeof(int));
            for (int row = _start; row < _end; row++) {
                int loc_row = row - _start;
                for (int pass = 0; pass < 2; pass++) {
                    for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                        int j = h_cols[jp];
                        int jred = cols_imap[j];
                        bool is_owned = jred < _loc_nnodes;
                        if ((pass == 0 && is_owned) || (pass == 1 && !is_owned)) {
                            int offset = h_loc_row_cts[loc_row] + h_loc_rowp[g][loc_row];
                            h_loc_cols[g][offset] = jred;
                            for (int ii = 0; ii < block_dim2; ii++) {
                                h_loc_vals[g][block_dim2 * offset + ii] =
                                    h_vals[block_dim2 * jp + ii];
                            }
                            h_loc_row_cts[loc_row]++;
                        }
                    }
                }
            }

            if (debug) {
                printf("GPU[%d] mat with nrows %d x ncols %d\n", g, loc_mb[g], loc_nb[g]);
                for (int row = 0; row < _loc_nnodes; row++) {
                    printf("row %d: ", row);
                    for (int jp = h_loc_rowp[g][row]; jp < h_loc_rowp[g][row + 1]; jp++) {
                        int col = h_loc_cols[g][jp];
                        printf(col < loc_mb[g] ? "%d," : "%dg,", col);
                    }
                    printf("\n");
                }
            }

            ghost_nnodes[g] = new int[ngpus];
            memset(ghost_nnodes[g], 0, ngpus * sizeof(int));
            h_ghost_nodes[g] = new int *[ngpus];
            memset(h_ghost_nodes[g], 0, ngpus * sizeof(int *));

            for (int col_red = 0; col_red < loc_nb[g]; col_red++) {
                if (col_red < loc_mb[g]) continue;
                int col = h_cols_map[g][col_red];
                int src = find_owned_gpu(col);
                if (src >= 0 && src != g) ghost_nnodes[g][src]++;
            }

            for (int src = 0; src < ngpus; src++) {
                h_ghost_nodes[g][src] = new int[ghost_nnodes[g][src]];
                memset(h_ghost_nodes[g][src], 0, ghost_nnodes[g][src] * sizeof(int));
            }

            int *temp = new int[ngpus];
            memset(temp, 0, ngpus * sizeof(int));
            for (int col_red = 0; col_red < loc_nb[g]; col_red++) {
                if (col_red < loc_mb[g]) continue;
                int col = h_cols_map[g][col_red];
                int src = find_owned_gpu(col);
                if (src >= 0 && src != g) {
                    h_ghost_nodes[g][src][temp[src]++] = col - start_node[src];
                }
            }

            if (debug) {
                for (int src = 0; src < ngpus; src++) {
                    printf("h_ghost_nodes[%d => %d] (%d): ", src, g, ghost_nnodes[g][src]);
                    printVec<int>(ghost_nnodes[g][src], h_ghost_nodes[g][src]);
                }
            }

            CHECK_CUDA(cudaMalloc((void **)&d_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_rowp[g], h_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc((void **)&d_loc_cols[g], loc_nnzb[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_cols[g], h_loc_cols[g], loc_nnzb[g] * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc((void **)&d_loc_vals[g], loc_nnz[g] * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(d_loc_vals[g], h_loc_vals[g], loc_nnz[g] * sizeof(T),
                                  cudaMemcpyHostToDevice));

            delete[] temp;
            delete[] cols_imap;
            delete[] h_loc_row_cts;
        }
    }

    int find_owned_gpu(int node) const {
        for (int g = 0; g < ngpus; g++) {
            if (start_node[g] <= node && node < end_node[g]) return g;
        }
        return -1;
    }

    void setup_ghost_nodes() {
        // This one is legitimately a vector over the physical GPUs, so GPUvec is fine here.
        x_wghost = new GPUvec<T>(cublasHandle, loc_nb, ngpus, N, block_dim, debug);

        int npairs = ngpus * ngpus;
        dst_offset_nnodes = new int[npairs];
        srcdest_nnodes = new int[npairs];
        temp_ct = new int[npairs];
        h_srcred_map = new int *[npairs];
        d_srcred_map = new int *[npairs];
        d_xred = new T *[npairs];
        xred_N = new int[npairs];

        memset(dst_offset_nnodes, 0, npairs * sizeof(int));
        memset(srcdest_nnodes, 0, npairs * sizeof(int));
        memset(temp_ct, 0, npairs * sizeof(int));
        memset(h_srcred_map, 0, npairs * sizeof(int *));
        memset(d_srcred_map, 0, npairs * sizeof(int *));
        memset(d_xred, 0, npairs * sizeof(T *));
        memset(xred_N, 0, npairs * sizeof(int));

        // Count and set offsets. Offsets are reduced-column offsets in x_wghost[dst], not global
        // nodes.
        for (int dst = 0; dst < ngpus; dst++) {
            int mb = loc_mb[dst];
            int nb = loc_nb[dst];
            int *cols_map = h_cols_map[dst];

            for (int col_red = mb; col_red < nb; col_red++) {
                int gcol = cols_map[col_red];
                int src = find_owned_gpu(gcol);
                if (src < 0 || src == dst) continue;
                int idx = pair_index(dst, src);
                if (srcdest_nnodes[idx] == 0) dst_offset_nnodes[idx] = col_red;
                srcdest_nnodes[idx]++;
            }
        }

        for (int idx = 0; idx < npairs; idx++) {
            if (srcdest_nnodes[idx] > 0) {
                h_srcred_map[idx] = new int[srcdest_nnodes[idx]];
                xred_N[idx] = srcdest_nnodes[idx] * block_dim;
            }
        }

        // Fill source-local node maps used by k_copyghostred on the source GPU.
        for (int dst = 0; dst < ngpus; dst++) {
            int mb = loc_mb[dst];
            int nb = loc_nb[dst];
            int *cols_map = h_cols_map[dst];

            for (int col_red = mb; col_red < nb; col_red++) {
                int gcol = cols_map[col_red];
                int src = find_owned_gpu(gcol);
                if (src < 0 || src == dst) continue;
                int idx = pair_index(dst, src);
                int src_loc_node = gcol - start_node[src];
                h_srcred_map[idx][temp_ct[idx]++] = src_loc_node;
            }
        }

        // Device maps and pairwise reduced buffers live on the source GPU.
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;
                int idx = pair_index(dst, src);
                if (srcdest_nnodes[idx] == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(
                    cudaMalloc((void **)&d_srcred_map[idx], srcdest_nnodes[idx] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_srcred_map[idx], h_srcred_map[idx],
                                      srcdest_nnodes[idx] * sizeof(int), cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaMalloc((void **)&d_xred[idx], xred_N[idx] * sizeof(T)));
            }
        }
    }

    void expandVecToGhost(GPUvec<T> *x) {
        for (int dst = 0; dst < ngpus; dst++) {
            // 1) Copy owned part into beginning of x_wghost[dst].
            CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
            int loc_N = x->getLocalSize(dst);
            T *loc_x = x->getPtr(dst);
            T *loc_xwg = x_wghost->getPtr(dst);
            CHECK_CUDA(cudaMemcpy(loc_xwg, loc_x, loc_N * sizeof(T), cudaMemcpyDeviceToDevice));

            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;
                int idx = pair_index(dst, src);
                int N_red = xred_N[idx];
                if (N_red == 0) continue;

                // 2) On src GPU: pack needed source-owned nodes into d_xred[idx].
                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                T *loc_x_src = x->getPtr(src);
                T *loc_x_red = d_xred[idx];
                int nnodes_red = N_red / block_dim;
                int *d_map = d_srcred_map[idx];
                dim3 block(128), grid((N_red + block.x - 1) / block.x);
                k_copyghostred<T>
                    <<<grid, block>>>(nnodes_red, block_dim, d_map, loc_x_src, loc_x_red);
                CHECK_CUDA(cudaGetLastError());

                // 3) Copy packed ghost data into dst's x_wghost at its reduced-column offset.
                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                int dst_offset = dst_offset_nnodes[idx] * block_dim;
                if (debug || src == dst) {
                    CHECK_CUDA(cudaMemcpy(loc_xwg + dst_offset, loc_x_red, N_red * sizeof(T),
                                          cudaMemcpyDeviceToDevice));
                } else {
                    CHECK_CUDA(cudaMemcpyPeer(loc_xwg + dst_offset, dst, loc_x_red, src,
                                              N_red * sizeof(T)));
                }
            }
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void mult(T a, GPUvec<T> *x, T b, GPUvec<T> *y) {
        expandVecToGhost(x);

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, loc_mb[g],
                loc_nb[g], loc_nnzb[g], &a, descrA, d_loc_vals[g], d_loc_rowp[g], d_loc_cols[g],
                block_dim, x_wghost->getPtr(g), &b, y->getPtr(g)));
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void mult(GPUvec<T> *x, GPUvec<T> *y) {
        T a = 1.0, b = 0.0;
        mult(a, x, b, y);
    }

    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrA = nullptr;
    int nnodes = 0, block_dim = 0, N = 0, ngpus = 0, block_dim2 = 0;
    int *start_node = nullptr, *end_node = nullptr;
    int *local_nnodes = nullptr, *local_N = nullptr;

    int *h_rowp = nullptr, *h_cols = nullptr;
    T *h_vals = nullptr;
    bool debug = false;

    int *loc_mb = nullptr, *loc_nb = nullptr, *loc_nnzb = nullptr, *loc_nnz = nullptr;
    int **h_loc_rowp = nullptr, **h_loc_cols = nullptr;
    int **d_loc_rowp = nullptr, **d_loc_cols = nullptr;
    T **h_loc_vals = nullptr, **d_loc_vals = nullptr;
    int **ghost_nnodes = nullptr, ***h_ghost_nodes = nullptr;

    // Physical per-GPU ghost-expanded matvec vector.
    GPUvec<T> *x_wghost = nullptr;

    int **h_cols_map = nullptr;

    // Pairwise ghost transfer data, indexed by idx = ngpus * dst + src.
    int *dst_offset_nnodes = nullptr;
    int *srcdest_nnodes = nullptr;
    int *temp_ct = nullptr;
    int **h_srcred_map = nullptr;
    int **d_srcred_map = nullptr;
    T **d_xred = nullptr;
    int *xred_N = nullptr;
};
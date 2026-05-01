#pragma once
#include <unordered_set>

#include "cuda_utils.h"
#include "gpumat.cuh"
#include "gpuvec.h"
#include "utils.h"

template <typename T>
class GPUbsrmat {
    // a matrix class for multi-GPU parallelism

   public:
    GPUbsrmat(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, int *h_rowp_,
              int *h_cols_, T *h_vals_, int ngpus_, int N_, int block_dim_ = 6, bool debug_ = false)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        // set the host version of matrix
        h_rowp = h_rowp_, h_cols = h_cols_, h_vals = h_vals_;
        ngpus = ngpus_, N = N_, block_dim = block_dim_;
        nnodes = N / block_dim;
        debug = debug_;
        block_dim2 = block_dim * block_dim;

        // setup standard vector storage (for ghost copies)
        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        // d_vals_owned = new T *[ngpus]; // change this to d_vals_ext size only
        // NOTE: ghost nodes handled in mat-vec products not in vectors
        printf("GPUbsrmat-internal vec with nnodes %d, ngpus %d\n", nnodes, ngpus);
        for (int g = 0; g < ngpus; g++) {
            start_node[g] = nnodes * g / ngpus;
            end_node[g] = nnodes * (g + 1) / ngpus;
            local_nnodes[g] = end_node[g] - start_node[g];
            local_N[g] = local_nnodes[g] * block_dim;
            printf("\tgpu[%d] nodes [%d,%d)\n", g, start_node[g], end_node[g]);
        }

        setup_local_gpu_matrices();
        setup_ghost_nodes();
    }

    void setup_local_gpu_matrices() {
        // create local non-square matrices for the multi-GPU mat-vec product on host
        // each GPU owns rows of the output vec
        // then columns will belong some to internal data and some to ghost nodes (with multi-GPU
        // transfer)
        printf("create local gpu matrices\n");
        loc_mb = new int[ngpus];
        loc_nb = new int[ngpus];
        loc_nnzb = new int[ngpus];
        loc_nnz = new int[ngpus];
        h_loc_rowp = new int *[ngpus];
        h_loc_cols = new int *[ngpus];
        h_loc_vals = new T *[ngpus];
        ghost_nnodes = new int *[ngpus];
        h_ghost_nodes = new int **[ngpus];
        d_loc_vals = new T *[ngpus];
        d_loc_rowp = new int *[ngpus];
        d_loc_cols = new int *[ngpus];
        h_cols_map = new int *[ngpus];

        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
        CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

        // maybe use multiple CPUs to do this part? TBD
        for (int g = 0; g < ngpus; g++) {
            // build a local non-square rowp+cols for each gpu (on host)
            int _loc_nnodes = local_nnodes[g];
            int _start = start_node[g], _end = end_node[g];
            int *h_loc_row_cts = new int[_loc_nnodes];
            std::unordered_set<int> unique_cols_set;
            memset(h_loc_row_cts, 0, _loc_nnodes * sizeof(int));
            loc_nnzb[g] = 0;
            int loc_row = 0;
            for (int row = _start; row < _end; row++) {
                loc_row = row - _start;
                for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                    int j = h_cols[jp];
                    unique_cols_set.insert(j);
                    h_loc_row_cts[loc_row]++;
                    loc_nnzb[g]++;
                }
            }
            std::vector<int> unique_cols(unique_cols_set.begin(), unique_cols_set.end());
            int *_h_loc_rowp = h_loc_rowp[g];
            _h_loc_rowp = new int[_loc_nnodes + 1];
            memset(_h_loc_rowp, 0, (_loc_nnodes + 1) * sizeof(int));
            for (int loc_row = 0; loc_row < _loc_nnodes; loc_row++) {
                _h_loc_rowp[loc_row + 1] = _h_loc_rowp[loc_row] + h_loc_row_cts[loc_row];
            }
            int *cols_imap = new int[nnodes];
            int *_cols_map = h_cols_map[g];
            _cols_map = new int[unique_cols.size()];
            memset(cols_imap, -1, nnodes * sizeof(int));
            int ghost_offset = _loc_nnodes;
            std::sort(unique_cols.begin(), unique_cols.end());
            printf("unique cols: ");
            for (auto col : unique_cols) {
                printf("%d,", col);
            }
            printf("\n");

            loc_mb[g] = _loc_nnodes;
            loc_nb[g] = unique_cols.size();

            // ghost nodes should have reduced cols after the owned cols(rows)
            for (int m = 0; m < loc_nb[g]; m++) {
                int col = unique_cols[m];
                if (_start <= col && col < _end) {
                    int colred = col - _start;
                    cols_imap[col] = colred;
                    _cols_map[colred] = col;
                } else {
                    int colred = ghost_offset;
                    cols_imap[col] = colred;
                    _cols_map[colred] = col;
                    ghost_offset++;
                }
            }
            int *_h_loc_cols = h_loc_cols[g];
            _h_loc_cols = new int[loc_nnzb[g]];
            memset(h_loc_row_cts, 0, _loc_nnodes * sizeof(int));
            for (int row = _start; row < _end; row++) {
                loc_row = row - _start;
                for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                    int j = h_cols[jp];
                    int jred = cols_imap[j];

                    if (jred < _loc_nnodes) {
                        // fills non ghost nodes or owned cols first
                        int offset = h_loc_row_cts[loc_row] + _h_loc_rowp[loc_row];
                        _h_loc_cols[offset] = jred;
                        h_loc_row_cts[loc_row]++;
                    }
                }

                // then fill ghost node cols
                for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                    int j = h_cols[jp];
                    int jred = cols_imap[j];

                    if (jred >= _loc_nnodes) {
                        // fills non ghost nodes or owned cols first
                        int offset = h_loc_row_cts[loc_row] + _h_loc_rowp[loc_row];
                        _h_loc_cols[offset] = jred;
                        h_loc_row_cts[loc_row]++;
                    }
                }
            }

            if (debug) {
                printf("GPU[%d] mat with nrows %d x ncols %d\n", g, loc_mb[g], loc_nb[g]);
                for (int row = 0; row < _loc_nnodes; row++) {
                    printf("row %d: ", row);
                    for (int jp = _h_loc_rowp[row]; jp < _h_loc_rowp[row + 1]; jp++) {
                        int col = _h_loc_cols[jp];
                        if (col < loc_mb[g]) {
                            printf("%d,", col);
                        } else {
                            printf("%dg,", col);
                        }
                    }
                    printf("\n");
                }
            }

            // now compute ghost copy maps and full d_x_full vecs (with owned + ghost nodes)
            int *_ghost_nnodes = ghost_nnodes[g];
            _ghost_nnodes = new int[ngpus];
            memset(_ghost_nnodes, 0, ngpus * sizeof(int));
            for (int ii = 0; ii < loc_nb[g]; ii++) {
                int col = _cols_map[ii];
                // check which ghost set (or gpu) it's in..
                for (int j = 0; j < ngpus; j++) {
                    if (g == j) continue;  // skip owned nodes (only ghost)
                    int _start2 = start_node[j], _end2 = end_node[j];
                    if (_start2 <= col && col < _end2) {
                        _ghost_nnodes[j]++;
                    }
                }
            }
            int **_h_ghost_nodes = h_ghost_nodes[g];
            _h_ghost_nodes = new int *[ngpus];
            for (int j = 0; j < ngpus; j++) {
                _h_ghost_nodes[j] = new int[_ghost_nnodes[j]];
                memset(_h_ghost_nodes[j], 0, _ghost_nnodes[j] * sizeof(int));
            }
            memset(_ghost_nnodes, 0, ngpus * sizeof(int));
            for (int ii = 0; ii < unique_cols.size(); ii++) {
                int col = _cols_map[ii];
                // check which ghost set (or gpu) it's in..
                for (int j = 0; j < ngpus; j++) {
                    if (g == j) continue;  // skip owned nodes (only ghost)
                    int _start2 = start_node[j], _end2 = end_node[j];
                    if (_start2 <= col && col < _end2) {
                        // map from ghost nodes in dx_full to owned node in other GPU
                        _h_ghost_nodes[j][_ghost_nnodes[j]] = col - _start2;
                        _ghost_nnodes[j]++;
                    }
                }
            }

            for (int j = 0; j < ngpus; j++) {
                printf("h_ghost_nodes[%d => %d] (%d): ", j, g, _ghost_nnodes[j]);
                printVec<int>(_ghost_nnodes[j], _h_ghost_nodes[j]);
            }

            // move host rowp, cols to device
            CHECK_CUDA(cudaMalloc((void **)&d_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_rowp[g], h_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc((void **)&d_loc_cols[g], loc_nnzb[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_cols[g], h_loc_cols[g], loc_nnzb[g] * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // make the vals for the local matrix
            if (h_vals != nullptr) {
                // not scalable for large DOF problems, but doable for small test problems
                T *_h_loc_vals = h_loc_vals[g];
                loc_nnz[g] = loc_nnzb[g] * block_dim2;
                _h_loc_vals = new T[loc_nnz[g]];
                for (int row = _start; row < _end; row++) {
                    loc_row = row - _start;
                    for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                        int j = h_cols[jp];
                        for (int kp = _h_loc_rowp[loc_row]; kp < _h_loc_rowp[loc_row + 1]; kp++) {
                            int kred = _h_loc_cols[kp];
                            int k = _cols_map[kred];
                            if (j == k) {
                                for (int ii = 0; ii < block_dim2; ii++) {
                                    _h_loc_vals[block_dim2 * kp + ii] =
                                        h_vals[block_dim2 * jp + ii];
                                }
                            }
                        }
                    }
                }

                // move host matrix values onto device
                CHECK_CUDA(cudaMalloc((void **)&d_loc_vals[g], loc_nnz[g] * sizeof(T)));
                CHECK_CUDA(cudaMemcpy(d_loc_vals[g], h_loc_vals[g], loc_nnz[g] * sizeof(T),
                                      cudaMemcpyHostToDevice));
            }

        }  // end of GPU loop
    }

    int find_owned_gpu(int node) {
        // find the GPU which owns this node (for standard vectors without ghosts)
        int owned_gpu = -1;
        for (int g = 0; g < ngpus; g++) {
            int start = start_node[g], end = end_node[g];
            if (start <= node && node < end) {
                owned_gpu = g;
            }
        }
        return owned_gpu;
    }

    void setup_ghost_nodes() {
        // setup ghost copy vectors and all things for ghost nodes

        // this x_wghost is the full vector used in mat-vec products on each local GPU
        // need to copy owned parts of vec + ghost parts of vec from other gpus
        x_wghost = new GPUvec<T>(cublasHandle, loc_nb, ngpus, N);

        // TODO : fill out the following
        dst_offset_nnodes = new int[ngpus * ngpus];
        memset(dst_offset_nnodes, 0, ngpus * ngpus);
        srcdest_nnodes = new int[ngpus * ngpus];
        memset(srcdest_nnodes, 0, ngpus * ngpus);
        temp_ct = new int[ngpus * ngpus];
        memset(temp_ct, 0, ngpus * ngpus);
        h_srcred_map = new int *[ngpus * ngpus];
        d_srcred_map = new int *[ngpus * ngpus];

        // dst gpu (destination)
        for (int dst = 0; dst < ngpus; dst++) {
            int *cols_map = h_cols_map[dst];  // colred to full cols
            // now build ghost node copy maps needed for kernels
            int mb = loc_mb[dst];
            int nb = loc_nb[dst];

            for (int col_red = 0; col_red < nb; col_red++) {
                int gcol = cols_map[col_red];
                if (col_red < mb) {
                    // then it's owned by this node (not a ghost node)
                    continue;
                } else {
                    // then it is a ghost node (find which ghost node it belongs to)
                    int src = find_owned_gpu(gcol);  // src GPU
                    if (srcdest_nnodes[ngpus * dst + src] == 0) {
                        dst_offset_nnodes[ngpus * dst + src] = gcol;
                    }
                    srcdest_nnodes[ngpus * dst + src]++;
                }
            }

            int red_node = 0;
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;
                h_srcred_map[ngpus * dst + src] = new int[srcdest_nnodes[ngpus * dst + src]];
            }
            // now fill out srcred map
            for (int col_red = 0; col_red < nb; col_red++) {
                int gcol = cols_map[col_red];
                if (col_red < mb) {
                    // then it's owned by this node (not a ghost node)
                    continue;
                } else {
                    // then it is a ghost node (find which ghost node it belongs to)
                    int src = find_owned_gpu(gcol);  // src GPU
                    int src_loc_node = gcol - start_node[src];
                    int offset = temp_ct[ngpus * dst + src];
                    h_srcred_map[ngpus * dst + src][offset] = src_loc_node;
                    temp_ct[ngpus * dst + src]++;
                }
            }
            // move h_srcredmap - host to device
            for (int src = 0; src < ngpus; src++) {
                CHECK_CUDA(cudaMalloc((void **)d_srcred_map[ngpus * dst + src],
                                      srcdest_nnodes[ngpus * dst + src] * block_dim * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_srcred_map[ngpus * dst + src],
                                      h_srcred_map[ngpus * dst + src],
                                      srcdest_nnodes[ngpus * dst + src] * block_dim * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
        }

        xred = new GPUvec<T>(cublasHandle, srcdest_nnodes, ngpus * ngpus, N);
    }

    void expandVecToGhost(GPUvec<T> *x) {
        // expand GPUvec x to include ghost nodes (for multi-GPU matvec product)

        // destination GPU (dst)
        for (int dst = 0; dst < ngpus; dst++) {
            // 1) copy owned nodes into x_wghost from same GPU
            CHECK_CUDA(cudaSetDevice(dst));
            int loc_N = x->getLocalSize(dst);
            T *loc_x = x->getPtr(dst);
            T *loc_xwg = x_wghost->getPtr(dst);
            CHECK_CUDA(cudaMemcpy(loc_xwg, loc_x, loc_N * sizeof(T), cudaMemcpyDeviceToDevice));

            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                // 2) copy ghost nodes (needed in dst vec) from gpu src to red-vec on src gpu
                CHECK_CUDA(cudaSetDevice(dst));
                T *loc_x_src = x->getPtr(src);
                T *loc_x_red = xred->getPtr(ngpus * dst + src);
                int N_red = xred->getLocalSize(ngpus * dst + src);
                int nnodes_red = N_red / block_dim;
                int *d_map = d_srcred_map[ngpus * dst + src];
                dim3 block(32), grid((nnodes_red + 31) / 32);
                k_copyghostred<T>
                    <<<grid, block>>>(nnodes_red, block_dim, d_map, loc_x_src, loc_x_red);

                // 3) copy from red vec on src gpu to ghost nodes on dst gpu
                int dst_offset = dst_offset_nnodes[ngpus * dst + src] * block_dim;
                CHECK_CUDA(
                    cudaMemcpyPeer(loc_xwg + dst_offset, dst, loc_x_red, src, N_red * sizeof(T)));
            }
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void mult(T &a, GPUvec<T> *x, T &b, GPUvec<T> *y) {
        // a * A * x + b * y => y on multi-GPU vectors

        // copy x into x_wghost with ghost nodes
        expandVecToGhost(x);

        // do local GPU matvec product
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int *loc_rowp = d_loc_rowp[g];
            int *loc_cols = d_loc_cols[g];
            T *loc_vals = d_loc_vals[g];
            T *x_loc_wghost = x_wghost->getPtr(g);
            T *y_loc = y->getPtr(g);
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, loc_mb[g], loc_nb[g],
                                          loc_nnzb[g], &a, descrA, loc_vals, loc_rowp, loc_cols,
                                          block_dim, x_loc_wghost, &b, y_loc));
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void mult(GPUvec<T> &x, GPUvec<T> &y) {
        // A * x => y
        T a = 1.0, b = 0.0;
        mult(a, x, b, y);
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

    // local device matrices (non-square)
    int *loc_mb, *loc_nb, *loc_nnzb, *loc_nnz;
    int **h_loc_rowp, **h_loc_cols;
    int **d_loc_rowp, **d_loc_cols;
    T **h_loc_vals, **d_loc_vals;
    int **ghost_nnodes, ***h_ghost_nodes;
    GPUvec<T> *xred, *x_wghost;  // includes ghost nodes

    // ghost node utils
    int **h_cols_map;
    int *dst_offset_nnodes, *srcdest_nnodes, *temp_ct;
    int **h_srcred_map, **d_srcred_map;
};
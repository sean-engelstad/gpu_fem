#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>

#include "cuda_utils.h"

template <typename T>
class GPUvec {
   public:
    GPUvec(cublasHandle_t *cublasHandles_, cudaStream_t *streams_, int ngpus_, int N_,
           int block_dim_ = 6, bool debug_ = false)
        : cublasHandles(cublasHandles_),
          streams(streams_),
          ngpus(ngpus_),
          block_dim(block_dim_),
          N(N_),
          debug(debug_),
          owns_layout(true) {
        nnodes = N / block_dim;

        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        d_vals_owned = new T *[ngpus];

        memset(d_vals_owned, 0, ngpus * sizeof(T *));

        if (debug) printf("GPUvec with nnodes %d, ngpus %d\n", nnodes, ngpus);

        for (int g = 0; g < ngpus; g++) {
            start_node[g] = nnodes * g / ngpus;
            end_node[g] = nnodes * (g + 1) / ngpus;
            local_nnodes[g] = end_node[g] - start_node[g];
            local_N[g] = local_nnodes[g] * block_dim;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[g], local_N[g] * sizeof(T)));

            CHECK_CUDA(cudaMemsetAsync(d_vals_owned[g], 0, local_N[g] * sizeof(T), streams[g]));
        }

        sync_all_streams();
    }

    GPUvec(cublasHandle_t *cublasHandles_, cudaStream_t *streams_, const int *local_nnodes_,
           int ngpus_, int N_, int block_dim_ = 6, bool debug_ = false)
        : cublasHandles(cublasHandles_),
          streams(streams_),
          ngpus(ngpus_),
          block_dim(block_dim_),
          N(N_),
          debug(debug_),
          owns_layout(true) {
        nnodes = N / block_dim;

        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        d_vals_owned = new T *[ngpus];

        memset(d_vals_owned, 0, ngpus * sizeof(T *));

        if (debug) printf("GPUvec with custom local sizes, ngpus %d\n", ngpus);

        int node_offset = 0;

        for (int g = 0; g < ngpus; g++) {
            start_node[g] = node_offset;
            local_nnodes[g] = local_nnodes_[g];
            end_node[g] = start_node[g] + local_nnodes[g];
            node_offset = end_node[g];
            local_N[g] = local_nnodes[g] * block_dim;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[g], local_N[g] * sizeof(T)));

            CHECK_CUDA(cudaMemsetAsync(d_vals_owned[g], 0, local_N[g] * sizeof(T), streams[g]));
        }

        sync_all_streams();
    }

    ~GPUvec() {
        sync_all_streams();

        if (d_vals_owned) {
            for (int g = 0; g < ngpus; g++) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                if (d_vals_owned[g]) cudaFree(d_vals_owned[g]);
            }
            delete[] d_vals_owned;
        }

        delete[] start_node;
        delete[] end_node;
        delete[] local_nnodes;
        delete[] local_N;
    }

    void sync_all_streams() const {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

    void setFromHost(const T *h_vals) {
        for (int g = 0; g < ngpus; g++) {
            int start = start_node[g];

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            CHECK_CUDA(cudaMemcpyAsync(d_vals_owned[g], &h_vals[block_dim * start],
                                       local_N[g] * sizeof(T), cudaMemcpyHostToDevice, streams[g]));
        }

        sync_all_streams();
    }

    void copyTo(GPUvec<T> *y) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            T *loc_x = getPtr(g);
            T *loc_y = y->getPtr(g);

            CHECK_CUDA(cudaMemcpyAsync(loc_y, loc_x, local_N[g] * sizeof(T),
                                       cudaMemcpyDeviceToDevice, streams[g]));
        }

        sync_all_streams();
    }

    T getResidual() {
        T total_dot2 = 0.0;
        T *h_dot = new T[ngpus];
        memset(h_dot, 0, ngpus * sizeof(T));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            T *loc_x = getPtr(g);

            CHECK_CUBLAS(cublasDdot(cublasHandles[g], local_N[g], loc_x, 1, loc_x, 1, &h_dot[g]));
        }

        sync_all_streams();

        for (int g = 0; g < ngpus; g++) {
            total_dot2 += h_dot[g];
        }

        delete[] h_dot;

        return sqrt(total_dot2);
    }

    void scale(T alpha) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            T *loc_x = getPtr(g);

            CHECK_CUBLAS(cublasDscal(cublasHandles[g], local_N[g], &alpha, loc_x, 1));
        }

        sync_all_streams();
    }

    T dotProd(GPUvec<T> *y) {
        T total_dot = 0.0;
        T *h_dot = new T[ngpus];
        memset(h_dot, 0, ngpus * sizeof(T));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            T *loc_x = getPtr(g);
            T *loc_y = y->getPtr(g);

            CHECK_CUBLAS(cublasDdot(cublasHandles[g], local_N[g], loc_x, 1, loc_y, 1, &h_dot[g]));
        }

        sync_all_streams();

        for (int g = 0; g < ngpus; g++) {
            total_dot += h_dot[g];
        }

        delete[] h_dot;

        return total_dot;
    }

    void axpy(T alpha, GPUvec<T> *x) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);

            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }

        sync_all_streams();
    }

    void axpby(T alpha, GPUvec<T> *x, T beta) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);

            CHECK_CUBLAS(cublasDscal(cublasHandles[g], local_N[g], &beta, loc_y, 1));

            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }

        sync_all_streams();
    }

    T *getPtr(int g) { return d_vals_owned[g]; }

    int getLocalSize(int g) const { return local_N[g]; }
    int getStartNode(int g) const { return start_node[g]; }
    int getEndNode(int g) const { return end_node[g]; }

    int ngpus = 0;
    int block_dim = 0;
    int N = 0;
    int nnodes = 0;

    bool debug = false;
    bool owns_layout = true;

    int *start_node = nullptr;
    int *end_node = nullptr;
    int *local_nnodes = nullptr;
    int *local_N = nullptr;

    T **d_vals_owned = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;
};
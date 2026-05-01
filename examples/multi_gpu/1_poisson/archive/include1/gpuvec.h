#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>

#include "cuda_utils.h"

template <typename T>
class GPUvec {
    // a vector class for multi-GPU parallelism

   public:
    GPUvec(cublasHandle_t &cublasHandle_, int ngpus_, int N_, int block_dim_ = 6,
           bool debug_ = false)
        : cublasHandle(cublasHandle_),
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
        for (int i = 0; i < ngpus; i++) {
            start_node[i] = nnodes * i / ngpus;
            end_node[i] = nnodes * (i + 1) / ngpus;
            local_nnodes[i] = end_node[i] - start_node[i];
            local_N[i] = local_nnodes[i] * block_dim;
            if (debug) printf("\tgpu[%d] nodes [%d,%d)\n", i, start_node[i], end_node[i]);

            CHECK_CUDA(cudaSetDevice(debug ? 0 : i));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[i], local_N[i] * sizeof(T)));
            CHECK_CUDA(cudaMemset(d_vals_owned[i], 0, local_N[i] * sizeof(T)));
        }
    }

    GPUvec(cublasHandle_t &cublasHandle_, const int *local_nnodes_, int ngpus_, int N_,
           int block_dim_ = 6, bool debug_ = false)
        : cublasHandle(cublasHandle_),
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
        for (int i = 0; i < ngpus; i++) {
            start_node[i] = node_offset;
            local_nnodes[i] = local_nnodes_[i];
            end_node[i] = start_node[i] + local_nnodes[i];
            node_offset = end_node[i];
            local_N[i] = local_nnodes[i] * block_dim;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : i));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[i], local_N[i] * sizeof(T)));
            CHECK_CUDA(cudaMemset(d_vals_owned[i], 0, local_N[i] * sizeof(T)));
        }
    }

    ~GPUvec() {
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

    void setFromHost(const T *h_vals) {
        for (int i = 0; i < ngpus; i++) {
            int start = start_node[i];
            CHECK_CUDA(cudaSetDevice(debug ? 0 : i));
            CHECK_CUDA(cudaMemcpy(d_vals_owned[i], &h_vals[block_dim * start],
                                  local_N[i] * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copyTo(GPUvec<T> *y) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = getPtr(g);
            T *loc_y = y->getPtr(g);
            CHECK_CUDA(cudaMemcpy(loc_y, loc_x, local_N[g] * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    T getResidual() {
        T total_dot2 = 0.0;
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = getPtr(g);
            T loc_dot2 = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, local_N[g], loc_x, 1, loc_x, 1, &loc_dot2));
            total_dot2 += loc_dot2;
        }
        return sqrt(total_dot2);
    }

    void scale(T alpha) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = getPtr(g);
            CHECK_CUBLAS(cublasDscal(cublasHandle, local_N[g], &alpha, loc_x, 1));
        }
    }

    T dotProd(GPUvec<T> *y) {
        T total_dot = 0.0;
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = getPtr(g);
            T *loc_y = y->getPtr(g);
            T loc_dot = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, local_N[g], loc_x, 1, loc_y, 1, &loc_dot));
            total_dot += loc_dot;
        }
        return total_dot;
    }

    void axpy(T alpha, GPUvec<T> *x) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }
    }

    void axpby(T alpha, GPUvec<T> *x, T beta) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);
            CHECK_CUBLAS(cublasDscal(cublasHandle, local_N[g], &beta, loc_y, 1));
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }
    }

    T *getPtr(int g) { return d_vals_owned[g]; }
    int getLocalSize(int g) const { return local_N[g]; }
    int getStartNode(int g) const { return start_node[g]; }
    int getEndNode(int g) const { return end_node[g]; }

    int ngpus, block_dim, N, nnodes;
    bool debug = false;
    bool owns_layout = true;
    int *start_node = nullptr;
    int *end_node = nullptr;
    int *local_nnodes = nullptr;
    int *local_N = nullptr;

    T **d_vals_owned = nullptr;
    cublasHandle_t &cublasHandle;
};
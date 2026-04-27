#pragma once
#include "cuda_utils.h"

template <typename T>
class GPUvec {
    // a vector class for multi-GPU parallelism

   public:
    GPUvec(cublasHandle_t &cublasHandle_, int ngpus_, int N_, int block_dim_ = 6,
           bool debug = false)
        : cublasHandle(cublasHandle_), ngpus(ngpus_), block_dim(block_dim_), N(N_) {
        nnodes = N / block_dim;

        start_node = new int[ngpus];
        end_node = new int[ngpus];
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        d_vals_owned = new T *[ngpus];
        // NOTE: ghost nodes handled in mat-vec products not in vectors
        printf("GPUvec with nnodes %d, ngpus %d\n", nnodes, ngpus);
        for (int i = 0; i < ngpus; i++) {
            start_node[i] = nnodes * i / ngpus;
            end_node[i] = nnodes * (i + 1) / ngpus;
            local_nnodes[i] = end_node[i] - start_node[i];
            local_N[i] = local_nnodes[i] * block_dim;
            printf("\tgpu[%d] nodes [%d,%d)\n", i, start_node[i], end_node[i]);

            if (!debug) CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[i], local_N[i] * sizeof(T)));
        }
    }

    GPUvec(cublasHandle_t &cublasHandle_, int *local_nnodes_, int ngpus_, int N_,
           int block_dim_ = 6, bool debug = false)
        : cublasHandle(cublasHandle_), ngpus(ngpus_), block_dim(block_dim_), N(N_) {
        nnodes = N / block_dim;

        local_nnodes = local_nnodes_;  // local_N sizes provided from external
        local_N = new int[ngpus];
        d_vals_owned = new T *[ngpus];
        // NOTE: ghost nodes handled in mat-vec products not in vectors
        printf("GPUvec with nnodes %d, ngpus %d\n", nnodes, ngpus);
        for (int i = 0; i < ngpus; i++) {
            local_N[i] = local_nnodes[i] * block_dim;
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[i], local_N[i] * sizeof(T)));
        }
    }

    ~GPUvec() {
        if (d_vals_owned) {
            for (int g = 0; g < ngpus; g++) {
                CHECK_CUDA(cudaSetDevice(g));
                if (d_vals_owned[g]) cudaFree(d_vals_owned[g]);
            }
            delete[] d_vals_owned;
        }

        delete[] start_node;
        delete[] end_node;
        delete[] local_nnodes;
        delete[] local_N;
    }

    void setFromHost(T *h_vals) {
        for (int i = 0; i < ngpus; i++) {
            local_N[i] = local_nnodes[i] * block_dim;
            int start = start_node[i];
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaMemcpy(d_vals_owned[i], &h_vals[block_dim * start],
                                  local_N[i] * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copyTo(GPUvec<T> *y) {
        // copy this vec (x) to destination vec y
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int loc_N = local_N[g];
            T *loc_x = getPtr(g);
            T *loc_y = y->getPtr(g);
            CHECK_CUDA(cudaMemcpy(loc_y, loc_x, loc_N * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    T getResidual() {
        T total_dot2 = 0.0;
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = getPtr(g);
            T loc_dot2;
            CHECK_CUBLAS(cublasDdot(cublasHandle, local_N[g], loc_x, 1, loc_x, 1, &loc_dot2));
            total_dot2 += loc_dot2;
        }
        return sqrt(total_dot2);
    }

    void scale(T &alpha) {
        // alpha * x => x (this vec is x)
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = getPtr(g);
            CHECK_CUBLAS(cublasDscal(cublasHandle, local_N[g], &alpha, loc_x, 1));
        }
    }

    T dotProd(GPUvec<T> *y) {
        T total_dot2 = 0.0;
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = d_vals_owned[g];
            T *loc_y = y->getPtr(g);
            T loc_dot2;
            CHECK_CUBLAS(cublasDdot(cublasHandle, local_N[g], loc_x, 1, loc_y, 1, &loc_dot2));
            total_dot2 += loc_dot2;
        }
        return total_dot2;
    }

    void axpy(T &alpha, GPUvec<T> *x) {
        // this vector is y, then it computes alpha * x + y => y
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }
    }

    void axpby(T &alpha, GPUvec<T> *x, T &beta) {
        // this vector is y, then it computes alpha * x + beta * y => y
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getPtr(g);
            T *loc_y = getPtr(g);
            // beta * y into y
            CHECK_CUBLAS(cublasDscal(cublasHandle, local_N[g], &beta, loc_y, 1));

            // alpha * x + y2 => y (where y2 = beta * y)
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, local_N[g], &alpha, loc_x, 1, loc_y, 1));
        }
    }

    T *getPtr(int g) { return d_vals_owned[g]; }

    int getLocalSize(int g) const { return local_N[g]; }

    int getStartNode(int g) const { return start_node[g]; }

    int getEndNode(int g) const { return end_node[g]; }

    int ngpus, block_dim, N, nnodes;
    int *start_node = nullptr;
    int *end_node = nullptr;
    int *local_nnodes = nullptr;
    int *local_N = nullptr;

    T **d_vals_owned = nullptr;
    cublasHandle_t &cublasHandle;
};

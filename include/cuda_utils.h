#pragma once

#include <iostream>

#ifdef USE_GPU
#define __SHARED__ __shared__
#define __HOST_DEVICE__ __host__ __device__
#define __HOST__ __host__
#define __DEVICE__ __device__
#define __GLOBAL__ __global__
#else
#define __SHARED__
#define __HOST_DEVICE__
#define __HOST__
#define __DEVICE__
#define __GLOBAL__
#endif

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

template <uint32_t xdim = 1, uint32_t ydim = 1, uint32_t zdim = 1,
          uint32_t max_registers_per_thread = 255,
          uint32_t elements_per_block = 1>
class ExecParameters {
  public:
};

#ifdef USE_GPU
// Usage: put gpuErrchk(...) around cuda function calls
#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_CUSPARSE(call)                                                   \
    {                                                                          \
        cusparseStatus_t err = call;                                           \
        if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
            std::cerr << "CUSPARSE error: " << err << std::endl;               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_CUBLAS(func)                                                     \
    {                                                                          \
        cublasStatus_t status = (func);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            printf("CUBLAS API failed at line %d with error: %d\n", __LINE__,  \
                   status);                                                    \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

#define CHECK_CUSOLVER(func)                                                   \
    {                                                                          \
        cusolverStatus_t status = (func);                                      \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            printf("CUSOLVER API failed at line %d with error: %d\n",          \
                   __LINE__, status);                                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

void cuda_show_kernel_error() {
    auto err = cudaGetLastError();
    std::cout << "error code: " << err << "\n";
    std::cout << "error string: " << cudaGetErrorString(err) << "\n";
}
#endif
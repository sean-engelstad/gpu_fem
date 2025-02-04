#pragma once

#include <iostream>

#ifdef USE_GPU
#define __SHARED__ __shared__
#define __HOST_DEVICE__ __host__ __device__
#define __DEVICE__ __device__
#define __GLOBAL__ __global__
#else
#define __SHARED__
#define __HOST_DEVICE__
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
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

void cuda_show_kernel_error() {
  auto err = cudaGetLastError();
  std::cout << "error code: " << err << "\n";
  std::cout << "error string: " << cudaGetErrorString(err) << "\n";
}
#endif
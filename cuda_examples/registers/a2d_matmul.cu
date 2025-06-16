#include <cstdio>
#include "a2dcore.h"
#include <chrono>

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }


template <typename T>
__global__ void drill_strain_kernel_v1(T *global_out) {
    // regular matmul with drill strains
    T vars[3];
    vars[0] = 1.515;
    vars[1] = -1.414;
    vars[2] = 2.367;

    T Tmat[9];
    #pragma unroll
    for (int i = 0 ; i < 9; i++) {
        Tmat[i] = 1.0443 + 2.35435 * i - 0.0213 * i * i + 1e-4 * (blockIdx.x + threadIdx.x * 2.13);
    }

    T C[9];
    C[0] = 1.0;
    C[4] = 1.0;
    C[8] = 1.0;
    C[1] = -vars[2];
    C[2] = vars[1];
    C[5] = -vars[0];
    C[3] = vars[2];
    C[6] = -vars[1];
    C[7] = vars[0];

    T tmp[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, C, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

    // does it compile away unused instructions here?
    global_out[0] += 0.5 * (C[3] - C[1]);
}

template <typename T>
__global__ void drill_strain_kernel_v2(T *global_out) {
    // regular matmul with drill strains
    T vars[3];
    vars[0] = 1.515;
    vars[1] = -1.414;
    vars[2] = 2.367;

    T Tmat[9];
    #pragma unroll
    for (int i = 0 ; i < 9; i++) {
        Tmat[i] = 1.0443 + 2.35435 * i - 0.0213 * i * i + 1e-4 * (blockIdx.x + threadIdx.x * 2.13);
    }

    T C[9];
    C[0] = 1.0;
    C[4] = 1.0;
    C[8] = 1.0;
    C[1] = -vars[2];
    C[2] = vars[1];
    C[5] = -vars[0];
    C[3] = vars[2];
    C[6] = -vars[1];
    C[7] = vars[0];

    // split up into exact computations (if T = [t1,t2,n])
    // C3 = t2^T * C * t1 (are these the same calculation? no bc C is antisym, but they are negative of each other right)
    // T C3 = Tmat[0] * (C[0] * Tmat[1] + C[1] * Tmat[4] + C[2] * Tmat[7]) + \
    //        Tmat[3] * (C[3] * Tmat[1] + C[4] * Tmat[4] + C[5] * Tmat[7]) + \
    //        Tmat[6] * (C[6] * Tmat[1] + C[7] * Tmat[4] + C[8] * Tmat[7]);

    // // C1 = t1^T * C * t2 = -C3 (antisym actually)
    // // T C1 = -C3;
    // T C1 = -Tmat[0] * (C[0] * Tmat[1] + C[1] * Tmat[4] + C[2] * Tmat[7]) - \
    //        Tmat[3] * (C[3] * Tmat[1] + C[4] * Tmat[4] + C[5] * Tmat[7]) - \
    //        Tmat[6] * (C[6] * Tmat[1] + C[7] * Tmat[4] + C[8] * Tmat[7]);

    // code it up the same but now with the new a2d core function for scalar triple product
    // pretend Tmat is transposed now
    T C3 = A2D::Mat3x3VecTripleProduct<T>(&Tmat[0], C, &Tmat[3]);
    T C1 = -C3;

    // does it compile away unused instructions here?
    global_out[0] += 0.5 * (C3 - C1);
}

int main() {
    dim3 block(32);
    dim3 grid(10000);

    double *out;
    cudaMalloc((void **)&out, sizeof(double));
    cudaMemset(out, 0.0, sizeof(double));

    // run both of them twice to prime the GPU
    drill_strain_kernel_v1<<<grid, block>>>(out);
    drill_strain_kernel_v2<<<grid, block>>>(out);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start1 = std::chrono::high_resolution_clock::now();
    drill_strain_kernel_v1<<<grid, block>>>(out);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto stop1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt1 = stop1 - start1;
    printf("finished v1 kernel (A2D::MatMul) in %.4e\n", dt1.count());

    auto start2 = std::chrono::high_resolution_clock::now();
    drill_strain_kernel_v2<<<grid, block>>>(out);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto stop2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt2 = stop2 - start2;
    printf("finished v2 kernel with one scalar triple prod (hand-coded) in %.4e\n", dt2.count());
}

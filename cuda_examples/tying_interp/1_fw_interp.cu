#include "include/basis.h"
#include "include/director.h"
#include "include/isotropic.h"
#include "include/kernels.cuh"
#include "linalg/_linalg.h"
#include <chrono>

template <typename T, class Basis, int elems_per_block = 32>
__global__ void reg_interp_kernel(int num_elements, T *xpts, T *vars, T *out) {

    // int iquad = threadIdx.x;
    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    // bool active_thread = global_elem < num_elements;
    // int local_thread =
    //     (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    __SHARED__ T block_xpts[elems_per_block][12];
    __SHARED__ T block_vars[elems_per_block][24];
    // __SHARED__ T block_normals[elems_per_block][nxpts_per_elem];

    // xpts, vars stored in elem format here..
    for (int i = threadIdx.x; i < 12; i += 4) {
        block_xpts[local_elem][i] = xpts[global_elem * 12 + i];
    }

    for (int i = threadIdx.x; i < 24; i += 4) {
        block_vars[local_elem][i] = vars[global_elem * 24 + i];
    }
    __syncthreads();

    // now compute some tying strains dot products and then add to final out (the regular more register way prob)
    /* regular interp method */
    int itying = 0;
    T pt[2];
    Basis::template getTyingPoint<0>(itying, pt);

    // Interpolate the field value
    T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, block_xpts[local_elem], Xxi, Xeta);
    Basis::template interpFieldsGrad<6, 3>(pt, block_vars[local_elem], Uxi, Ueta);

    // store g11 strain
    T local_out = A2D::VecDotCore<T, 3>(Uxi, Xxi);

    // just atomic add (we could warp reduce here), but not necessary to compare registers
    atomicAdd(out, local_out);
}

template <typename T, class Basis, int elems_per_block = 32>
__global__ void light_interp_kernel(int num_elements, T *xpts, T *vars, T *out) {

    // int iquad = threadIdx.x;
    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    // bool active_thread = global_elem < num_elements;
    // int local_thread =
    //     (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    __SHARED__ T block_xpts[elems_per_block][12];
    __SHARED__ T block_vars[elems_per_block][24];
    // __SHARED__ T block_normals[elems_per_block][nxpts_per_elem];

    // xpts, vars stored in elem format here..
    for (int i = threadIdx.x; i < 12; i += 4) {
        block_xpts[local_elem][i] = xpts[global_elem * 12 + i];
    }

    for (int i = threadIdx.x; i < 24; i += 4) {
        block_vars[local_elem][i] = vars[global_elem * 24 + i];
    }
    __syncthreads();

    // now compute some tying strains dot products and then add to final out (the regular more register way prob)
    /* regular interp method */
    int itying = 0;
    T pt[2];
    Basis::template getTyingPoint<0>(itying, pt);

    // Interpolate the field value
    T local_out = Basis::template interpFieldsGradDotProduct<3, 6, 3, 0, 0>(
                                  pt, xpts, vars);

    // just atomic add (we could warp reduce here), but not necessary to compare registers
    atomicAdd(out, local_out);
}


int main() {
    using T = double;
    using Quad = QuadLinearQuadrature<T>;
    using Basis = ShellQuadBasis<T, Quad>;

    const int nelems = 32000;
    dim3 block(4, 32);
    int nblocks = nelems/32;
    dim3 grid(nblocks);

    // compute random xpts and vars
    T *h_xpts = new T[nelems * 12];
    T *h_vars = new T[nelems * 24];
    // random values on host
    for (int i = 0; i < nelems * 12; i++) {
        h_xpts[i] = ((T)rand()) / RAND_MAX;
    }
    for (int i = 0; i < nelems * 24; i++) {
        h_vars[i] = ((T)rand()) / RAND_MAX;
    }

    T *xpts, *vars;
    cudaMalloc((void **)&xpts, 12 * nelems * sizeof(double));
    cudaMalloc((void **)&vars, 24 * nelems * sizeof(double));
    cudaMemcpy(xpts, h_xpts, 12 * nelems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vars, h_vars, 24 * nelems * sizeof(double), cudaMemcpyHostToDevice);

    // scalar output
    double *out;
    cudaMalloc((void **)&out, sizeof(double));
    cudaMemset(out, 0.0, sizeof(double));

    // warmup kernel
    reg_interp_kernel<T, Basis, 32><<<grid,block>>>(nelems, xpts, vars, out);
    cudaDeviceSynchronize();

    // call the kernel (regular one)
    auto start1 = std::chrono::high_resolution_clock::now();
    reg_interp_kernel<T, Basis, 32><<<grid,block>>>(nelems, xpts, vars, out);
    cudaDeviceSynchronize();
    auto stop1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt1 = stop1 - start1;
    printf("finished reg interp kernel v1 in %.4e\n", dt1.count());

    // call the lighter kernel
    auto start2 = std::chrono::high_resolution_clock::now();
    light_interp_kernel<T, Basis, 32><<<grid,block>>>(nelems, xpts, vars, out);
    cudaDeviceSynchronize();
    auto stop2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt2 = stop2 - start2;
    printf("finished light interp kernel v2 in %.4e\n", dt2.count());

    // // 
    // cudaDeviceSynchronize();
    return 0;
};
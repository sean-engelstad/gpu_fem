#pragma once
#include "cuda_utils.h"

template <typename T>
__global__ void k_copyghostred(const int nnodes_red, const int block_dim, const int *d_redmap,
                               const T *loc_x_src, T *loc_x_red) {
    int N_red = nnodes_red * block_dim;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_red) return;

    int red_node = tid / block_dim;
    int idim = tid % block_dim;
    int src_node = d_redmap[red_node];

    loc_x_red[tid] = loc_x_src[block_dim * src_node + idim];
}
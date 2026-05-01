#pragma once
#include "cuda_utils.h"

template <typename T>
__global__ void k_copyghostred(const int nnodes_red, const int block_dim, const int *d_redmap, 
    const T *loc_x_src, T *loc_x_red) {
    // copy ghost nodes from source vector on src GPU to reduced vec for (src,dst) pair )but stored on src GPU)
    // which will following this get copied from src GPU to dst GPU with cudaMemcpyPeer
    // this is because only subset of ghost nodes needed from src vector in the x_wghost dst vector (so need reduced step)

    int N_red = nnodes_red * block_dim;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_red) return;

    int red_node = tid / block_dim;
    int idim = tid % block_dim;
    int src_node = d_redmap[red_node];
    int red_dof = tid;
    int src_dof = block_dim * src_node + idim;

    loc_x_red[red_dof] = loc_x_src[src_dof];
}
#pragma once
#include "cuda_utils.h"


template <typename T>
__global__ void k_pack_ghost_red(int nnodes, int block_dim, const int *map, const T *x, T *xred) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nnodes * block_dim;
    if (tid < size) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;
        int src_node = map[inode];
        xred[tid] = x[src_node * block_dim + idof];
    }
}

template <typename T>
__global__ void k_place_ghost_red(int nnodes, int block_dim, const int *map, const T *xred,
                                  T *xloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nnodes * block_dim;
    if (tid < size) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;
        int dst_node = map[inode];
        xloc[dst_node * block_dim + idof] = xred[tid];
    }
}

template <typename T>
__global__ void k_scatter_owned_to_local(int nnodes, int block_dim, const int *map, const T *xowned,
                                         T *xloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nnodes * block_dim;
    if (tid < size) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;
        int loc_node = map[inode];
        xloc[loc_node * block_dim + idof] = xowned[tid];
    }
}

template <typename T>
__GLOBAL__ void k_vec_apply_bcs(int nbcs, int *bcs, T *data) {
    // no need for perm here since this is called on unreordered state of vectors like res, loads, etc.
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x) {
        int idof = bcs[ibc];
        data[idof] = 0.0;
    }
}

template <typename T, bool ones_on_diag = true>
__GLOBAL__ void k_mat_apply_row_bcs(const int block_dim, int n_owned_bcs, int *d_owned_bcs, 
    const int *rowp, const int *cols, T *values) {
    int this_thread_bc = blockIdx.x * blockDim.x + threadIdx.x;
    int block_dim2 = block_dim * block_dim;

    if (this_thread_bc < n_owned_bcs) {
        // get data associated with the row of this BC in the BSR matrix
        int bc_dof = d_owned_bcs[this_thread_bc];
        int block_row = bc_dof / block_dim;
        // int _block_row = bc_dof / block_dim;
        // int block_row = iperm[_block_row]; // old to new brow
        int inner_row = bc_dof % block_dim;
        int global_row = block_dim * block_row + inner_row;
        for (int jp = rowp[block_row]; jp < rowp[block_row+1]; jp++) {
            int block_col = cols[jp];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int inner_block = block_dim * inner_row + inner_col;
                int global_col = block_dim * block_col + inner_col;
                T diag_val = ones_on_diag ? 1.0 : 0.0;
                values[block_dim2 * jp + inner_block] = (global_col == global_row) ? diag_val : 0.0;
            }
        }
    }
}


template <typename T, bool ones_on_diag = true>
__GLOBAL__ void k_mat_apply_col_bcs(const int block_dim, int n_local_bcs, int *d_local_bcs, 
    const int *tr_rowp, const int *tr_cols, const int *tr_map, T *values) {
    int this_thread_bc = blockIdx.x * blockDim.x + threadIdx.x;
    int block_dim2 = block_dim * block_dim;

    if (this_thread_bc < n_local_bcs) {
        int bc_dof = d_local_bcs[this_thread_bc];
        // get data associated with the row of this BC in the BSR matrix
        // int _block_col = bc_dof / block_dim;
        // int block_col = iperm[_block_col]; // old to new bcol
        int block_col = bc_dof / block_dim;
        int inner_col = bc_dof % block_dim;
        int global_col = block_dim * block_col + inner_col;
        for (int jp_tr = tr_rowp[block_col]; jp_tr < tr_rowp[block_col+1]; jp_tr++) {
            int block_row = tr_cols[jp_tr];
            
            for (int inner_row = 0; inner_row < block_dim; inner_row++) {
                int inner_block = block_dim * inner_row + inner_col;
                int global_row = block_dim * block_row + inner_row;
                int jp = tr_block_map[jp_tr];
                T diag_val = ones_on_diag ? 1.0 : 0.0;
                values[block_dim2 * jp + inner_block] = (global_col == global_row) ? diag_val : 0.0;
            }
        }
    }
}
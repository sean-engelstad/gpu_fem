// kernel functions and device helper functions for shell multigrid
#pragma once

template <typename T>
__global__ static void k_copyBlockDiagFromBsrMat(int nnodes, int block_dim, int *kmat_diagp,
                                                       T *kmat_vals, T *diag_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int diag_nnz = nnodes * block_dim2;
    if (tid < diag_nnz) {
        int node_ind = tid / block_dim2, inner_block_ind = tid % block_dim2;
        int kmat_nzb_ind = kmat_diagp[node_ind];
        int kmat_nz_ind = block_dim2 * kmat_nzb_ind + inner_block_ind;

        T val = kmat_vals[kmat_nz_ind];
        diag_vals[tid] = val;
    }
}

template <typename T>
__global__ static void k_computeDiagRowScales(int nnodes, int block_dim, T *d_diag_vals, T *d_diag_scales) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nrows = nnodes * block_dim;
    if (tid < nrows) {
        int row = tid, node_ind = tid / block_dim;
        int inner_row = row % block_dim;
        int inner_col = inner_row;
        int nz_diag_ind = block_dim2 * node_ind + block_dim * inner_col + inner_row;
        T row_scale = d_diag_vals[nz_diag_ind];
        d_diag_scales[row] = row_scale;

        // scale that row of the block diag matrix
        for (int inner_col2 = 0; inner_col2 < block_dim; inner_col2++) {
            int nz_diag_ind2 = block_dim2 * node_ind + block_dim * inner_col2 + inner_row;
            d_diag_vals[nz_diag_ind2] /= row_scale;
        }
    }
}

template <typename T>
__global__ static void k_reapplyDiagRowScales(int nnodes, int block_dim, const T *d_diag_scales, T *d_diag_vals, T *d_diag_inv_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nrows = nnodes * block_dim;
    if (tid < nrows) {
        int row = tid, node_ind = tid / block_dim;
        int inner_row = row % block_dim;
        T row_scale = d_diag_scales[row];

        // scale that row of the block diag matrix
        for (int inner_col = 0; inner_col < block_dim; inner_col++) {
            int nz_diag_ind2 = block_dim2 * node_ind + block_dim * inner_col + inner_row;
            d_diag_vals[nz_diag_ind2] *= row_scale;
            d_diag_inv_vals[nz_diag_ind2] /= row_scale;
        }
    }
}

template <typename T>
__global__ static void k_singleToDoublePointer(int nnodes, int block_dim, T *singlePtr, T **doublePtr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    if (tid < nnodes) {
        int inode = tid;
        // int iloc = tid % block_dim2;
        // no need for new data => just point the inner 6x6 pointers to the existing parts of singlePtr object
        doublePtr[inode] = singlePtr + block_dim2 * tid;
    }
}

template <typename T>
__global__ static void k_setSingleVal(int N, int ind, T val, T *vec) {
    // single thread only.. (or just repeats which is fine)
    vec[ind] = val;
}

template <typename T>
__global__ static void k_singleToDoublePointerVec(int nnodes, int block_dim, T *singlePtr, T **doublePtr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nnodes) {
        int inode = tid;
        doublePtr[inode] = singlePtr + block_dim * tid;
    }
}

template <typename T>
__global__ static void k_setBlockUnitVec(int nnodes, int block_dim, int ii, T *vec) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nnodes) {
        int inode = tid;
        vec[block_dim * inode + ii] = 1.0;
    }
}

template <typename T>
__global__ static void k_setLUinv_operator(int nnodes, int block_dim, int ii, const T *rhs, T *Dinv_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvec = nnodes * block_dim;
    if (tid < nvec) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        // copy values of this rhs result from (LU)^-1 triang solve linear operator into Dinv
        Dinv_vals[block_dim2 * inode + block_dim * idof + ii] = rhs[tid];
    }
}

// template <typename T>
// __global__ static void k_Dinv_LU_triang_solves()
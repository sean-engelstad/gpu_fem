#pragma once
#include "bsr_utils.h"
#include "vec.h"
#ifdef USE_GPU
#include "bsr_mat.cuh"
#endif // USE_GPU

template <class Vec> class BsrMat {
  public:
    using T = typename Vec::type;
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
#endif // USE_GPU

    __HOST_DEVICE__ BsrMat(const BsrData &bsr_data, Vec &values)
        : bsr_data(bsr_data), values(values) {}
    __HOST__ BsrMat(const BsrData &bsr_data) : bsr_data(bsr_data) {
        int nvalues = bsr_data.getNumValues();
        values = Vec(nvalues);
    }

    __HOST_DEVICE__ void zeroValues() { values.zeroValues(); }
    __HOST_DEVICE__ BsrData getBsrData() const { return bsr_data; }
    __HOST_DEVICE__ int get_nnz() { return bsr_data.getNumValues(); }
    __HOST__ Vec getVec() { return values; }
    __HOST_DEVICE__ T *getPtr() { return values.getPtr(); }
    __HOST_DEVICE__ const T *getPtr() const { return values.getPtr(); }

    __HOST__ void apply_bcs(DeviceVec<int> bcs) {

        // some prelim values needed for both cases
        int nbcs = bcs.getSize();
        const int *rowPtr = bsr_data.rowPtr;
        const int *colPtr = bsr_data.colPtr;
        int nnodes = bsr_data.nnodes;
        T *valPtr = values.getPtr();

        int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        int block_dim = bsr_data.block_dim;

#ifdef USE_GPU
        dim3 block = bcs_block;
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // launch kernel to apply BCs to the full matrix
        apply_mat_bcs_kernel<T, DeviceVec>
            <<<grid, block>>>(bcs, rowPtr, colPtr, nnodes, valPtr,
                              blocks_per_elem, nnz_per_block, block_dim);

#else  // not USE_GPU

        // loop over each bc
        for (int ibc = 0; ibc < nbcs; ibc++) {
            int node = bcs[ibc];
            int inner_row = node % block_dim;
            int block_row = node / block_dim;
            int istart = rowPtr[block_row];
            int iend = rowPtr[block_row + 1];
            T *val = &valPtr[nnz_per_block * istart];

            // set bc row to zero
            for (int col_ptr_ind = istart; col_ptr_ind < iend; col_ptr_ind++) {
                int block_col = colPtr[col_ptr_ind];
                for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                    int inz = block_dim * inner_row + inner_col;
                    int glob_col = block_col * block_dim + inner_col;
                    // ternary operation will be more friendly on the GPU (even
                    // though this is CPU here)
                    val[inz] = (glob_col == node) ? 1.0 : 0.0;
                }
                val += nnz_per_block;
            }
            // TODO : for bc column want K^T rowPtr, colPtr map probably
            // to do it without if statements (compute map on CPU assembler
            // init) NOTE : zero out columns only needed for nonzero BCs
        }
#endif // USE_GPU
    }

    __HOST_DEVICE__
    void addElementMatrixValues(const T scale, const int ielem,
                                const int dof_per_node,
                                const int nodes_per_elem,
                                const int32_t *elem_conn, const T *elem_mat) {
        // similar method to vec.h or Vec.addElementValues but here for matrix
        // and here for Bsr format
        int dof_per_elem = dof_per_node * nodes_per_elem;
        int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        int block_dim = bsr_data.block_dim;
        const int32_t *elem_ind_map = bsr_data.elemIndMap;
        T *valPtr = values.getPtr();

        // loop over each of the blocks in the kelem
        for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++) {
            int istart = nnz_per_block *
                         elem_ind_map[blocks_per_elem * ielem + elem_block];
            T *val = &valPtr[istart];
            int block_row = elem_block / nodes_per_elem;
            int block_col = elem_block % nodes_per_elem;

            // loop over each nz in each block of kelem
            for (int inz = 0; inz < nnz_per_block; inz++) {
                int inner_row = inz / block_dim;
                int inner_col = inz % block_dim;
                int row = block_dim * block_row + inner_row;
                int col = block_dim * block_col + inner_col;

                val[inz] += scale * elem_mat[dof_per_elem * row + col];
            }
        }
    }

#ifdef USE_GPU
    __DEVICE__
    void addElementMatrixValuesFromShared(const bool active_thread,
                                          const int start, const int stride,
                                          const T scale, const int ielem,
                                          const int dof_per_node,
                                          const int nodes_per_elem,
                                          const int32_t *elem_conn,
                                          const T *shared_elem_mat) {
        // similar method to vec.h or Vec.addElementValues but here for matrix
        // and here for Bsr format
        int dof_per_elem = dof_per_node * nodes_per_elem;
        int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        int block_dim = bsr_data.block_dim;
        const int32_t *elem_ind_map = bsr_data.elemIndMap;
        T *valPtr = values.getPtr();

        // loop over each of the blocks in the kelem
        for (int elem_block = start; elem_block < blocks_per_elem;
             elem_block += stride) {
            int istart = nnz_per_block *
                         elem_ind_map[blocks_per_elem * ielem + elem_block];
            T *val = &valPtr[istart];
            int block_row = elem_block / nodes_per_elem;
            int block_col = elem_block % nodes_per_elem;

            // loop over each nz in each block of kelem
            for (int inz = 0; inz < nnz_per_block; inz++) {
                int inner_row = inz / block_dim;
                int inner_col = inz % block_dim;
                int row = block_dim * block_row + inner_row;
                int col = block_dim * block_col + inner_col;

                atomicAdd(&val[inz],
                          scale * shared_elem_mat[dof_per_elem * row + col]);
            }
        }
    }

#endif // USE_GPU

    template <typename I> __HOST_DEVICE__ T &operator[](const I i) {
        return values[i];
    }
    template <typename I> __HOST_DEVICE__ const T &operator[](const I i) const {
        return values[i];
    }

  private:
    const BsrData bsr_data;
    Vec values;
};

template <class Assembler, class Vec>
__HOST__ BsrMat<Vec> createBsrMat(Assembler &assembler) {
    return BsrMat<Vec>(assembler.getBsrData());
}
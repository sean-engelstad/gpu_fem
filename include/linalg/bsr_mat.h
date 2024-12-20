#pragma once
#include "bsr_utils.h"

template <class Vec> class BsrMat {
  public:
    using T = typename Vec::type;
    __HOST_DEVICE__ BsrMat(const BsrData &bsr_data, Vec &values)
        : bsr_data(bsr_data), values(values) {}
    __HOST_DEVICE__ BsrMat(const BsrData &bsr_data) : bsr_data(bsr_data) {
        int nvalues = bsr_data.getNumValues();
        values = Vec(nvalues);
    }

    __HOST_DEVICE__ int get_nnz() { return bsr_data.getNumValues(); }
    __HOST_DEVICE__ Vec getVec() { return values; }
    __HOST_DEVICE__ T *getPtr() { return values.getPtr(); }

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
__HOST_DEVICE__ BsrMat<Vec> createBsrMat(Assembler &assembler) {
    return BsrMat<Vec>(assembler.getBsrData());
}
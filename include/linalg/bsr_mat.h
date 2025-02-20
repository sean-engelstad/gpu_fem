#pragma once
#include "bsr_utils.h"
#include "vec.h"
#ifdef USE_GPU
#include "../cuda_utils.h"
#include "bsr_mat.cuh"
#endif // USE_GPU

template <class Vec_> class BsrMat {
  public:
    using T = typename Vec_::type;
    using Vec = Vec_;
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
    static constexpr dim3 nodes_block = dim3(32);
#endif // USE_GPU

    __HOST_DEVICE__ BsrMat(const BsrData &bsr_data, Vec &values)
        : bsr_data(bsr_data), values(values) {}
    __HOST__ BsrMat(const BsrData &bsr_data) : bsr_data(bsr_data) {
        int nvalues = bsr_data.getNumValues();
        values = Vec(nvalues);
    }

    __HOST__ void zeroValues() { values.zeroValues(); }
    __HOST_DEVICE__ BsrData getBsrData() const { return bsr_data; }
    __HOST_DEVICE__ int get_nnz() { return bsr_data.getNumValues(); }
    __HOST__ Vec getVec() { return values; }
    __HOST__ HostVec<T> createHostVec() { return values.createHostVec(); }
    __HOST_DEVICE__ T *getPtr() { return values.getPtr(); }
    __HOST_DEVICE__ const T *getPtr() const { return values.getPtr(); }
    __HOST_DEVICE__ int *getPerm() { return bsr_data.perm; }
    __HOST_DEVICE__ int *getIPerm() { return bsr_data.iperm; }
    __HOST_DEVICE__ int getBlockDim() { return bsr_data.block_dim; }
    __HOST_DEVICE__ int *getRowPtr() { return bsr_data.rowPtr; }
    __HOST_DEVICE__ int *getColPtr() { return bsr_data.colPtr; }

    __HOST__ void apply_bcs(HostVec<int> bcs) {

        // some prelim values needed for both cases
        int nbcs = bcs.getSize();
        const index_t *rowPtr = bsr_data.rowPtr;
        const index_t *colPtr = bsr_data.colPtr;
        const index_t *perm = bsr_data.perm;
        int nnodes = bsr_data.nnodes;
        T *valPtr = values.getPtr();

        int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        int block_dim = bsr_data.block_dim;

        // loop over each bc
        for (int ibc = 0; ibc < nbcs; ibc++) {
            // zero out the bc rows
            int _glob_row = bcs[ibc];
            int glob_row = perm[_glob_row]; // the bc dof
            int bc_temp = glob_row;
            int inner_row =
                glob_row % block_dim; // the local dof constrained in this node
            int block_row = glob_row / block_dim; // equiv to bc node

            // set bc row to zero
            for (int col_ptr_ind = rowPtr[block_row];
                 col_ptr_ind < rowPtr[block_row + 1]; col_ptr_ind++) {

                T *val = &valPtr[nnz_per_block * col_ptr_ind];

                int block_col = colPtr[col_ptr_ind];
                for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                    int inz =
                        block_dim * inner_row + inner_col; // nz entry in block
                    int glob_col = block_col * block_dim + inner_col;
                    // ternary operation will be more friendly on the GPU (even
                    // though this is CPU here)
                    val[inz] = (glob_row == glob_col) ? 1.0 : 0.0;
                }
            }

            // TODO : try adding bc to column how too, but will want to speed
            // this up eventually or just use CPU for debug for now set bc row
            // to zero
            for (int block_row2 = 0; block_row2 < nnodes; block_row2++) {
                for (int col_ptr_ind = rowPtr[block_row2];
                     col_ptr_ind < rowPtr[block_row2 + 1]; col_ptr_ind++) {
                    T *val = &valPtr[nnz_per_block * col_ptr_ind];

                    int block_col = colPtr[col_ptr_ind];
                    for (int inz = 0; inz < block_dim * block_dim; inz++) {
                        int inner_row2 = inz / block_dim;
                        int inner_col = inz % block_dim;
                        int glob_row2 = block_row2 * block_dim + inner_row2;

                        int glob_col = block_col * block_dim + inner_col;

                        if (glob_col != bc_temp)
                            continue; // only apply bcs to column here so need
                                      // matching column

                        // ternary operation will be more friendly on the
                        // GPU (even though this is CPU here)
                        val[inz] = (glob_row2 == glob_col) ? 1.0 : 0.0;
                    }
                    // val += nnz_per_block;
                }
            }

            // TODO : for bc column want K^T rowPtr, colPtr map probably
            // to do it without if statements (compute map on CPU assembler
            // init) NOTE : zero out columns only needed for nonzero BCs
        }
    }

    __HOST__ void apply_bcs(DeviceVec<int> bcs) {

        // some prelim values needed for both cases
        int nbcs = bcs.getSize();
        const index_t *rowPtr = bsr_data.rowPtr;
        const index_t *colPtr = bsr_data.colPtr;
        const index_t *perm = bsr_data.perm;
        const index_t *transpose_rowPtr = bsr_data.transpose_rowPtr;
        const index_t *transpose_colPtr = bsr_data.transpose_colPtr;
        const index_t *transpose_block_map = bsr_data.transpose_block_map;
        int nnodes = bsr_data.nnodes;
        T *valPtr = values.getPtr();

        int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        int block_dim = bsr_data.block_dim;
        // int num_global_rows = block_dim * nnodes;

#ifdef USE_GPU
        dim3 block = bcs_block;
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // launch kernel to apply BCs to the full matrix
        apply_mat_bcs_rows_kernel<T, DeviceVec>
            <<<grid, block>>>(bcs, rowPtr, colPtr, perm, nnodes, valPtr,
                              blocks_per_elem, nnz_per_block, block_dim);
        CHECK_CUDA(cudaDeviceSynchronize());

        apply_mat_bcs_cols_kernel<T, DeviceVec><<<grid, block>>>(
            bcs, transpose_rowPtr, transpose_colPtr, transpose_block_map, perm,
            nnodes, valPtr, blocks_per_elem, nnz_per_block, block_dim);

        CHECK_CUDA(cudaDeviceSynchronize());
#endif // USE_GPU
    }

    void copyValuesTo(BsrMat<Vec> mat) {
        // copy values from this matrix to another matrix 'mat'
        // assume the other matrix has same nz locations as this matrix but
        // possibly more for preconditioner

        const index_t *rowp = bsr_data.rowPtr;
        const index_t *cols = bsr_data.colPtr;
        T *vals = values.getPtr();

        int *t_rowp = mat.getRowPtr();
        int *t_cols = mat.getColPtr();
        T *t_vals = mat.getPtr();
        int block_dim = bsr_data.block_dim;
        // int block_dim2 = block_dim * block_dim;
        int nnodes = bsr_data.nnodes;

#ifndef USE_GPU
        // CPU version
        for (int irow = 0; irow < nnodes; irow++) {
            int t_jp = t_rowp[irow];
            int t_jp_max = t_rowp[irow + 1];
            for (int jp = rowp[irow]; jp < rowp[irow + 1]; jp++) {
                int bcol = cols[jp];
                // have other mat catch up to the same column
                for (; t_cols[t_jp] < bcol; t_jp++) {
                }
                // want to put a debug check that same column # here?
                // copy entire block into here
                for (int inz = 0; inz < block_dim2; inz++) {
                    t_vals[block_dim2 * t_jp + inz] =
                        vals[block_dim2 * jp + inz];
                }
            }
        }
#endif

#ifdef USE_GPU
        // GPU code

        // need to write a GPU version of the copyValues above..
        dim3 block = nodes_block;
        int nblocks = (nnodes + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // launch kernel to apply BCs to the full matrix
        copy_mat_values_kernel<T><<<grid, block>>>(
            nnodes, block_dim, rowp, cols, vals, t_rowp, t_cols, t_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
#endif
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
        const index_t *elem_ind_map = bsr_data.elemIndMap;
        T *valPtr = values.getPtr();

        // loop over each of the blocks in the kelem
        for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++) {
            // perm already applied to elem_ind_map
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
        T *valPtr = values.getPtr();

        // V1 - use elem_ind_map to do assembly
        // perm already applied to elem_ind_map (so no need for it here)
        const index_t *elem_ind_map = bsr_data.elemIndMap;

        const index_t *loc_elem_ind_map =
            &elem_ind_map[blocks_per_elem * ielem];
        // if (start == 0 && ielem == 1) {
        //     printf("loc_elem_ind_map: ");
        //     printVec<int32_t>(16, loc_elem_ind_map);
        // }

        for (int elem_block = start; elem_block < blocks_per_elem;
             elem_block += stride) {
            int glob_block_ind = loc_elem_ind_map[elem_block];
            int istart = nnz_per_block * glob_block_ind;
            T *val = &valPtr[istart];
            // printf("ielem %d, glob_block_ind %d, elem_block %d\n", ielem,
            //        glob_block_ind, elem_block);
            int elem_block_row = elem_block / nodes_per_elem;
            int elem_block_col = elem_block % nodes_per_elem;

            // int gblock = istart / nnz_per_block;

            // loop over each nz in each block of kelem
            for (int inz = 0; inz < nnz_per_block; inz++) {
                int local_row = inz / block_dim;
                int local_col = inz % block_dim;
                int erow = block_dim * elem_block_row + local_row;
                int ecol = block_dim * elem_block_col + local_col;

                // printf("%d,%d,%d,%d,%d,%.4e\n", ielem, glob_block_ind,
                //        elem_block, erow, ecol,
                //        scale * shared_elem_mat[dof_per_elem * erow + ecol]);

                atomicAdd(&val[inz],
                          scale * shared_elem_mat[dof_per_elem * erow + ecol]);
            }
        }

        // V2 with elem_conn directly (might be slower without elem_ind_map)
        // const int32_t *rowPtr = bsr_data.rowPtr;
        // const int32_t *colPtr = bsr_data.colPtr;
        // for (int elem_block = start; elem_block < blocks_per_elem;
        //      elem_block += stride) {

        //     // use elem_conn to figure out where this is in the global matrix
        //     // first block row and columns (outer block/nodal level)
        //     int elem_block_row = elem_block / nodes_per_elem;
        //     int elem_block_col = elem_block % nodes_per_elem;

        //     // now global block row, col (global nodes)
        //     int glob_block_row = elem_conn[elem_block_row];
        //     int glob_block_col = elem_conn[elem_block_col];

        //     // TODO : change this back to using elem_ind_map? so faster..

        //     // use colPtr to find the global locations
        //     for (int col_ptr_ind = rowPtr[glob_block_row];
        //          col_ptr_ind < rowPtr[glob_block_row + 1]; col_ptr_ind++) {

        //         T *val = &valPtr[nnz_per_block * col_ptr_ind];

        //         int check_block_col = colPtr[col_ptr_ind];
        //         if (check_block_col == glob_block_col) {
        //             // add values in now
        //             T *val = &valPtr[nnz_per_block * col_ptr_ind];

        //             // loop over each nz in each block of kelem
        //             for (int inz = 0; inz < nnz_per_block; inz++) {
        //                 int inner_row = inz / block_dim;
        //                 int inner_col = inz % block_dim;
        //                 int erow = block_dim * elem_block_row + inner_row;
        //                 int ecol = block_dim * elem_block_col + inner_col;

        //                 int gblock = col_ptr_ind;

        //                 printf("%d,%d,%d,%d,%d,%.4e\n", ielem, gblock,
        //                        elem_block, erow, ecol,
        //                        scale *
        //                            shared_elem_mat[dof_per_elem * erow +
        //                            ecol]);

        //                 atomicAdd(
        //                     &val[inz],
        //                     scale *
        //                         shared_elem_mat[dof_per_elem * erow + ecol]);
        //             }

        //             break;
        //         }
        //     }
        // }
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
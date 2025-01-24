template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_mat_bcs_rows_kernel(Vec<int> bcs, const int *rowPtr, const int *colPtr,
                                 int32_t nnodes, T *values,
                                 int32_t blocks_per_elem, int32_t nnz_per_block,
                                 int32_t block_dim) {
    // apply bcs to the rows of the matrix
    int this_thread_bc = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x;
    int num_bcs = bcs.getSize();

    // assume num_bcs is reasonable that you can parallelize over it
    // with one bc per thread
    // if very large, we can change parallelization strategy
    if (this_thread_bc < num_bcs) {
        // get data associated with the row of this BC in the BSR matrix
        int bc_dof = bcs[this_thread_bc];
        int global_row = bc_dof;
        int block_row =  bc_dof / block_dim;
        int inner_row = bc_dof % block_dim;
        int istart = rowPtr[block_row];
        int iend = rowPtr[block_row + 1];
        T *val = &values[nnz_per_block * istart];

        // now loop over all columns, setting this entire row zero
        for (int col_ptr_ind = istart; col_ptr_ind < iend; col_ptr_ind++) {
            int block_col = colPtr[col_ptr_ind];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int inz = block_dim * inner_row + inner_col;
                int global_col = block_dim * block_col + inner_col;

                // zeros out all entries in this BC row except for
                // the diagonal entry which we set to 1
                // ternary operation here should prevent warp divergence
                val[inz] = (global_col == global_row) ? 1.0 : 0.0;
            }
            val += nnz_per_block; // go to the next nonzero block (on this row)
        }
    }

}
    
template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_mat_bcs_cols_kernel(Vec<int> bcs, 
                                 const int *transpose_rowPtr, const int *transpose_colPtr,
                                 const int *transpose_block_map,
                                 int32_t nnodes, T *values,
                                 int32_t blocks_per_elem, int32_t nnz_per_block,
                                 int32_t block_dim) {
    // apply bcs to the rows of the matrix
    int this_thread_bc = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x;
    int num_bcs = bcs.getSize();

    // assume num_bcs is reasonable that you can parallelize over it
    // with one bc per thread
    // if very large, we can change parallelization strategy

    // here we use the transpose rowPtr and colPtr which makes it easier
    // to parallelize over the bc columns now
    // we now have each thread zero out one BC column

    if (this_thread_bc < num_bcs) {
        // get data associated with the row of this BC in the BSR matrix
        int bc_dof = bcs[this_thread_bc];
        int global_col = bc_dof;
        int block_col =  bc_dof / block_dim;
        int inner_col = bc_dof % block_dim;
        // the convention here is that the transpose_rowPtr
        // takes in a block_col and the transpose_colPtr
        // indicates location of sparse rows in the original matrix
        int istart = transpose_rowPtr[block_col];
        int iend = transpose_rowPtr[block_col + 1];

        // now loop over all columns, setting this entire row zero
        for (int row_ptr_ind = istart; row_ptr_ind < iend; row_ptr_ind++) {
            int block_row = transpose_colPtr[row_ptr_ind];
            for (int inner_row = 0; inner_row < block_dim; inner_row++) {
                int inz = block_dim * inner_row + inner_col;
                int global_row = block_dim * block_row + inner_row;

                int values_block_ind = transpose_block_map[row_ptr_ind];
                // printf("thread %d, bc %d, tr_block %d setting glob_block %d zero\n", 
                //     this_thread_bc, bc_dof, row_ptr_ind, values_block_ind);

                // zeros out all entries in this BC col except for
                // the diagonal entry which we set to 1
                // ternary operation here should prevent warp divergence
                values[nnz_per_block * values_block_ind + inz] = (global_col == global_row) ? 1.0 : 0.0;
            }
        }
    }

}
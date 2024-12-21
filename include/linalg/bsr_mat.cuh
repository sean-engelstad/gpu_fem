template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_mat_bcs_kernel(Vec<int> bcs, const int *rowPtr, const int *colPtr,
                                 int32_t nnodes, T *values,
                                 int32_t blocks_per_elem, int32_t nnz_per_block,
                                 int32_t block_dim) {

    // loop over each bc
    for (int ibc = blockIdx.x * blockDim.x + threadIdx.x; ibc < bcs.getSize(); ibc += blockDim.x) {
        int node = bcs[ibc];
        int inner_row = node % block_dim;
        int block_row = node / block_dim;
        int istart = rowPtr[block_row];
        int iend = rowPtr[block_row + 1];
        T *val = &values[nnz_per_block * istart];

        // set bc row to zero and diag to 1
        for (int col_ptr_ind = istart; col_ptr_ind < iend; col_ptr_ind++) {
            int block_col = colPtr[col_ptr_ind];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int inz = block_dim * inner_row + inner_col;
                int glob_col = block_col * block_dim + inner_col;
                // ternary operation is more friendly on the GPU (even
                // though this is CPU here)
                val[inz] = (glob_col == node) ? 1.0 : 0.0;
            }
            val += nnz_per_block;
        }

        // TODO : zero bc columns for nonzero disp bcs
    } // end of BC loop
}

// TODO : for bc column want K^T rowPtr, colPtr map probably
// to do it without if statements (compute map on CPU assembler init)
// NOTE : zero out columns only needed for nonzero BCs

// set bc col to zero (template code that needs improvement for
// parallelization) for (int inode = 0; inode < nnodes; inode++) {
//     block_row = inode / block_dim;
//     inner_row = inode % block_dim;
//     istart = rowPtr[block_row];
//     iend = rowPtr[block_col];
//     val = &values[nnz_per_block * istart];
//     int dest_block_col = node / block_dim;

//     for (int *block_col = &colPtr[istart];
//             block_col < &colPtr[iend]; block_col++) {
//         if (dest_block_col == block_col[0]) {
//             // now iterate over inner cols
//             for (int inner_col = 0; inner_col < block_dim;
//                     inner_col++) {
//                 int inz = block_dim * inner_row + inner_col;
//                 val[inz] = 0.0;
//             }
//         }
//     }
//     val += nnz_per_block;
// }
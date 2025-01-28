template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_vec_bcs_kernel(Vec<int> bcs, T *data) {
    int nbcs = bcs.getSize();
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x) {
        int idof = bcs[ibc];
        data[idof] = 0.0;
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void permute_vec_kernel(int N, T *data, T *temp, int block_dim, int *perm) {
    
    // TODO : could also have each block in the grid do more work with for loop
    // for larger problems?

    // copying data from data to temp using permutation
    int orig_idof = blockIdx.x * blockDim.x + threadIdx.x;
    int perm_idof = Vec<T>::permuteDof(orig_idof, perm, block_dim);
    T val = data[orig_idof];
    temp[perm_idof] = val;
}
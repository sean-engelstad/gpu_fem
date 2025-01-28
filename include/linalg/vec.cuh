template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_vec_bcs_kernel(Vec<int> bcs, T *data, int block_dim, int *perm) {
    int nbcs = bcs.getSize();
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x) {
        int _idof = bcs[ibc];
        int idof = Vec<T>::permuteDof(_idof, perm, block_dim);
        data[idof] = 0.0;
    }
}
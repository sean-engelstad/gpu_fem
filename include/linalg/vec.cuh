template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_bcs_kernel(Vec<int> bcs, T *data) {
    int nbcs = bcs.getSize();
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x) {
        int idof = bcs[ibc];
        data[idof] = 0.0;
    }
}
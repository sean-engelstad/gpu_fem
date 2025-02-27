#include "cuda_utils.h"

template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_vec_bcs_kernel(Vec<int> bcs, T *data)
{
    int nbcs = bcs.getSize();
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x)
    {
        int idof = bcs[ibc];
        data[idof] = 0.0;
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void permute_vec_kernel(int num_nodes, T *data, T *temp, int block_dim, int *perm)
{

    // TODO : could also have each block in the grid do more work with for loop
    // for larger problems?

    // copying data from data to temp using permutation
    int orig_inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (orig_inode < num_nodes)
    {
        for (int inner_dof = 0; inner_dof < block_dim; inner_dof++)
        {
            int orig_idof = block_dim * orig_inode + inner_dof;
            int perm_idof = Vec<T>::permuteDof(orig_idof, perm, block_dim);
            T val = data[orig_idof];
            temp[perm_idof] = val;
        }
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void vec_add_kernel(Vec<T> v1, Vec<T> v2, Vec<T> v3)
{
    int thread_start = blockDim.x * blockIdx.x + threadIdx.x;
    int N = v1.getSize();

    int i = thread_start;
    if (i < N) {
        atomicAdd(&v3[i], v1[i] + v2[i]);
        // printf("adding ind %d\n", i);
    }

    // for (int i = thread_start; i < (thread_start + blockDim.x) && i < N; i += blockDim.x)
    // {
    //     atomicAdd(&v3[i], v1[i] + v2[i]);
    // }
}
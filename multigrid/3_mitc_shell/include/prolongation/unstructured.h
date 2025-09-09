#pragma once

#include "unstruct_utils.h"
#include "unstructured.cuh"

template <class Basis>
class UnstructuredProlongation {
   public:
    using T = double;
    static constexpr bool structured = false;

    static void prolongate(int nnodes_fine, int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems,
                           T *d_n2e_xis, int *d_coarse_iperm, int *d_fine_iperm,
                           DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
        // zero anything out again
        int N_coarse = coarse_soln_in.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = dx_fine.getSize();

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_prolongate<T, Basis>
            <<<grid, block>>>(coarse_soln_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr,
                              d_n2e_elems, d_n2e_xis, nnodes_fine, d_fine_iperm, dx_fine.getPtr());
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    static void restrict_defect(int nnodes_fine, const int *d_coarse_conn, int *d_n2e_ptr,
                                int *d_n2e_elems, T *d_n2e_xis, int *d_coarse_iperm,
                                int *d_fine_iperm, DeviceVec<T> fine_defect_in,
                                DeviceVec<T> coarse_defect_out, T *d_weights) {
        // zero anything out again
        int N_coarse =
            coarse_defect_out.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = fine_defect_in.getSize();
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_restrict<T, Basis><<<grid, block>>>(
            fine_defect_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems,
            d_n2e_xis, nnodes_fine, d_fine_iperm, coarse_defect_out.getPtr(), d_weights);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // TRY not normalizing restriction (that kind of breaks P and P^T relationship..) to
        // comment, seems to be fine either way.. this out.. normalize
        // int nblock2 = (N_coarse + 31) / 32;
        // dim3 grid2(nblock2);
        // k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_defect_out.getPtr(), d_weights);
    }
};
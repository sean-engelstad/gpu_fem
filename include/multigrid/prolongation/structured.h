// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "linalg/vec.h"
#include "structured.cuh"

template <ProlongationGeom geom>
class StructuredProlongation {
  public:
    using T = double;
    static constexpr bool structured = true;
    static constexpr bool assembly = false;

    static void prolongate(int nelems_fine, int *d_coarse_iperm, int *d_fine_iperm,
                           DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine, T *d_weights) {
        // zero temp so we can store dx in it
        int N_coarse = coarse_soln_in.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = dx_fine.getSize();
        // int nnodes_fine = N_fine / 6;
        // int nxe_fine = sqrt((float)nnodes_fine) - 1;
        int nxe_fine = sqrt((float)nelems_fine);
        int nxe_coarse = nxe_fine / 2;
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        // launch kernel so coalesced with every group of N_coarse threads covering whole domain
        // then repeats with extra group of grids (9x for 9x adjacent fine nodes of FD stencil)
        dim3 block(32);
        int nblocks_x = (nelems_fine + 31) / 32;
        dim3 grid(nblocks_x);

        k_plate_prolongate<T, geom><<<grid, block>>>(nxe_coarse, nxe_fine, nelems_fine, d_coarse_iperm,
                                               d_fine_iperm, coarse_soln_in.getPtr(),
                                               dx_fine.getPtr(), d_weights);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("end of prolongate\n");

        // now normalize by the weights so partition of unity remains
        int nblock2 = (N_fine + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_fine, dx_fine.getPtr(), d_weights);
    }

    static void restrict_defect(int nelems_fine, int *d_coarse_iperm, int *d_fine_iperm,
                                DeviceVec<T> fine_defect_in, DeviceVec<T> coarse_defect_out,
                                T *d_weights) {
        // zero temp so we can store dx in it
        int N_coarse =
            coarse_defect_out.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = fine_defect_in.getSize();
        // int nnodes_fine = N_fine / 6;
        // int nxe_fine = sqrt((float)nnodes_fine) - 1;
        int nxe_fine = sqrt((float)nelems_fine);
        int nxe_coarse = nxe_fine / 2;
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        // launch kernel so coalesced with every group of N_coarse threads covering whole domain
        // then repeats with extra group of grids (9x for 9x adjacent fine nodes of FD stencil)
        dim3 block(32);
        int nblocks_x = (nelems_fine + 31) / 32;
        dim3 grid(nblocks_x);

        k_plate_restrict<T, geom><<<grid, block>>>(nxe_coarse, nxe_fine, nelems_fine, d_coarse_iperm,
                                             d_fine_iperm, fine_defect_in.getPtr(),
                                             coarse_defect_out.getPtr(), d_weights);

        // now normalize by the weights so partition of unity remains
        int nblock2 = (N_coarse + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_defect_out.getPtr(), d_weights);
        // I actually use the fine vec d_int_temp here for convenience (it's fine cause it's larger
        // than coarse)
    }
};
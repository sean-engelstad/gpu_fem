// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "prolongation.cuh"

class PlateProlongation {
    using T = double;

  public:
    static void prolongate(int nelems_fine, const int *d_fine_elem_conn, int *d_coarse_iperm, int *d_fine_iperm,
        DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine, int *d_int_temp) {
        // zero temp so we can store dx in it
        int N_coarse = coarse_soln_in.getSize(); // this includes dof per node (not num nodes here)
        int N_fine = dx_fine.getSize();
        int nnodes_fine = N_fine / 6;
        int nxe_fine = sqrt((float)nnodes_fine) - 1;
        
        int nxe_coarse = nxe_fine / 2;

        // launch kernel so coalesced with every group of N_coarse threads covering whole domain
        // then repeats with extra group of grids (9x for 9x adjacent fine nodes of FD stencil)
        dim3 block(32);
        int nblocks_x = (nelems_fine + 31) / 32;
        dim3 grid(nblocks_x, 2); 

        // one cool thing is that each fine element only touches one coarse node, that's the basis of my algorithm here..
        // other option would be to use fine rowp + cols for adj nodes of each coarse node (but would also need fine iperm then..)
        // call even then odd elems -> cause otherwise even + odd elems share coarse node and this results in bank conflicts..
        k_plate_prolongate<T><<<grid, block>>>(nxe_coarse, nxe_fine, nelems_fine, d_fine_elem_conn, d_coarse_iperm, d_fine_iperm,
                                          coarse_soln_in.getPtr(), dx_fine.getPtr(), d_int_temp);

        // now normalize by the weights so partition of unity remains
        int nblock2 = (N_fine + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_fine, dx_fine.getPtr(), d_int_temp);
    }
};
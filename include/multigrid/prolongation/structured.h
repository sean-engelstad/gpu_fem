// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "linalg/vec.h"
#include "structured.cuh"

template <class Assembler, ProlongationGeom geom>
class StructuredProlongation {
  public:
    using T = double;
    static constexpr bool structured = true;
    static constexpr bool assembly = false;

    StructuredProlongation(Assembler &fine_assembler) {
        // extract some relevant data from the coarse and fine assemblers
        nelems_fine = fine_assembler.get_num_elements();
        d_fine_iperm = fine_assembler.getBsrData().iperm;
        ndof_fine = fine_assembler.get_num_vars();
        d_weights = DeviceVec<T>(ndof_fine);
    }
    
    void init_coarse_data(Assembler &coarse_assembler) {
        // has to be called separately (since coarse grid isn't made at same time as fine grid)
        d_coarse_iperm = coarse_assembler.getBsrData().iperm;
    }

    void prolongate(DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
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

        // now normalize by the weights so partition of unity remains
        int nblock2 = (N_fine + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_fine, dx_fine.getPtr(), d_weights);
    }

    void restrict_defect(DeviceVec<T> fine_defect_in, DeviceVec<T> coarse_defect_out) {
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

  private:
    int nelems_fine, int *d_coarse_iperm, int *d_fine_iperm;
    int ndof_fine;
    T *d_weights;
};
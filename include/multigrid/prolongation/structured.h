// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "linalg/vec.h"
#include "_structured.cuh"

template <class Assembler, ProlongationGeom geom>
class StructuredProlongation {
  public:
    using T = double;
    static constexpr bool structured = true;
    static constexpr bool assembly = false;

    StructuredProlongation(Assembler &fine_assembler) {
        // extract some relevant data from the coarse and fine assemblers
        nelems_fine = fine_assembler.get_num_elements();
        N_fine = fine_assembler.get_num_vars();
        d_fine_iperm = fine_assembler.getBsrData().iperm;
        d_weights = DeviceVec<T>(N_fine).getPtr();
        nxe_fine = sqrt((float)nelems_fine);
    }

    // no update required for structured prolongs
    void update_after_assembly() {}
    
    void init_coarse_data(Assembler &coarse_assembler) {
        // has to be called separately (since coarse grid isn't made at same time as fine grid)
        d_coarse_iperm = coarse_assembler.getBsrData().iperm;
        N_coarse = coarse_assembler.get_num_vars();
        nelems_coarse = coarse_assembler.get_num_elements();
        nxe_coarse = sqrt((float)nelems_coarse);
        assert(nxe_fine == (2 * nxe_coarse));
    }

    void prolongate(DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
        // zero weights to ensure partition of unity
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        /* launch the struct prolong kernel */
        dim3 block(32);
        int nblocks_x = (nelems_fine + 31) / 32;
        dim3 grid(nblocks_x);
        k_plate_prolongate<T, geom><<<grid, block>>>(nxe_coarse, nxe_fine, nelems_fine, d_coarse_iperm,
                                               d_fine_iperm, coarse_soln_in.getPtr(),
                                               dx_fine.getPtr(), d_weights);

        /* ensure partition of unity by weight normalization */
        int nblock2 = (N_fine + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_fine, dx_fine.getPtr(), d_weights);
    }

    template <bool normalize = false>
    void restrict_vec(DeviceVec<T> fine_vec_in, DeviceVec<T> coarse_vec_out) {
        // zero weights to ensure partition of unity 
        if constexpr (normalize) {
            cudaMemset(d_weights, 0.0, N_fine * sizeof(T));
        }

        /* launch struct restrict kernel */
        dim3 block(32);
        int nblocks_x = (nelems_fine + 31) / 32;
        dim3 grid(nblocks_x);
        k_plate_restrict<T, geom><<<grid, block>>>(nxe_coarse, nxe_fine, nelems_fine, d_coarse_iperm,
                                             d_fine_iperm, fine_vec_in.getPtr(),
                                             coarse_vec_out.getPtr(), d_weights);

        // now normalize by the weights so partition of unity remains (only for restricting soln in NL problems, not loads))
        if constexpr (normalize) {
            int nblock2 = (N_coarse + 31) / 32;
            dim3 grid2(nblock2);
            k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_vec_out.getPtr(), d_weights);
        }
    }

  private:
    int N_coarse, N_fine, nelems_coarse, nelems_fine;
    int nxe_coarse, nxe_fine;
    int *d_coarse_iperm, *d_fine_iperm;
    T *d_weights;
};
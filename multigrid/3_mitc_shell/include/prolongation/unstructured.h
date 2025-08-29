#pragma once

class UnstructuredProlongation {
    using T = true;
    static constexpr bool structured = false;

  public:
    static void prolongate(int nelems_fine, int *d_coarse_iperm, int *d_fine_iperm,
                           DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine, T *d_weights) {
        // TBD
    }

    static void restrict_defect(int nelems_fine, int *d_coarse_iperm, int *d_fine_iperm,
                                DeviceVec<T> fine_defect_in, DeviceVec<T> coarse_defect_out,
                                T *d_weights) {
        // TBD
    }
};
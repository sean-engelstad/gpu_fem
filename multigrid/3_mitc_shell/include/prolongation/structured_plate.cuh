// helper GPU kernels for geometry-specific prolongation operators
// NOTE : I will write more general one when it comes to wingbox later..

template <typename T>
__global__ static void k_plate_prolongate(const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *coarse_soln_in, T *dx_fine, T *d_fine_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid; // int odd_ielem = _ielem % 2 == 1; // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int fine_elem = tid;
    if (fine_elem >= nelems_fine) return;
    // TODO : on other meshes, use elem conn and some graph maps (not plate method here which is simpler)

    // write it first, then we'll add some device helper methods?
    int nx_c = nxe_coarse + 1, nx_f = nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    // int coarse_elem = nxe_coarse * iye_c + ixe_c;

    for (int local_fine = 0; local_fine < 4; local_fine++) {
        int ix_f = ixe_f + local_fine % 2, iy_f = iye_f + local_fine / 2;

        for (int local_coarse = 0; local_coarse < 4; local_coarse++) {
            int ix_c = ixe_c + local_coarse % 2, iy_c = iye_c + local_coarse / 2;

            // now compute dx and dy between coarse and fine nodes..
            int ix_cf = 2 * ix_c, iy_cf = 2 * iy_c;
            int dx = abs(ix_cf - ix_f), dy = abs(iy_cf - iy_f);

            // diff cases with adjacencies
            int case1 = dx == 0 && dy == 0; // fine node matches coarse
            int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
            int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem

            // scaling from FEA basis
            T scale = case1 * 1.0 + case2 * 0.5 + case3 * 0.25;

            // get fine and coarse indices now..
            int coarse_node = nx_c * iy_c + ix_c;
            int perm_coarse_node = d_coarse_iperm[coarse_node];
            int fine_node = iy_f * nx_f + ix_f;
            int perm_fine_node = d_fine_iperm[fine_node];

            // now loop over each DOF..
            for (int idof = 0; idof < 6; idof++) {
                int coarse_dof = 6 * perm_coarse_node + idof;
                int fine_dof = 6 * perm_fine_node + idof;
                T val = coarse_soln_in[coarse_dof];
                val *= scale;

                atomicAdd(&dx_fine[fine_dof], val);
                atomicAdd(&d_fine_wts[fine_dof], scale);
            }
        }
    }

}

template <typename T>
__global__ static void k_plate_restrict(const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *defect_fine_in, T *defect_coarse_out, T *d_coarse_wts) {
    // restriction for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid; // int odd_ielem = _ielem % 2 == 1; // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int fine_elem = tid;
    if (fine_elem >= nelems_fine) return;
    // TODO : on other meshes, use elem conn and some graph maps (not plate method here which is simpler)
    
    // write it first, then we'll add some device helper methods?
    int nx_c = nxe_coarse + 1, nx_f = nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    // int coarse_elem = nxe_coarse * iye_c + ixe_c;

    for (int local_fine = 0; local_fine < 4; local_fine++) {
        int ix_f = ixe_f + local_fine % 2, iy_f = iye_f + local_fine / 2;

        for (int local_coarse = 0; local_coarse < 4; local_coarse++) {
            int ix_c = ixe_c + local_coarse % 2, iy_c = iye_c + local_coarse / 2;

            // now compute dx and dy between coarse and fine nodes..
            int ix_cf = 2 * ix_c, iy_cf = 2 * iy_c;
            int dx = abs(ix_cf - ix_f), dy = abs(iy_cf - iy_f);

            // diff cases with adjacencies
            int case1 = dx == 0 && dy == 0; // fine node matches coarse
            int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
            int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem

            // scaling from FEA basis
            T scale = case1 * 1.0 + case2 * 0.5 + case3 * 0.25;

            // get fine and coarse indices now..
            int coarse_node = nx_c * iy_c + ix_c;
            int perm_coarse_node = d_coarse_iperm[coarse_node];
            int fine_node = iy_f * nx_f + ix_f;
            int perm_fine_node = d_fine_iperm[fine_node];

            // now loop over each DOF..
            for (int idof = 0; idof < 6; idof++) {
                int coarse_dof = 6 * perm_coarse_node + idof;
                int fine_dof = 6 * perm_fine_node + idof;
                T val = defect_fine_in[fine_dof];
                val *= scale;

                atomicAdd(&defect_coarse_out[coarse_dof], defect_fine_in[fine_dof]);
                atomicAdd(&d_coarse_wts[coarse_dof], scale);
            }
        }
    }
}

template <typename T>
__global__ static void k_vec_normalize(int N, T *vec_in, T *weights) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        vec_in[tid] /= (weights[tid] + 1e-12);
    }
}

/* helper device functions */
// template <typename T>
// __device__ void d_get_
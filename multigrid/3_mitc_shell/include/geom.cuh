// helper GPU kernels for geometry-specific prolongation operators
// NOTE : I will write more general one when it comes to wingbox later..

template <typename T>
__global__ static void k_plate_prolongate(const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_elem_conn, const int *d_coarse_perm, const int *d_fine_perm,
    const T *coarse_soln_in, T *dx_fine, int *d_fine_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int _ielem = tid;
    int odd_ielem = _ielem % 2 == 1;
    int ielem = _ielem / 2 + nelems / 2 * (1 - odd_ielem);
    if (ielem >= nelems_fine) return;

    // get the coarse node for this element as fine ind (there's only one bc)
    int nx_c = nxe_coarse + 1, nx_f = nxe_fine + 1;
    int coarse_ind_f = -1;
    for (int local_node = 0; local_node < 4; local_node++) {
        int inode = d_elem_conn[4 * ielem + local_node];
        int ix = inode % nx_f, iy = inode % nx_f;
        int is_coarse = ix % 2 == 0 && iy % 2 == 0;
        coarse_ind_f += is_coarse * inode;
    }
    int ix_0 = coarse_ind_f % nx_f, iy_0 = coarse_ind_f / nx_f;
    // also get the actual coarse ind
    int coarse_ind_c = nx_c * (iy_0 / 2) + ix_0 / 2;
    int perm_coarse_ind_c = d_coarse_perm[coarse_ind_c];
    __syncthreads(); // divides registers
    
    for (int idof = 0; idof < 6; idof++) {
        T coarse_val = coarse_soln_in[6 * perm_coarse_ind_c + idof];

        // now loop over 4 connected nodes in the element..
        for (int local_node = 0; local_node < 4; local_node++) {
            int inode = d_elem_conn[4 * ielem + local_node];
            int ix = inode % nx_f, iy = inode / nx_f;

            // since it's just linear interp => no need to compute basis functions
            // weight normalization is sufficient..
            int fine_ind = iy * nx_f + ix;
            int perm_fine_ind = d_fine_perm[fine_ind];
            
            atomicAdd(&dx_fine[6 * perm_fine_ind + idof], coarse_val);
            atomicAdd(&d_fine_wts[6 * perm_fine_ind + idof], 1);
        }
    }
}

template <typename T>
__global__ static void k_vec_normalize(int N, T *vec_in, int *weights) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        vec_in[tid] /= weights[tid];
    }
}
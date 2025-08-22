// helper GPU kernels for geometry-specific prolongation operators
// NOTE : I will write more general one when it comes to wingbox later..

template <typename T>
__global__ static void k_plate_prolongate(const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_elem_conn, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *coarse_soln_in, T *dx_fine, int *d_fine_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid;
    // int odd_ielem = _ielem % 2 == 1;
    // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int ielem = tid;
    if (ielem >= nelems_fine) return;

    // get the coarse node for this element as fine ind (there's only one bc)
    int nx_c = nxe_coarse + 1, nx_f = nxe_fine + 1;
    int coarse_ind_f = -1;
    for (int local_node = 0; local_node < 4; local_node++) {
        int _inode = d_elem_conn[4 * ielem + local_node];
        int ix = _inode % nx_f, iy = _inode / nx_f;
        int is_coarse = ix % 2 == 0 && iy % 2 == 0;
        coarse_ind_f += is_coarse * _inode;
    }
    int ix_0 = coarse_ind_f % nx_f, iy_0 = coarse_ind_f / nx_f;
    // also get the actual coarse ind
    int coarse_ind_c = nx_c * (iy_0 / 2) + ix_0 / 2;
    // printf("coarse_ind_f = %d : ix_0 %d, iy_0 %d => coarse_ind_c %d\n", coarse_ind_f, ix_0, iy_0, coarse_ind_c);
    int perm_coarse_ind_c = d_coarse_iperm[coarse_ind_c];
    __syncthreads(); // divides registers    

    // now loop over 4 connected nodes in the element..
    for (int local_node = 0; local_node < 4; local_node++) {
        int inode = d_elem_conn[4 * ielem + local_node];
        // if (inode < 3) 
            // printf("ielem %d, local_node %d => inode %d\n", ielem, local_node, inode);

        for (int idof = 0; idof < 6; idof++) {
            T coarse_val = coarse_soln_in[6 * perm_coarse_ind_c + idof];

            // since it's just linear interp => no need to compute basis functions
            // weight normalization is sufficient..
            int perm_fine_ind = d_fine_iperm[inode];

            // if (inode < 10 && idof == 2) {
            //     printf("tid %d, ielem %d: coarse node %d => fine node %d; permuted %d => %d\n", tid, ielem, coarse_ind_c, inode, perm_coarse_ind_c, perm_fine_ind);
            //     printf("tid %d, coarse val %.2e\n", tid, coarse_val);
            // }

            // DEBUG
            // int ix = inode % nx_f, iy = inode / nx_f;
            // if (ix == 1 && idof == 2 && inode < 70) {
            //     printf("prolong: ielem %d, inode %d, ix %d, iy %d => perm node %d with w-val %.2e\n", ielem, inode, ix, iy, perm_fine_ind, coarse_val);
            // }
            
            atomicAdd(&dx_fine[6 * perm_fine_ind + idof], coarse_val);
            atomicAdd(&d_fine_wts[6 * perm_fine_ind + idof], 1);
        }
    }
}

template <typename T>
__global__ static void k_plate_restrict(const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_elem_conn, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *defect_fine_in, T *defect_coarse_out, int *d_coarse_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid;
    // int odd_ielem = _ielem % 2 == 1;
    // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int ielem = tid;
    if (ielem >= nelems_fine) return;

    // get the coarse node for this element as fine ind (there's only one bc)
    int nx_c = nxe_coarse + 1, nx_f = nxe_fine + 1;
    int coarse_ind_f = -1;
    for (int local_node = 0; local_node < 4; local_node++) {
        int _inode = d_elem_conn[4 * ielem + local_node];
        int ix = _inode % nx_f, iy = _inode / nx_f;
        int is_coarse = ix % 2 == 0 && iy % 2 == 0;
        coarse_ind_f += is_coarse * _inode;
    }
    int ix_0 = coarse_ind_f % nx_f, iy_0 = coarse_ind_f / nx_f;
    // also get the actual coarse ind
    int coarse_ind_c = nx_c * (iy_0 / 2) + ix_0 / 2;
    // printf("coarse_ind_f = %d : ix_0 %d, iy_0 %d => coarse_ind_c %d\n", coarse_ind_f, ix_0, iy_0, coarse_ind_c);
    int perm_coarse_ind_c = d_coarse_iperm[coarse_ind_c];
    __syncthreads(); // divides registers
    

    // now loop over 4 connected nodes in the element..
    for (int local_node = 0; local_node < 4; local_node++) {
        int inode = d_elem_conn[4 * ielem + local_node];

        for (int idof = 0; idof < 6; idof++) {
            // since it's just linear interp => no need to compute basis functions
            // weight normalization is sufficient..
            int perm_fine_ind = d_fine_iperm[inode];

            T fine_val = defect_fine_in[6 * perm_fine_ind + idof];

            // DEBUG
            // int ix = inode % nx_f, iy = inode / nx_f;
            // if (ix == 1 && idof == 2 && inode < 70) {
            //     printf("restrict: ielem %d, inode %d, ix %d, iy %d => perm node %d with w-val %.2e\n", ielem, inode, ix, iy, perm_fine_ind, fine_val);
            //     printf("nx_f %d, nxe_f %d\n", nx_f, nxe_fine);
            // }

            // printf("coarse node %d => fine node %d; permuted %d => %d\n", coarse_ind_c, inode, perm_coarse_ind_c, perm_fine_ind);
            // printf("coarse defect[%d] += %.4e\n", 6 * perm_coarse_ind_c + idof, fine_val);
            
            atomicAdd(&defect_coarse_out[6 * perm_coarse_ind_c + idof], fine_val);
            atomicAdd(&d_coarse_wts[6 * perm_coarse_ind_c + idof], 1);
        }
    }
}

template <typename T>
__global__ static void k_vec_normalize(int N, T *vec_in, int *weights) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        vec_in[tid] /= (weights[tid] + 1e-12);
    }
}
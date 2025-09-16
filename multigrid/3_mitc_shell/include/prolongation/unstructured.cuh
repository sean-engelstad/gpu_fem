#pragma once

/* kernel functions now.. */

template <typename T, class Basis>
__global__ static void k_unstruct_prolongate(const T *coarse_soln, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *fine_soln) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];
    
    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get local coarse elem disps.. (with permutations here?)
        T c_elem_disps[24];
        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            c_elem_disps[i] = coarse_soln[6 * perm_coarse_node + loc_dof];
        }

        // interp the disps from coarse disps
        T fine_disp_add[6];
        Basis::template interpFields<6, 6>(pt, c_elem_disps, fine_disp_add);

        // if (fine_node == 2042) {
        // if (fine_node == 1977) {
        //     int ix_f = fine_node % 65, iy_f = fine_node / 65;
        //     printf("prolong, fine node %d : ix_f %d, iy_f %d\n", fine_node, ix_f, iy_f);
        //     printf("\tfrom coarse element %d with xi %.2f, eta %.2f\n", ielem_c, pt[0], pt[1]);
        // }

        // now add into the fine solution and fine weights
        T scale = 1.0 / (double) num_attached_elems;
        for (int idof = 0; idof < 6; idof++) {
            atomicAdd(&fine_soln[6 * perm_fine_node + idof], fine_disp_add[idof] * scale);
        }
    }
}

template <typename T, class Basis>
__global__ static void k_unstruct_restrict(const T *fine_defect_in, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *coarse_soln_out, T *coarse_wts) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];

    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get fine defect disps..
        const T *fine_nodal_defect = &fine_defect_in[6 * perm_fine_node];

        // now do interpFieldsTranspose to coarse defect on each node in element
        T coarse_elem_defect[24];
        memset(coarse_elem_defect, 0.0, 24 * sizeof(T));
        Basis::template interpFieldsTranspose<6, 6>(pt, fine_nodal_defect, coarse_elem_defect);

        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            T scale = 1.0 / (double) num_attached_elems;
            atomicAdd(&coarse_soln_out[6 * perm_coarse_node + loc_dof], 
                coarse_elem_defect[i] * scale);
            atomicAdd(&coarse_wts[6 * perm_coarse_node + loc_dof], scale);
        }
    }
}

template <typename T, class Basis>
__global__ static void k_prolong_mat_assembly(const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, int *d_rowp, int *d_cols, int block_dim, T *d_vals) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];
    int block_dim2 = block_dim * block_dim;
    
    // attached element loop
    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // TODO : make this part faster.. not efficient yet (getting some basis coefficients here)
        // useful to use transpose product to get the FEA basis.. instead of derivs
        // not sure this part actually works though..
        T c_elem_vals[24];
        memset(c_elem_vals, 0.0, 24 * sizeof(T));
        T f_node_vals[6];
        memset(f_node_vals, 0.0, 6 * sizeof(T));
        f_node_vals[0] = 1.0;
        Basis::template interpFieldsTranspose<6, 6>(pt, f_node_vals, c_elem_vals);

        // if (perm_fine_node == 2) {
        //     printf("c_elem_vals on celem %d: ", ielem_c);
        //     printVec<T>(24, c_elem_vals);
        //     printf("xis on celem %d: ", ielem_c);
        //     printVec<T>(2, pt);
        //     printf("celem_nodes on celem %d: (block_dim = %d): ", ielem_c, block_dim);
        //     for (int m = 0; m < 4; m++) {
        //         printf("%d ", d_coarse_iperm[c_elem_nodes[m]]);
        //     }
        //     printf("\n");
        //     // printVec<int>(4, c_elem_nodes);
        // }

        // now entries 0,6,12, .. etc hold the coefficients of interp to each node (same for each [0,6) dof)
        T scale = 1.0 / (double) num_attached_elems;
        // dof and local node in element loop
        for (int i = 0; i < 4; i++) {
            // int loc_node = i / 6, loc_dof = i % 6;
            int loc_node = i; // in CSR version (same operator for each node)
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            T N_cf = c_elem_vals[6 * loc_node];
            // now find the cols to add this into prolong matrix.. definitely could be more efficient here.. come back to this
            for (int jp2 = d_rowp[perm_fine_node]; jp2 < d_rowp[perm_fine_node + 1]; jp2++) {
                int col = d_cols[jp2];
                T scale2 = scale * (col == perm_coarse_node);

                // only add into diagonal entries in each nodal block.. (NOTE : this may be inefficient then, we'll see..)
                // int P_nz_ind = block_dim2 * jp2 + block_dim * loc_dof + loc_dof;
                int P_nz_ind = jp2; // in new CSR version (same for each node)

                // if (block_dim == 1) {
                // csr case, divide scal2 by block_dim (temp hack)
                // scale2 /= 6;
                // }

                atomicAdd(&d_vals[P_nz_ind], scale2 * N_cf);
            } // end of loop through that row
        } // end of loop through the local elem dof
    } // end of attached element loop
}

template <typename T, class Basis>
__global__ static void k_restrict_mat_assembly(const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, int *d_rowp, int *d_cols, int block_dim, T *d_vals) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    int block_dim2 = block_dim * block_dim;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];

    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // TODO : make this part faster.. not efficient yet (getting some basis coefficients here)
        // useful to use transpose product to get the FEA basis.. instead of derivs
        T c_elem_vals[24];
        memset(c_elem_vals, 0.0, 24 * sizeof(T));
        T f_node_vals[6];
        memset(f_node_vals, 0.0, 6 * sizeof(T));
        f_node_vals[0] = 1.0;
        Basis::template interpFieldsTranspose<6, 6>(pt, f_node_vals, c_elem_vals);

        // now entries 0,6,12, .. etc hold the coefficients of interp to each node (same for each [0,6) dof)
        T scale = 1.0 / (double) num_attached_elems;
        // dof and local node in element loop
        for (int i = 0; i < 4; i++) {
            // int loc_node = i / 6, loc_dof = i % 6;
            int loc_node = i; // in CSR version (same operator for each node)
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            T N_cf = c_elem_vals[6 * loc_node];
            // now find the cols to add this into prolong matrix.. definitely could be more efficient here.. come back to this
            for (int jp2 = d_rowp[perm_coarse_node]; jp2 < d_rowp[perm_coarse_node + 1]; jp2++) {
                int col = d_cols[jp2];
                T scale2 = scale * (col == perm_fine_node);
                // if (block_dim == 1) {
                    // csr case, divide scal2 by block_dim (temp hack)
                // scale2 /= 6;
                // }

                // only add into diagonal entries in each nodal block.. (NOTE : this may be inefficient then, we'll see..)
                // int PT_nz_ind = block_dim2 * jp2 + block_dim * loc_dof + loc_dof;
                int PT_nz_ind = jp2; // in new CSR version (same for each node)

                atomicAdd(&d_vals[PT_nz_ind], scale2 * N_cf);
            } // end of loop through that row
        } // end of loop through the local elem dof
    } // end of attached element loop
}

template <typename T>
__global__ static void k_csr_mat_vec(const int nnzb, const int block_dim, const int *d_rows, const int *d_cols, const T *d_vals, const T *vec_in, T *vec_out) {
    // fast CSR mat-vec kernel (does same prolong / restrict for every dof per node)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nnzb) return;

    // pseudo-csr block style prod..
    int row = d_rows[tid];
    int col = d_cols[tid];
    T coeff = d_vals[tid];


    for (int idof = 0; idof < 6; idof++) {
        T val_in = vec_in[block_dim * col + idof];
        // if (row == 500 && idof == 2) {
        //     int dof_out = block_dim * row + idof;
        //     printf("cpnode %d to fpnode %d, idof %d with val_in %.2e, A[r,c] %.2e and val_out %.2e\n", col, row, idof, val_in, coeff, val_in * coeff);
        // }
        atomicAdd(&vec_out[block_dim * row + idof], coeff * val_in);
    }
}
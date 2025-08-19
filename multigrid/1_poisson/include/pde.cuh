/* helper device methods with prefix d_ for device */
__device__ static int d_getElemInd() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ static int d_getElemGlobalNode(int nxe, int ielem, int local_node) {
    int ixe = ielem % nxe, iye = ielem / nxe;
    int nx = nxe + 1;  // get num nodes on line
    int node1 = iye * nx + ixe;

    // get local node indices (in 2x2 elem) => for global node ind now
    int lx = local_node % 2, ly = local_node / 2;
    return node1 + lx + nx * ly;
}

__device__ static int d_getFDConnectedNodePair(int nx, int row, int col) {
    // determine whether global row and col are connected in finite diff stencil
    int rx = row % nx, ry = row / nx;
    int cx = col % nx, cy = col / nx;
    int dx = abs(cx - rx), dy = abs(cy - ry);
    int case1 = dx == 1 && dy == 0;
    int case2 = dx == 0 && dy == 1;
    int case3 = dx == 0 && dy == 0;
    return case1 || case2 || case3;
}

/* helper kernel functions with prefix k_ for kernel function*/
__global__ static void k_vecset(int N, int value, int *d_vec) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    // cuda memset fails with large values here.. (vectorize only works with small values..)
    if (ind < N) {
        d_vec[ind] = value;
    }
}

__global__ static void k_minElemsPerNode(int nxe, int nelems, int *d_node_min_elems) {
    // for each node get the min elem_ind connected to it
    int ielem = d_getElemInd(), local_node = blockIdx.y;
    if (ielem >= nelems) return;

    int global_node = d_getElemGlobalNode(nxe, ielem, local_node);
    atomicMin(&d_node_min_elems[global_node], ielem);
}

template <typename I>
__global__ static void k_csrNUNIQRowCounts(int nxe, int nelems, int *d_csrNUNIQRowCounts) {
    // non-unique row counts
    int ielem = d_getElemInd();
    if (ielem >= nelems) return;

    // get the i,j for local node indices (i,j) each in [0,1,2,3] of elem
    int local_ind = blockIdx.y;  // again block idx not thread idx used so that first group of
                                    // nelems threads does first (0,0) pair of each element
    // then we do the (1,1) pairs, etc., this makes the adds coalesced
    // we actually want to add loc cols j on consecutive threads for coalescing
    int i = local_ind / 4, j = local_ind % 4;
    int g_row = d_getElemGlobalNode(nxe, ielem, i);
    int g_col = d_getElemGlobalNode(nxe, ielem, j);

    int connected = d_getFDConnectedNodePair(nxe+1, g_row, g_col);

    // now we write into the main
    atomicAdd(&d_csrNUNIQRowCounts[g_row], connected);
}

template <typename I>
__global__ static void k_uniqueRowCount(int nx, int N, int *d_minConnectedNode, int *d_csrRowCounts, I *d_csr_nnz) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        int validConnectedCol = d_minConnectedNode[ind] < N; // if it's above N, there were no min connected cols left => it adds 0

        // check also the two nodes are connected in FD stencil
        int connected = d_getFDConnectedNodePair(nx, ind, validConnectedCol) * validConnectedCol;

        d_csrRowCounts[ind] += validConnectedCol;
        atomicAdd(d_csr_nnz, validConnectedCol);
    }
}

__global__ static void k_csrMaxRowCount(int N, int *d_csrRowCounts, int *d_maxRowCount) {
    // get the max row count
    // could also use thrust to do it
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicMax(d_maxRowCount, d_csrRowCounts[tid]);
    }
}

__global__ static void k_getMinConnectedNode(int nxe, int nrows, int nelems, int *d_minConnectedNode_in, int *d_minConnectedNode_out) {
    // compute the number of values on CSR rowp (row counts)
    int ielem = d_getElemInd();
    if (ielem >= nelems) return;

    // get the i,j for local node indices (i,j) each in [0,1,2,3] of elem
    int local_ind = blockIdx.y;  // again block idx not thread idx used so that first group of
                                    // nelems threads does first (0,0) pair of each element
    // then we do the (1,1) pairs, etc., this makes the adds coalesced
    // we actually want to add loc cols j on consecutive threads for coalescing
    int i = local_ind / 4, j = local_ind % 4;
    int g_row = d_getElemGlobalNode(nxe, ielem, i);
    int g_col = d_getElemGlobalNode(nxe, ielem, j);

    // had syncthreads here before => that doesn't work  (cause it only syncs threads in thread block, not whole GPU)
    // syncthreads is useful for splitting up register usage nicely and dropping registers on each thread block still
    int b_larger_than_prev = g_col > d_minConnectedNode_in[g_row];

    // check also the two nodes are connected in FD stencil
    b_larger_than_prev = d_getFDConnectedNodePair(nxe + 1, g_row, g_col) * b_larger_than_prev;

    // now do atomic min, this should get min connected node larger than previous (efficiently)
    // printf("g_row %d, g_col %d, valid %d\n", g_row, g_col, b_larger_than_prev); // race condition stuff here.. (fixed by two arrays now)
    atomicMin(&d_minConnectedNode_out[g_row], g_col * b_larger_than_prev + nrows * (1 - b_larger_than_prev));
}

__global__ static void k_addMinConnectedNodeRows(int i_minNode, int N,
                                                int *d_csr_rowp, int *d_csr_rows) {
    // add the min connected node currently to the csr cols device array
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int start = d_csr_rowp[ind];
        int end = d_csr_rowp[ind + 1];
        // __syncthreads();
        int cols_ind = start + i_minNode;
        int valid = cols_ind < end;

        // valid bool helps us add without branching
        d_csr_rows[cols_ind] += ind * valid;
    }
}

__global__ static void k_addMinConnectedNodeCols(int i_minNode, int N, int *d_minConnectedNode,
                                                int *d_csr_rowp, int *d_csr_cols) {
    // add the min connected node currently to the csr cols device array
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int start = d_csr_rowp[ind];
        int end = d_csr_rowp[ind + 1];
        // __syncthreads();
        int cols_ind = start + i_minNode;
        int conn_node = d_minConnectedNode[ind];
        int valid = cols_ind < end;

        // valid bool helps us add without branching
        d_csr_cols[cols_ind] += conn_node * valid;
    }
}

// old version => slightly more mem requirements
// __global__ static void k_addMinConnectedNode(int i_minNode, int N, int *d_minConnectedNode,
//                                                 int *d_csr_rowp, int *d_csr_rows, int *d_csr_cols) {
//     // add the min connected node currently to the csr cols device array
//     int ind = blockIdx.x * blockDim.x + threadIdx.x;
//     if (ind < N) {
//         int start = d_csr_rowp[ind];
//         int end = d_csr_rowp[ind + 1];
//         int cols_ind = start + i_minNode;
//         int conn_node = d_minConnectedNode[ind];
//         int valid = cols_ind < end;

//         // valid bool helps us add without branching
//         d_csr_rows[cols_ind] += ind * valid;
//         d_csr_cols[cols_ind] += conn_node * valid;
//     }
// }

// this kernel is very expensive.. for high DOF, not great
// __global__ static void k_elemIndMap(int nxe, int nelems, int *d_rowp, int *d_cols, int *d_elem_ind_map) {
//     // compute the elem ind map from rowp, cols on GPU
//     int ielem = d_getElemInd();
//     if (ielem >= nelems) return;

//     // get the i,j for local node indices (i,j) each in [0,1,2,3] of elem
//     int local_ind = blockIdx.y;  // again block idx not thread idx used so that first group of
//                                     // nelems threads does first (0,0) pair of each element
//     // then we do the (1,1) pairs, etc., this makes the adds coalesced
//     // we actually want to add loc cols j on consecutive threads for coalescing
//     int i = local_ind / 4, j = local_ind % 4;
//     int g_row = d_getElemGlobalNode(nxe, ielem, i);
//     int g_col = d_getElemGlobalNode(nxe, ielem, j);

//     // get rowp, cols (NOTE : this part is prob not very efficient rn..) TBD on that
//     int csr_ind = 0;
//     for (int jp = d_rowp[g_row]; jp < d_rowp[g_row+1]; jp++) {
//         csr_ind += jp * (d_cols[jp] == g_col);
//     }
//     // printf("ielem %d, local_ind %d, g_row %d, g_col %d, csr_ind %d\n", ielem, local_ind, g_row, g_col, csr_ind);

//     d_elem_ind_map[16 * ielem + local_ind] = csr_ind;
// }

template <typename T>
__global__ static void k_assembleLHS(int nxe, int nnz, int nelems, int *d_rows, int *d_cols, T dx, T *d_lhs) {
    int csr_ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (csr_ind < nnz) {
        int row = d_rows[csr_ind];
        int col = d_cols[csr_ind];

        // finite diff stencil for 2d laplacian
        T val = (row == col) ? 4.0 : -1.0;
        val *= 1.0 / dx / dx;

        d_lhs[csr_ind] = val;
    }
}

template <typename T>
__global__ static void k_applyBCsLHS(int nx, int N, int csr_nnz, int *d_rows, int *d_cols, T *d_lhs) {
    // NOTE : this kernel is not very efficient yet..
    int csr_ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (csr_ind < csr_nnz) {
        T val = d_lhs[csr_ind];

        int row = d_rows[csr_ind];
        int col = d_cols[csr_ind];

        int rx = row % nx, ry = row / nx;
        int cx = col % nx, cy = col / nx;
        int r_bndry = rx == 0 || rx == nx - 1 || ry == 0 || ry == nx - 1;
        int c_bndry = cx == 0 || cx == nx - 1 || cy == 0 || cy == nx - 1;
        int bndry = r_bndry || c_bndry;
        int diag_bndry = (row == col) && bndry;

        // apply bcs to values
        val = !bndry ? val : 0.0;
        val = !diag_bndry ? val : 1.0;

        // now we write into the 
        d_lhs[csr_ind] = val;
    }
}

template <typename T>
__global__ static void k_assembleRHS(int nx, int N, T dx, T *d_rhs) {
    // assemble lhs with exponential load f(x,y) = -2 * exp(x*y)
    
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        int ix = ind % nx, iy = ind / nx;
        T x = ix * dx, y = iy * dx;
        T load = -2.0 * exp(x * y);

        // apply bcs to load here
        int bndry = (ix == 0 || ix == (nx-1) || iy == 0 || (iy == (nx - 1)));
        load = !bndry ? load : exp(x*y); // non-zero dirichlet condition on bndry

        d_rhs[ind] = load;
    }
}

template <typename T>
__global__ static void k_initSoln(int nx, int N, T dx, T *d_soln) {
    // set soln to dirichlet bc on bndry and zero elsewhere
    
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        int ix = ind % nx, iy = ind / nx;
        T x = ix * dx, y = iy * dx;

        // apply bcs to load here
        int bndry = (ix == 0 || ix == (nx-1) || iy == 0 || (iy == (nx - 1)));
        T val = !bndry ? 0.0 : exp(x*y); // non-zero dirichlet condition on bndry

        d_soln[ind] = val;
    }
}

template <typename T>
__global__ static void k_getTrueSoln(int nx, int N, T dx, T *d_true_soln) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        int ix = ind % nx, iy = ind / nx;
        T x = ix * dx, y = iy * dx;
        T val = exp(x * y);

        // true soln is exp(x * y)
        d_true_soln[ind] = val;
    }
}

template <typename T>
__global__ static void k_get_diag_inv(int N, int *d_rowp, int *d_cols, T *d_lhs, T *diag_inv) {
    // kernel to extract diag inv out..
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int start = d_rowp[row], end = d_rowp[row+1];
    diag_inv[row] = 0.0; // default
    for (int jp = start; jp < end; jp++) {
        // branching not too bad here?
        if (d_cols[jp] == row) {
            diag_inv[row] = 1.0 / d_lhs[jp];
            break;
        }
    }
}

template <typename T>
__global__ static void k_diag_inv_vec(int N, T *diag_inv, T *vec_in, T *vec_out) {
    // kenrel to compute Dinv * vec_in => vec_out
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // very simple, but not implemented in cublas / cusparse so have to do myself
    vec_out[row] = vec_in[row] * diag_inv[row];
}

template <typename T>
__global__ static void k_coarse_fine(int nxe_fine, int N_f, const T *this_coarse_vec, T *finer_vec) {
    // go from a coarser grid to this fine level
    int nxe_coarse = nxe_fine / 2;
    int nx_c = nxe_coarse + 1;
    int N_coarse = nx_c * nx_c;
    // int nxe_f = nxe_coarse * 2;
    int nx_f = 2 * nxe_coarse + 1;

    // get the fine node right on the coarse node
    int coarse_ind = threadIdx.x + blockDim.x * blockIdx.x;
    int ix_c = coarse_ind % nx_c, iy_c = coarse_ind / nx_c;
    int ix_f0 = 2 * ix_c, iy_f0 = 2 * iy_c;

    // get the current adjacent fine point we're doing..
    int i_adj = blockIdx.y; // one of the nearby 9 points
    // ix_adj, iy_adj are local offsets btw [-1,0,1] inclusive
    int ix_adj = i_adj % 3 - 1, iy_adj = i_adj / 3 - 1;
    int ix_f = ix_f0 + ix_adj, iy_f = iy_f0 + iy_adj;
    int fine_ind = nx_f * iy_f + ix_f;

    // now only perform updates if in range and if on boundary => update becomes zero
    if (0 <= fine_ind && fine_ind < N_f && coarse_ind < N_coarse) {

        // compute scale based on dx and dy offsets on fine mesh
        int case1 = ix_adj == 0 && iy_adj == 0; // coarse and fine nodes match
        int case2 = (abs(ix_adj) == 1 && iy_adj == 0) || (ix_adj == 0 && abs(iy_adj) == 1); // fine node on edge from coarse node
        int case3 = !case1 && !case2; // fine node on interior of coarse element
        T scale = case1 * 0.25 + case2 * 0.125 + case3 * 0.05625;
        scale *= 4.0; // *4 so operator is partition of unity for coarse-fine (same coeff but not *4 for fine=>coarse)

        // no update if either on bndry
        int coarse_bndry = (ix_c == 0 || ix_c == (nx_c - 1) || iy_c == 0 || iy_c == (nx_c - 1));
        int fine_bndry = (ix_f == 0 || ix_f == (nx_f - 1) || iy_f == 0 || iy_f == (nx_f - 1));
        int bndry = coarse_bndry || fine_bndry;
        scale *= !bndry; // makes it zero if bndry

        atomicAdd(&finer_vec[fine_ind], scale * this_coarse_vec[coarse_ind]);
    }
}

template <typename T>
__global__ static void k_fine_coarse(int nxe_coarse, int N_coarse, const T *finer_vec_in, T *coarse_vec_out) {
    // go from finer grid than this one => to this coarse grid defect
    int nx_c = nxe_coarse + 1;
    // int nxe_f = nxe_coarse * 2;
    int nx_f = 2 * nxe_coarse + 1;
    int N_f = nx_f * nx_f;

    // get the fine node right on the coarse node
    int coarse_ind = threadIdx.x + blockDim.x * blockIdx.x;
    int ix_c = coarse_ind % nx_c, iy_c = coarse_ind / nx_c;
    int ix_f0 = 2 * ix_c, iy_f0 = 2 * iy_c;

    // get the current adjacent fine point we're doing..
    int i_adj = blockIdx.y; // one of the nearby 9 points
    // ix_adj, iy_adj are local offsets btw [-1,0,1] inclusive
    int ix_adj = i_adj % 3 - 1, iy_adj = i_adj / 3 - 1;
    int ix_f = ix_f0 + ix_adj, iy_f = iy_f0 + iy_adj;
    int fine_ind = nx_f * iy_f + ix_f;

    // now only perform updates if in range and if on boundary => update becomes zero
    if (0 <= fine_ind && fine_ind < N_f && coarse_ind < N_coarse) {

        // compute scale based on dx and dy offsets on fine mesh
        int case1 = ix_adj == 0 && iy_adj == 0; // coarse and fine nodes match
        int case2 = (abs(ix_adj) == 1 && iy_adj == 0) || (ix_adj == 0 && abs(iy_adj) == 1); // fine node on edge from coarse node
        int case3 = !case1 && !case2; // fine node on interior of coarse element
        T scale = case1 * 0.25 + case2 * 0.125 + case3 * 0.05625;

        // no update if either on bndry
        int coarse_bndry = (ix_c == 0 || ix_c == (nx_c - 1) || iy_c == 0 || iy_c == (nx_c - 1));
        int fine_bndry = (ix_f == 0 || ix_f == (nx_f - 1) || iy_f == 0 || iy_f == (nx_f - 1));
        int bndry = coarse_bndry || fine_bndry;
        scale *= !bndry; // makes it zero if bndry

        // printf("fine %d, coarse %d, fb %d, cb %d : scale %.2e\n", fine_ind, coarse_ind, fine_bndry, coarse_bndry, scale);
        // printf("fine_ind %d => coarse_ind %d, with scale %.2e on fine val %.2e\n", fine_ind, coarse_ind, scale, finer_vec_in[fine_ind]);

        atomicAdd(&coarse_vec_out[coarse_ind], scale * finer_vec_in[fine_ind]);
    }

}
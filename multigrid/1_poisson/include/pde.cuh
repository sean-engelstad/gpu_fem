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

/* helper kernel functions with prefix k_ for kernel function*/
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

    // now we write into the main
    atomicAdd(&d_csrNUNIQRowCounts[g_row], 1);
}

template <typename I>
__global__ static void k_uniqueRowCount(int N, int *d_minConnectedNode, int *d_csrRowCounts, I *d_csr_nnz) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind < N) {
        int validConnectedCol = d_minConnectedNode[ind] < N; // if it's above N, there were no min connected cols left => it adds 0
        d_csrRowCounts[ind] += validConnectedCol;
        atomicAdd(d_csr_nnz, validConnectedCol);
    }
}

// template <typename I>
// __global__ static void k_csrRowCounts(int nxe, int nelems, int *d_nodeMinElems,
//                                         int *d_csr_row_counts, I *d_csr_nnz) {
//     // compute the number of values on CSR rowp (row counts)
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

//     // get bools for whether these global nodes have this as min element or not
//     int i_min_elem = d_nodeMinElems[g_row];
//     int j_min_elem = d_nodeMinElems[g_col];
//     // if both have same min elem, both need it (they lie on same edge0)
//     int case1 = (i_min_elem == j_min_elem) && (ielem == i_min_elem && ielem == j_min_elem);
//     // if not same min element, they don't lie on same edge, only need ielem to match one of them
//     int case2 = (i_min_elem != j_min_elem) && (ielem == i_min_elem || ielem == j_min_elem);
//     int b_match_min_elem = case1 || case2;

//     // now we write into the main
//     atomicAdd(&d_csr_row_counts[g_row], b_match_min_elem);
//     atomicAdd(d_csr_nnz, b_match_min_elem);
// }

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

    // now do atomic min, this should get min connected node larger than previous (efficiently)
    // printf("g_row %d, g_col %d, valid %d\n", g_row, g_col, b_larger_than_prev); // race condition stuff here.. (fixed by two arrays now)
    atomicMin(&d_minConnectedNode_out[g_row], g_col * b_larger_than_prev + nrows * (1 - b_larger_than_prev));
}

__global__ static void k_addMinConnectedNode(int i_minNode, int N, int *d_minConnectedNode,
                                                int *d_csr_rowp, int *d_csr_cols) {
    // add the min connected node currently to the csr cols device array
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int start = d_csr_rowp[ind];
        int end = d_csr_rowp[ind + 1];
        int cols_ind = start + i_minNode;
        int conn_node = d_minConnectedNode[ind];
        int valid = cols_ind < end;

        // valid bool helps us add without branching
        d_csr_cols[cols_ind] += conn_node * valid;
    }
}
#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "../utils.h"
#include "chrono"
#include "cuda_utils.h"
#include "reordering_utils.h"
#include "stdlib.h"
#include "vec.h"

typedef int index_t; // for sparse utils

// dependency to smdogroup/sparse-utils repo
#include "sparse_utils/sparse_symbolic.h"
#include "sparse_utils/sparse_utils.h"

class BsrData {
    /* holds bsr sparsity pattern */
public:
    BsrData() = default;
    __HOST__ BsrData(const int nelems, const int nnodes, const int &nodes_per_elem,
        const int &block_dim, const int32_t *conn)
        : nelems(nelems), nnodes(nnodes), nodes_per_elem(nodes_per_elem), elem_conn(conn),
          block_dim(block_dim), elem_ind_map(nullptr), tr_rowp(nullptr), tr_cols(nullptr),
          tr_block_map(nullptr), host(true) {

        _get_row_col_ptrs_sparse();

        // make a nominal ordering (no permutation)
        perm = new int32_t[nnodes];
        iperm = new int32_t[nnodes];
        for (int inode = 0; inode < nnodes; inode++) {
            perm[inode] = inode;
            iperm[inode] = inode;
        }
        
        // also initialize the elem_ind_map and transpose block cols (need to compute again later though)
        _compute_symbolic_maps();
    }

    /* length of values array */
    __HOST_DEVICE__ int32_t getNumValues() const { return nnzb * block_dim * block_dim; }

    __HOST__ void _get_row_col_ptrs_sparse() {
        /* get the rowp, cols from the element connectivity using sparse utils */
        auto su_mat = SparseUtils::BSRMatFromConnectivityCUDA<double, 1>(nelems, nnodes, nodes_per_elem, elem_conn);
        nnzb = su_mat->nnz;
        rowp = su_mat->rowp;
        cols = su_mat->cols;
    }

    __HOST__ void compute_full_LU_pattern(double fill_factor = 10.0, const bool print = false) {
        /* computes the sparsity pattern for a full LU for direct LU solve
           with AMD reordering, consider making enum for reordering types here? */
        auto start = std::chrono::high_resolution_clock::now();
        if (print) {
            printf("begin symbolic factorization::\n");
            printf("\tnnzb = %d\n", nnzb);
        }

        // compute fillin
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowp, cols, nullptr);
        int nnzb_old = nnzb;
        auto su_mat2 = BSRMatAMDFactorSymbolicCUDA(su_mat, fill_factor);
        // TODO : add options for different reordering than AMD here

        // delete previous pointers here
        if (rowp) delete[] rowp;
        if (cols) delete[] cols;
        if (perm) delete[] perm;
        if (iperm) delete[] iperm;

        // get full LU pattern and AMD reordering from sparse utils object
        nnzb = su_mat2->nnz;
        rowp = su_mat2->rowp;
        cols = su_mat2->cols;
        perm = su_mat2->iperm; // my reordering definition is flipped from sparse utils
        iperm = su_mat2->perm;

        // now compute symbolic maps for fast GPU processing
        _compute_symbolic_maps();

        if (print) {
            printf(
                "\tsymbolic factorization with fill_factor %.2f from nnzb %d to "
                "%d NZ\n",
                fill_factor, nnzb_old, nnzb);
        }
    }


    __HOST__ void _compute_symbolic_maps() {
        /* computes symbolic maps such as elem_ind_map and transpose mappings
           for efficient kernel processing */

        if (elem_ind_map) delete[] elem_ind_map;
        if (tr_rowp) delete[] tr_rowp;
        if (tr_cols) delete[] tr_cols;
        if (tr_block_map) delete[] tr_block_map;
        
        // 1) compute the elem ind map -------------
        /* elem_ind_map is nelems x 4 x 4 array for nodes_per_elem = 4 
           it's useful in telling how to add each block matrix to global*/
        int nodes_per_elem2 = nodes_per_elem * nodes_per_elem;
        elem_ind_map = new index_t[nelems * nodes_per_elem2]();
        for (int ielem = 0; ielem < nelems; ielem++) {
            const int32_t *local_conn = &elem_conn[nodes_per_elem * ielem];
            for (int block_row = 0; block_row < nodes_per_elem; block_row++) {
                int32_t _global_block_row = local_conn[block_row];
                int32_t global_block_row = perm[_global_block_row];
                int col_istart = rowp[global_block_row];
                int col_iend = rowp[global_block_row + 1];
                for (int block_col = 0; block_col < nodes_per_elem; block_col++) {
                    int32_t _global_block_col = local_conn[block_col];
                    int32_t global_block_col = perm[_global_block_col];
                    // get matching ind in cols for the global_block_col
                    for (int i = col_istart; i < col_iend; i++) {
                        if (cols[i] == global_block_col) {
                            // add this component of block kelem matrix into
                            // elem_ind_map
                            int block_ind = nodes_per_elem * block_row + block_col;
                            elem_ind_map[nodes_per_elem2 * ielem + block_ind] = i;
                            break;
                        }
                    }
                }
            }
        }

        // 2) compute the transpose matrix maps --------------        
        /* difficult to apply bcs to cols BSR with column-major format, but easy to apply to rows
           so compute here the the sparsity maps of the transpose matrix */

        // get transpose rowp, cols of BSR matrix
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowp, cols, nullptr);
        auto su_mat_transpose = SparseUtils::BSRMatMakeTransposeSymbolic(su_mat);
        tr_rowp = su_mat_transpose->rowp;
        tr_cols = su_mat_transpose->cols;

        // also compute a transpose block map map
        // from transpose_block_ind => orig_block_ind (of the values arrays)
        tr_block_map = new int32_t[nnzb];
        int32_t *temp_col_local_block_ind_ctr = new int32_t[nnodes];
        memset(temp_col_local_block_ind_ctr, 0, nnodes * sizeof(int32_t));
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int orig_block_ind = rowp[block_row]; orig_block_ind < rowp[block_row + 1];
                 orig_block_ind++) {
                int32_t block_col = cols[orig_block_ind];
                int32_t orig_val_ind = orig_block_ind;
                int32_t transpose_val_ind =
                    tr_rowp[block_col] + temp_col_local_block_ind_ctr[block_col];
                temp_col_local_block_ind_ctr[block_col]++; // increment number of block cols in this col
                tr_block_map[transpose_val_ind] = orig_val_ind;
            }
        }

        delete[] temp_col_local_block_ind_ctr;
    }

    __HOST__ BsrData createHostBsrData() {
        /* create new BsrData object with all Device sparsity data copied to the Host */
        BsrData new_bsr; // bypass main constructor so we don't end up making host data
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;
        new_bsr.host = true;

        // create HostVec wrapper objects in CUDA, transfer to device and get ptr for new object
        int n_eim = nelems * nodes_per_elem * nodes_per_elem;
        DeviceVec<int> d_rowp(nnodes+1, rowp), d_cols(nnzb, cols), d_perm(nnodes, perm), d_iperm(nnodes, iperm),
                    d_elem_ind_map(n_eim, elem_ind_map), d_tr_rowp(nnodes+1, tr_rowp), d_tr_cols(nnzb, tr_cols), 
                    d_tr_block_map(nnzb, tr_block_map);
        new_bsr.rowp = d_rowp.createHostVec().getPtr();
        new_bsr.cols = d_cols.createHostVec().getPtr();
        new_bsr.perm = d_perm.createHostVec().getPtr();
        new_bsr.iperm = d_iperm.createHostVec().getPtr();
        new_bsr.elem_ind_map = d_elem_ind_map.createHostVec().getPtr();
        new_bsr.tr_rowp = d_tr_rowp.createHostVec().getPtr();
        new_bsr.tr_cols = d_tr_cols.createHostVec().getPtr();
        new_bsr.tr_block_map = d_tr_block_map.createHostVec().getPtr();

        return new_bsr;
    }

    __HOST__ BsrData createDeviceBsrData() {
        /* create new BsrData object with all host sparsity data copied to the device */
        BsrData new_bsr; // bypass main constructor so we don't end up making host data
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;
        new_bsr.host = false;

        // create HostVec wrapper objects in CUDA, transfer to device and get ptr for new object
        int n_eim = nelems * nodes_per_elem * nodes_per_elem;
        HostVec<int> h_rowp(nnodes+1, rowp), h_cols(nnzb, cols), h_perm(nnodes, perm), h_iperm(nnodes, iperm),
                    h_elem_ind_map(n_eim, elem_ind_map), h_tr_rowp(nnodes+1, tr_rowp), h_tr_cols(nnzb, tr_cols), 
                    h_tr_block_map(nnzb, tr_block_map);
        new_bsr.rowp = h_rowp.createDeviceVec().getPtr();
        new_bsr.cols = h_cols.createDeviceVec().getPtr();
        new_bsr.perm = h_perm.createDeviceVec().getPtr();
        new_bsr.iperm = h_iperm.createDeviceVec().getPtr();
        new_bsr.elem_ind_map = h_elem_ind_map.createDeviceVec().getPtr();
        new_bsr.tr_rowp = h_tr_rowp.createDeviceVec().getPtr();
        new_bsr.tr_cols = h_tr_cols.createDeviceVec().getPtr();
        new_bsr.tr_block_map = h_tr_block_map.createDeviceVec().getPtr();

        return new_bsr;
    }

    __HOST__ void free() {
        /* cleanup / delete data after use*/
        if (!this->host) {
            // delete data on device
            if (rowp) cudaFree(rowp);
            if (cols) cudaFree(cols);
            if (perm) cudaFree(perm);
            if (iperm) cudaFree(iperm);
            if (elem_ind_map) cudaFree(elem_ind_map);
            if (tr_rowp) cudaFree(tr_rowp);
            if (tr_cols) cudaFree(tr_cols);
            if (tr_block_map) cudaFree(tr_block_map);
        } else {
            // delete data on host
            if (rowp) delete[] rowp;
            if (cols) delete[] cols;
            if (perm) delete[] perm;
            if (iperm) delete[] iperm;
            if (elem_ind_map) delete[] elem_ind_map;
            if (tr_rowp) delete[] tr_rowp;
            if (tr_cols) delete[] tr_cols;
            if (tr_block_map) delete[] tr_block_map;
        }
    }


    // object data, kept public =---------------------------
    int32_t nnzb, nelems, nnodes, nodes_per_elem, block_dim;
    const int32_t *elem_conn;
    int32_t *rowp, *cols, *elem_ind_map;
    int32_t *perm, *iperm;
    int32_t *tr_rowp, *tr_cols, *tr_block_map;
    bool host;
};
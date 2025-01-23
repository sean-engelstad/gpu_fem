#pragma once
#include "../base/utils.h"
#include "chrono"
#include "cuda_utils.h"
#include "stdlib.h"
#include "vec.h"
#include <algorithm>
#include <vector>

// typedef std::size_t index_t;

// in order to work with cusparse index_t has to be int
typedef int index_t;

// dependency to smdogroup/sparse-utils repo
#include "sparse_utils/sparse_symbolic.h"
#include "sparse_utils/sparse_utils.h"

// pre declaration for readability and use in the BsrData class below
__HOST__ void get_row_col_ptrs(const int &nelems, const int &nnodes,
                               const int32_t *conn, const int &nodes_per_elem,
                               int &nnzb, index_t *&rowPtr, index_t *&colPtr);

__HOST__ void sparse_utils_fillin(const int &nnodes, int &nnzb,
                                  index_t *&rowPtr, index_t *&colPtr,
                                  double fill_factor, const bool print = false);

__HOST__ void get_elem_ind_map(const int &nelems, const int &nnodes,
                               const int32_t *conn, const int &nodes_per_elem,
                               const int &nnzb, index_t *&rowPtr,
                               index_t *&colPtr, index_t *&elemIndMap);

class BsrData {
  public:
    BsrData() = default;
    __HOST__ BsrData(const int nelems, const int nnodes,
                     const int &nodes_per_elem, const int &block_dim,
                     const int32_t *conn)
        : nelems(nelems), nnodes(nnodes), nodes_per_elem(nodes_per_elem),
          conn(conn), block_dim(block_dim) {
        get_row_col_ptrs(nelems, nnodes, conn, nodes_per_elem, nnzb, rowPtr,
                         colPtr);
        elemIndMap = nullptr;
    }

    __HOST__ BsrData(const int nnodes, const int block_dim, const int nnzb,
                     index_t *origRowPtr, index_t *origColPtr,
                     double fill_factor = 100.0, const bool print = false)
        : nnodes(nnodes), block_dim(block_dim), nnzb(nnzb), rowPtr(origRowPtr),
          colPtr(origColPtr) {
        sparse_utils_fillin(nnodes, this->nnzb, rowPtr, colPtr, fill_factor,
                            print);
        elemIndMap = nullptr;
    }

    __HOST__ void symbolic_factorization(double fill_factor = 10.0,
                                         const bool print = false) {
        // do symbolic factorization for fillin
        auto start = std::chrono::high_resolution_clock::now();
        if (print) {
            printf("begin symbolic factorization::\n");
        }
        sparse_utils_fillin(nnodes, nnzb, rowPtr, colPtr, fill_factor, print);
        // if (print) {
        //     printf("\t1/2 done with sparse utils fillin\n");
        // }

        get_elem_ind_map(nelems, nnodes, this->conn, nodes_per_elem, nnzb,
                         rowPtr, colPtr, this->elemIndMap);
        // if (print) {
        //     printf("\t2/2 done with elem ind map\n");
        // }

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        if (print) {
            printf("\tfinished symbolic factorization in %d microseconds\n",
                   (int)duration.count());
        }
    }

    __HOST__ BsrData createDeviceBsrData() { // deep copy each array onto the
        BsrData new_bsr;
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;

        // make host vec copies of these pointers
        HostVec<index_t> h_rowPtr(this->nnodes + 1, this->rowPtr);
        HostVec<index_t> h_colPtr(nnzb, this->colPtr);
        int nodes_per_elem2 = this->nodes_per_elem * this->nodes_per_elem;
        int n_elemIndMap = this->nelems * nodes_per_elem2;
        HostVec<index_t> h_elemIndMap(n_elemIndMap, this->elemIndMap);

        // now use the Vec utils to create device pointers of them
        auto d_rowPtr = h_rowPtr.createDeviceVec();
        auto d_colPtr = h_colPtr.createDeviceVec();
        auto d_elemIndMap = h_elemIndMap.createDeviceVec();

        // now copy to the new BSR structure
        new_bsr.rowPtr = d_rowPtr.getPtr();
        new_bsr.colPtr = d_colPtr.getPtr();
        new_bsr.elemIndMap = d_elemIndMap.getPtr();

        // // debug
        // printf("h_rowPtr: ");
        // printVec<int>(new_bsr.nnodes + 1, h_rowPtr.getPtr());
        // printf("\n");
        // DeviceVec<int> d_rowPtr2(new_bsr.nnodes + 1, new_bsr.rowPtr);
        // auto h_rowPtr2 = d_rowPtr2.createHostVec();
        // printf("h_rowPtr2: ");
        // printVec<int>(new_bsr.nnodes, h_rowPtr2.getPtr());
        // printf("\n");

        return new_bsr;
    }

    __HOST_DEVICE__ int32_t getNumValues() const {
        // length of values array
        return nnzb * block_dim * block_dim;
    }

    // private: (keep public for now?)
    int32_t nnzb; // num nonzero blocks (in full matrix)
    int32_t nelems, nnodes;
    int32_t nodes_per_elem; // kelem : nodes_per_elem^2 dense matrix
    int32_t block_dim;      // equiv to vars_per_node (each block is 6x6)
    const int32_t *conn;    // element connectivity
    index_t *rowPtr, *colPtr, *elemIndMap;
};

// main utils
// ---------------------------------------------------

__HOST_DEVICE__ bool node_in_elem_conn(const int &nodes_per_elem,
                                       const int inode, const int *elem_conn) {
    for (int i = 0; i < nodes_per_elem; i++) {
        if (elem_conn[i] == inode) {
            return true;
        }
    }
    return false;
}

__HOST__ void get_row_col_ptrs(
    const int &nelems, const int &nnodes, const int *conn,
    const int &nodes_per_elem,
    int &nnzb,        // num nonzero blocks
    index_t *&rowPtr, // array of len nnodes+1 for how many cols in each row
    index_t *&colPtr  // array of len nnzb of the column indices for each block
) {

    // could launch a kernel to do this somewhat in parallel?

    nnzb = 0;
    std::vector<index_t> _rowPtr(nnodes + 1, 0);
    std::vector<index_t> _colPtr;

    // loop over each block row checking nz values
    for (int inode = 0; inode < nnodes; inode++) {
        std::vector<index_t> temp;
        for (int ielem = 0; ielem < nelems; ielem++) {
            const int *elem_conn = &conn[ielem * nodes_per_elem];
            if (node_in_elem_conn(nodes_per_elem, inode, elem_conn)) {
                for (int in = 0; in < nodes_per_elem; in++) {
                    temp.push_back(elem_conn[in]);
                }
            }
        }

        // first sort the vector
        std::sort(temp.begin(), temp.end());

        // remove duplicates
        auto last = std::unique(temp.begin(), temp.end());
        temp.erase(last, temp.end());

        // add this to _colPtr
        _colPtr.insert(_colPtr.end(), temp.begin(), temp.end());

        // add num non zeros to nnzb for this row, also update
        nnzb += temp.size();
        _rowPtr[inode + 1] = nnzb;
    }

    // copy data to output pointers (deep copy)
    rowPtr = new index_t[nnodes + 1];
    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new index_t[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);
}

__HOST__ void sparse_utils_fillin(const int &nnodes, int &nnzb,
                                  index_t *&rowPtr, index_t *&colPtr,
                                  double fill_factor, const bool print) {
    std::vector<index_t> _rowPtr(nnodes + 1, 0);
    std::vector<index_t> _colPtr(index_t(fill_factor * nnzb));
    int nnzb_old = nnzb;

    nnzb = SparseUtils::CSRFactorSymbolic(nnodes, rowPtr, colPtr, _rowPtr,
                                          _colPtr);
    if (print) {
        printf("\tsymbolic factorization with fill_factor %.2f from nnzb %d to "
               "%d NZ\n",
               fill_factor, nnzb_old, nnzb);
    }

    // resize this after running the symbolic factorization
    // TODO : can use less than this full nnzb in the case of preconditioning or
    // incomplete ILU?
    _colPtr.resize(nnzb);

    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new index_t[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);
}

__HOST__ void get_elem_ind_map(
    const int &nelems, const int &nnodes, const int32_t *conn,
    const int &nodes_per_elem,
    const int &nnzb,  // num nonzero blocks
    index_t *&rowPtr, // array of len nnodes+1 for how many cols in each row
    index_t *&colPtr, // array of len nnzb of the column indices for each block
    index_t *&elemIndMap) {

    int nodes_per_elem2 = nodes_per_elem * nodes_per_elem;

    // determine where each global_node node of this elem
    // should be added into the ind of colPtr as map

    elemIndMap = new index_t[nelems * nodes_per_elem2]();
    // elemIndMap is nelems x 4 x 4 array for nodes_per_elem = 4
    // shows how to add each block matrix into global
    for (int ielem = 0; ielem < nelems; ielem++) {

        const int32_t *local_conn = &conn[nodes_per_elem * ielem];

        // TODO : might be error here due to sorting the local conn..
        // std::vector<int32_t> sorted_conn;
        // for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
        //     sorted_conn.push_back(local_conn[lnode]);
        // }
        // std::sort(sorted_conn.begin(), sorted_conn.end());

        // loop over each block row
        for (int n = 0; n < nodes_per_elem; n++) {
            // int global_node_row = sorted_conn[n];
            int32_t global_node_row = local_conn[n];
            // get the block col range of colPtr for this block row
            int col_istart = rowPtr[global_node_row];
            int col_iend = rowPtr[global_node_row + 1];

            // loop over each block col
            for (int m = 0; m < nodes_per_elem; m++) {
                // int global_node_col = sorted_conn[m];
                int32_t global_node_col = local_conn[m];

                // find the matching indices in colPtr for the global_node_col
                // of this elem_conn
                for (int i = col_istart; i < col_iend; i++) {
                    if (colPtr[i] == global_node_col) {
                        // add this component of block kelem matrix into
                        // elem_ind_map
                        int nm = nodes_per_elem * n + m;
                        elemIndMap[nodes_per_elem2 * ielem + nm] = i;
                    }
                }
            }
        }
    }
}

// nice reference utility (may not use this)
// template <typename T, int block_dim, int nodes_per_elem, int dof_per_elem>
// void assemble_sparse_bsr_matrix(
//     const int &nelems, const int &nnodes, const int &vars_per_node,
//     const int *conn,
//     int &nnzb,        // num nonzero blocks
//     int *&rowPtr,     // array of len nnodes+1 for how many cols in each row
//     int *&colPtr,     // array of len nnzb of the column indices for each
//     block int *&elemIndMap, // map of rowPtr, colPtr assembly locations for
//     kelem int &nvalues,     // length of values array = block_dim^2 * nnzb T
//     *&values) {
//     // get nnzb, rowPtr, colPtr
//     get_row_col_ptrs<nodes_per_elem>(nelems, nnodes, conn, nnzb, rowPtr,
//                                      colPtr);

//     // get elemIndMap to know how to add into the values array
//     get_elem_ind_map<T, nodes_per_elem>(nelems, nnodes, conn, nnzb, rowPtr,
//                                         colPtr, elemIndMap);

//     // create values array (T*& because we can't initialize it outside this
//     // method so
//     //     pointer itself changes, normally T* is enough but not here)
//     nvalues = block_dim * block_dim * nnzb;
//     int nnz_per_block = block_dim * block_dim;
//     int blocks_per_elem = nodes_per_elem * nodes_per_elem;
//     values = new T[nnz_per_block * nnzb];

//     // now add each kelem into values array as part of assembly process
//     for (int ielem = 0; ielem < nelems; ielem++) {
//         T kelem[dof_per_elem * dof_per_elem];
//         get_fake_kelem<T, dof_per_elem>(0.0, kelem);

//         // now use elemIndxMap to add into values
//         for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++)
//         {
//             int istart = nnz_per_block *
//                          elemIndMap[blocks_per_elem * ielem + elem_block];
//             T *val = &values[istart];
//             int elem_block_row = elem_block / nodes_per_elem;
//             int elem_block_col = elem_block % nodes_per_elem;
//             for (int inz = 0; inz < nnz_per_block; inz++) {
//                 int inner_row = inz / block_dim;
//                 int row = vars_per_node * elem_block_row + inner_row;
//                 int inner_col = inz % block_dim;
//                 int col = vars_per_node * elem_block_col + inner_col;

//                 val[inz] += kelem[dof_per_elem * row + col];
//             }
//         }
//     }
// }
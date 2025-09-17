#pragma once

#include "unstruct_utils.h"
#include "unstructured.cuh"

template <class Basis>
class UnstructuredProlongation {
   public:
    using T = double;
    static constexpr bool structured = false;
    static constexpr bool assembly = false;

    static void prolongate(int nnodes_fine, int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems,
                           T *d_n2e_xis, int *d_coarse_iperm, int *d_fine_iperm,
                           DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
        // zero anything out again
        int N_coarse = coarse_soln_in.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = dx_fine.getSize();

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_prolongate<T, Basis>
            <<<grid, block>>>(coarse_soln_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr,
                              d_n2e_elems, d_n2e_xis, nnodes_fine, d_fine_iperm, dx_fine.getPtr());
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    static void restrict_defect(int nnodes_fine, const int *d_coarse_conn, int *d_n2e_ptr,
                                int *d_n2e_elems, T *d_n2e_xis, int *d_coarse_iperm,
                                int *d_fine_iperm, DeviceVec<T> fine_defect_in,
                                DeviceVec<T> coarse_defect_out, T *d_weights) {
        // zero anything out again
        int N_coarse =
            coarse_defect_out.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = fine_defect_in.getSize();
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_restrict<T, Basis><<<grid, block>>>(
            fine_defect_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems,
            d_n2e_xis, nnodes_fine, d_fine_iperm, coarse_defect_out.getPtr(), d_weights);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // TRY not normalizing restriction (that kind of breaks P and P^T relationship..) to
        // comment, seems to be fine either way.. this out.. normalize
        // int nblock2 = (N_coarse + 31) / 32;
        // dim3 grid2(nblock2);
        // k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_defect_out.getPtr(), d_weights);
    }
};

template <class Basis>
class UnstructuredProlongationFast {
   public:
    using T = double;
    static constexpr bool structured = false;
    static constexpr bool assembly = true;

    static void assemble_matrices(int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems, T *d_n2e_xis, 
        BsrMat<DeviceVec<T>> &P_mat, BsrMat<DeviceVec<T>> &PT_mat) {

        // get some data out..
        auto P_bsr_data = P_mat.getBsrData();
        auto PT_bsr_data = PT_mat.getBsrData();
        int nnodes_fine = P_bsr_data.nnodes;
        int nnodes_coarse = PT_bsr_data.nnodes;
        int *d_coarse_iperm = PT_bsr_data.iperm;
        int *d_fine_iperm = P_bsr_data.iperm;
        int block_dim = P_bsr_data.block_dim;

        // assemble P mat
        dim3 block(32);
        dim3 grid((nnodes_fine + 31) / 32);
        int *d_P_rowp = P_bsr_data.rowp, *d_P_cols = P_bsr_data.cols;
        T *d_P_vals = P_mat.getPtr();
        k_prolong_mat_assembly<T,Basis><<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, 
            d_n2e_elems, d_n2e_xis, nnodes_fine, d_fine_iperm, d_P_rowp, d_P_cols, block_dim, d_P_vals);

        // assemble PT mat
        int *d_PT_rowp = PT_bsr_data.rowp, *d_PT_cols = PT_bsr_data.cols;
        T *d_PT_vals = PT_mat.getPtr();
        k_restrict_mat_assembly<T,Basis><<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, 
            d_n2e_elems, d_n2e_xis, nnodes_fine, d_fine_iperm, d_PT_rowp, d_PT_cols, block_dim, d_PT_vals);
    }

    static void prolongate(cusparseHandle_t handle, cusparseMatDescr_t &descr_P, BsrMat<DeviceVec<T>> prolong_mat,
                           DeviceVec<T> perm_coarse_soln_in, DeviceVec<T> perm_dx_fine) {
        // get important data & vecs out
        // printf("unstruct prolong fast start\n");
        auto P_bsr_data = prolong_mat.getBsrData();
        const int block_dim = P_bsr_data.block_dim;
        int *d_rowp = P_bsr_data.rowp, *d_cols = P_bsr_data.cols, nnzb = P_bsr_data.nnzb;
        T *d_vals = prolong_mat.getPtr();
        int mb = P_bsr_data.mb, nb = P_bsr_data.nb;


        // now do cusparse Bsrmv.. for P @ coarse_soln => dx_fine (permuted nodes order)
        // if constexpr (block_dim != 1) {
        //     T a = 1.0, b = 0.0;
        //     CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
        //                                         CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &a,
        //                                         descr_P, d_vals, d_rowp, d_cols, block_dim,
        //                                         perm_coarse_soln_in.getPtr(), &b, perm_dx_fine.getPtr()));
        // } else {
            // block_dim == 1, my own kernel here..
        dim3 block(32);
        dim3 grid((nnzb + 31) / 32);

        int *d_rows = P_bsr_data.rows;
        k_csr_mat_vec<T><<<grid, block>>>(nnzb, 6, d_rows, d_cols, d_vals, perm_coarse_soln_in.getPtr(), perm_dx_fine.getPtr());
        // }

        // T *h_perm_dx_fine = perm_dx_fine.createHostVec().getPtr();
        // printf("hperm dx fine:");
        // printVec<T>(100, h_perm_dx_fine);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("unstruct prolong fast end\n");
    }

    static void restrict_defect(cusparseHandle_t handle, cusparseMatDescr_t descr_PT, BsrMat<DeviceVec<T>> restrict_mat,
                           DeviceVec<T> fine_defect_in, DeviceVec<T> coarse_defect_out) {
        // get important data & vecs out
        // printf("unstruct restrict fast start\n");
        auto PT_bsr_data = restrict_mat.getBsrData();
        const int block_dim = PT_bsr_data.block_dim;
        int *d_rowp = PT_bsr_data.rowp, *d_cols = PT_bsr_data.cols, nnzb = PT_bsr_data.nnzb;
        T *d_vals = restrict_mat.getPtr();
        int mb = PT_bsr_data.mb, nb = PT_bsr_data.nb;


        // if constexpr (block_dim != 1) {
        //     T a = 1.0, b = 0.0;
        //     // printf("mb %d, nb %d, nnzb %d, a %.2e, b %.2e, block_dim %d\n", mb, nb, nnzb, a, b, block_dim);

        //     CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
        //                                         CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &a,
        //                                         descr_PT, d_vals, d_rowp, d_cols, block_dim,
        //                                         fine_defect_in.getPtr(), &b, coarse_defect_out.getPtr()));
        // } else {
        // block_dim == 1, my own kernel here..
        dim3 block(32);
        dim3 grid((nnzb + 31) / 32);

        int *d_rows = PT_bsr_data.rows;
        // int *h_rows = DeviceVec<int>(nnzb, d_rows).createHostVec().getPtr();
        // printf("h_rows: ");
        // printVec<int>(nnzb, h_rows);

        k_csr_mat_vec<T><<<grid, block>>>(nnzb, 6, d_rows, d_cols, d_vals, fine_defect_in.getPtr(), coarse_defect_out.getPtr());
        // }

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("unstruct restrict defect fast end\n");
    }
};
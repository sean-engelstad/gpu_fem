#pragma once

#include "_unstruct_utils.h"
#include "_unstructured.cuh"

template <class Assembler_, class Basis_, bool is_bsr_ = false>
class UnstructuredProlongation {
   public:
    using T = double;
    using Assembler = Assembler_;
    using Basis = Basis_;

    static constexpr bool structured = false;
    static constexpr bool assembly = true;
    static constexpr int is_bsr = is_bsr_;

    UnstructuredProlongation(Assembler &fine_assembler_, int ELEM_MAX_ = 10) {
        fine_assembler = fine_assembler_;
        ELEM_MAX = ELEM_MAX_;

        // init some data from fine assembler, and other startup
        CHECK_CUSPARSE(cusparseCreate(&handle));
        block_dim = fine_assembler.getBsrData().block_dim;

        // other startup
        descr_P = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_P));
        CHECK_CUSPARSE(cusparseSetMatType(descr_P, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_P, CUSPARSE_INDEX_BASE_ZERO));
        descr_PT = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_PT));
        CHECK_CUSPARSE(cusparseSetMatType(descr_PT, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_PT, CUSPARSE_INDEX_BASE_ZERO));
    }

    // nothing
    void update_after_assembly() {}

    void init_coarse_data(Assembler &coarse_assembler_) {
        coarse_assembler = coarse_assembler_;

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_01 = std::chrono::high_resolution_clock::now();
        construct_nz_pattern();
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_01 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> nz_pattern_time = end_01 - start_01;
        printf("\t\tprolong - construct nz pattern %.2e\n", nz_pattern_time.count());

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_02 = std::chrono::high_resolution_clock::now();
        assemble_matrices();
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_02 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assemble_prolong_time = end_02 - start_02;
        printf("\t\tprolong - assemble P and PT %.2e\n", assemble_prolong_time.count());

        // cleaner code
        // construct_nz_pattern();
        // assemble_matrices();
    }

    void construct_nz_pattern() {
        // call the utils method to get nz pattern and do any other steps
        prolong_mat = nullptr, restrict_mat = nullptr;
        d_coarse_conn = nullptr, d_n2e_ptr = nullptr, d_n2e_elems = nullptr, d_n2e_xis = nullptr;
        init_unstructured_grid_maps<T, Assembler, Basis, is_bsr>(fine_assembler, coarse_assembler,
            prolong_mat, restrict_mat, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, ELEM_MAX);

        // then get some data out of this (for more readable code later)
        P_bsr_data = prolong_mat->getBsrData();
        d_P_vals = prolong_mat->getPtr();
        d_P_rowp = P_bsr_data.rowp, d_P_rows = P_bsr_data.rows, d_P_cols = P_bsr_data.cols;
        P_nnzb = P_bsr_data.nnzb;
        nnodes_fine = P_bsr_data.nnodes;
        d_fine_iperm = P_bsr_data.iperm;

        PT_bsr_data = restrict_mat->getBsrData();
        d_PT_vals = restrict_mat->getPtr();
        d_PT_rowp = PT_bsr_data.rowp,  d_PT_rows = PT_bsr_data.rows, d_PT_cols = PT_bsr_data.cols;
        PT_nnzb = PT_bsr_data.nnzb;
        d_coarse_iperm = PT_bsr_data.iperm;
    }   

    void assemble_matrices() {
        /* assemble the P and PT bsr (or csr) matrices */
        int red_block_dim = is_bsr ? block_dim : 1; // 1 if csr prolong (same for all DOF per node)

        // assemble P mat
        dim3 block(32);
        dim3 grid((nnodes_fine + 31) / 32);
        k_prolong_mat_assembly<T, Basis, is_bsr>
            <<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                              nnodes_fine, d_fine_iperm, d_P_rowp, d_P_cols, red_block_dim, d_P_vals);

        // assemble PT mat
        k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block>>>(
            d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, nnodes_fine,
            d_fine_iperm, d_PT_rowp, d_PT_cols, red_block_dim, d_PT_vals);
    }

    void prolongate(DeviceVec<T> perm_coarse_soln_in, DeviceVec<T> perm_dx_fine) {
        // get important data & vecs out

        // now do cusparse Bsrmv.. for P @ coarse_soln => dx_fine (permuted nodes order)
        if constexpr (is_bsr) {
            T a = 1.0, b = 0.0;
            int mb = P_bsr_data.mb, nb = P_bsr_data.nb;
            CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, P_nnzb,
                                                &a, descr_P, d_P_vals, d_P_rowp, d_P_cols, block_dim,
                                                perm_coarse_soln_in.getPtr(), &b,
                                                perm_dx_fine.getPtr()));
        } else {
            // else CSR case (same transfer stencil for each dof per node)
            dim3 block(32);
            dim3 grid((P_nnzb + 31) / 32);
            k_csr_mat_vec<T><<<grid, block>>>(P_nnzb, block_dim, d_P_rows, d_P_cols, d_P_vals,
                                            perm_coarse_soln_in.getPtr(), perm_dx_fine.getPtr());
        }
    }

    void restrict_defect(DeviceVec<T> fine_defect_in, DeviceVec<T> coarse_defect_out) {

        if constexpr (is_bsr) {
            T a = 1.0, b = 0.0;
            int mb = PT_bsr_data.mb, nb = PT_bsr_data.nb;
            CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, PT_nnzb,
                                                &a, descr_PT, d_PT_vals, d_PT_rowp, d_PT_cols, block_dim,
                                                fine_defect_in.getPtr(), &b,
                                                coarse_defect_out.getPtr()));
        } else {
            dim3 block(32);
            dim3 grid((PT_nnzb + 31) / 32);
            k_csr_mat_vec<T><<<grid, block>>>(PT_nnzb, block_dim, d_PT_rows, d_PT_cols, d_PT_vals, fine_defect_in.getPtr(),
                                            coarse_defect_out.getPtr());
        }
    }

    // public
    BsrMat<DeviceVec<T>> *prolong_mat, *restrict_mat;

  private:
    Assembler fine_assembler, coarse_assembler;
    int ELEM_MAX; // the max number of nearest neighbor elements for NZ pattern construction

    cusparseHandle_t handle;
    BsrData P_bsr_data, PT_bsr_data;
    T *d_P_vals, *d_PT_vals;
    cusparseMatDescr_t descr_P, descr_PT;
    int *d_P_rowp, *d_P_rows, *d_P_cols;
    int *d_PT_rowp, *d_PT_rows, *d_PT_cols;
    int P_nnzb, PT_nnzb;

    int *d_coarse_conn, *d_n2e_ptr, *d_n2e_elems;
    T *d_n2e_xis;
    int *d_coarse_iperm, *d_fine_iperm;
    int nnodes_fine, block_dim;
};
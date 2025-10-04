
class DampedJacobiSmoother {

    void dampedJacobi(int n_iters, bool print = false, int print_freq = 10, T omega = 0.8,
                      bool rev_colors = false) {
        // bool time_debug = false;
        // bool time_debug = true;

        for (int iter = 0; iter < n_iters; iter++) {
            // compute Dinv * defect => soln update (aka temp)
            // -----------------------------------------------
            T a = 1.0, b = 0.0;
            // note in this case d_diag_LU_vals_color refers to Dinv form of LU factors on each
            // nodal block
            CHECK_CUSPARSE(cusparseDbsrmv(  // NOTE just uses descrKmat cause would be the same as
                                            // descrDinv (convenience)
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes,
                nnodes, nnodes, &a, descrKmat, d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                d_defect.getPtr(), &b, d_temp));

            // -----------------------------------------------------------------
            // soln and defect update
            a = omega;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            a = -omega, b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

            // --------------------------------------------------------------------------------
        }  // next block-GS iteration
    }

};

class Lexigraphic {

    void lexigraphicBlockGS(int n_iters, bool print = false, int print_freq = 10) {
        // this is lexigraphic or RCM GS (RCM if more general mesh..)

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        T a, b;

        for (int iter = 0; iter < n_iters; iter++) {
            // 1) (L+D)*dx = defect with triang solve
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, &alpha, descr_kmat_L, d_kmat_vals,
                d_kmat_rowp, d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(), d_temp,
                policy_L, kmat_pBuffer));  // prob only need U^-1 part for block diag.. TBD

            // 2) update d_soln += d_temp (aka dx)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            // 3) compute new defect = prev_defect - A * dx
            a = -1.0,
            b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tLX-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

};
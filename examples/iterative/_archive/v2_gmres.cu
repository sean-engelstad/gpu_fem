// cusparseMatDescr_t descr;
    // CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    // CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    // cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // // ct++;
    // // printf("checkpt%d\n", ct);

    // // Setup ILU preconditioner
    // bsrilu02Info_t iluInfo;
    // CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&iluInfo));

    // cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // void *buffer;
    // int bufferSize;
    // CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
    //     cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr, precond_vals,
    //     dp_rowp, dp_cols, block_dim, iluInfo, &bufferSize));
    // CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    // // ct++;
    // // printf("checkpt%d\n", ct);

    // // Analyze ILU(0) structure for efficient numeric factorization
    // CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
    //     cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr, precond_vals,
    //     dp_rowp, dp_cols, block_dim, iluInfo, policy, buffer));

    // // ct++;
    // // printf("checkpt%d\n", ct); // doesn't reach this checkpoint

    // // numeric ILU(0) factorization (in-place on precond_vals)
    // CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, CUSPARSE_DIRECTION_ROW, mb,
    //     nnzb, descr, precond_vals, dp_rowp, dp_cols,
    //     block_dim, iluInfo, policy, buffer));

    // // Check for zero pivots (numerical singularities)
    // int numerical_zero;
    // cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(cusparseHandle, iluInfo, &numerical_zero);
    // if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
    //     printf("Block U(%d, %d) is not invertible (zero pivot detected)\n", numerical_zero, numerical_zero);
    // }

    // ct++;
    // printf("checkpt%d\n", ct);

    // Define matrix descriptors
    // cusparseIndexBase_t baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    // cusparseSpMatDescr_t matL, matU;

    // CHECK_CUSPARSE(cusparseCreateBsr(&matA, mb, mb, kmat_nnzb, block_dim, block_dim, 
    //     dk_rowp, dk_cols, kmat_vals, 
    //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
    //     baseIdx, CUDA_R_64F, CUSPARSE_ORDER_ROW));

    // switch order of this
    // // perform ILU numerical factorization in M
    // cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, precond_vals, dp_rowp,
    //     dp_cols, block_dim, info_M, policy_M, pBuffer);
    // status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
    // if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    //     printf("block U(%d,%d) is not invertible\n", numerical_zero,
    //             numerical_zero);
    // }

    // perform ILU numerical factorization in M
    // cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, precond_vals, dp_rowp,
    //     dp_cols, block_dim, info_M, policy_M, pBuffer);
    // status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
    // if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    //     printf("block U(%d,%d) is not invertible\n", numerical_zero,
    //             numerical_zero);
    // }

    // Create vector descriptors
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_X, mb * block_dim, soln.getPtr(), CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_B, mb * block_dim, loads.getPtr(), CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_tmp1, mb * block_dim, tmp1.getPtr(), CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_tmp2, mb * block_dim, tmp2.getPtr(), CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_tmp3, mb * block_dim, tmp3.getPtr(), CUDA_R_64F));
    // CHECK_CUSPARSE(cusparseCreateDnVec(&d_R, mb * block_dim, resid.getPtr(), CUDA_R_64F));

    // auto h_resid = resid.createHostVec();
    // printf("h_resid: ");
    // printVec<double>(100, h_resid.getPtr());

    // auto h_loads = loads.createHostVec();
    // printf("h_loads: ");
    // printVec<double>(100, h_loads.getPtr());

    // return 0;

    // in first precond solve want to get buffer sizes (don't need in subsequent precond solves)
    // size_t bufferSizeL, bufferSizeU;
    // void *d_bufferL, *d_bufferU;
    // cusparseSpSVDescr_t spsvDescrL, spsvDescrU;

    // CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
        //     matA, d_tmp1, &zero, d_tmp2, CUDA_R_64F,
        //     CUSPARSE_SPMV_ALG_DEFAULT, d_bufferA))
        // doesn't work because SpMV only works for CSR not BSR

    // if (_b == 0.0) {
        //     cs[j] = 1.0;
        //     ss[j] = 0.0;
        // } else if (abs(_b) > abs(_a)) {
        //     double tau = -_a / _b;
        //     ss[j] = 1.0 / sqrt(1.0 + tau * tau);
        //     cs[j] = ss[j] * tau;
        // } else {
        //     double tau = -_b / _a;
        //     cs[j] = 1.0 / sqrt(1.0 + tau * tau);
        //     ss[j] = cs[j] * tau;
        // }

    // CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
    //     CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
    //     matA, d_X, &one, d_R, CUDA_R_64F,
    //     CUSPARSE_SPMV_ALG_DEFAULT, d_bufferA))
#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../../cuda_utils.h"
#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"

namespace CUSPARSE {

template <typename T>
void direct_cholesky_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                           bool can_print = false) {
    /* direct Cholesky solve => performs Chol factorization K = LL^T and then triangular solves
        Best for solving with same matrix and multiple rhs vectors such as aeroelastic + linear
       structures. */

    // example in NVIDIA website, https://docs.nvidia.com/cuda/cusparse/
    // with search: For example, suppose A is a real m-by-m matrix, where m=mb*blockDim.
    // The following code solves precondition system M*y = x, where M is the product of Cholesky
    // factorization L and its transpose.

    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // NOTE : need to ensure the rowp, cols and values are of cholesky pattern here
    // that is no cols entries in upper triangular portion and values match only diag + lower too
    // there is a method to call bsr_mat.switch_cholesky_pattern() or TODO : you could make cholesky
    // pattern from start and assemble like that

    if (can_print) {
        printf("begin cusparse direct LU solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    const BsrData &bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    T *d_vals = mat.getPtr();
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();

    // temp vector
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0;
    bsrsv2Info_t info_L = 0, info_LT = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_LT = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_LT = CUSPARSE_OPERATION_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_ichol0_factorization<T>(handle, descr_L, info_L, info_LT, &pBuffer, mb, nnzb,
                                              block_dim, d_vals, d_rowp, d_cols, trans_L, trans_LT,
                                              policy_L, policy_LT, dir);
    // CUSPARSE::debug_call(mb, nnzb, block_dim, d_rowp, d_cols);

    // Forward solve: L*z = rhs
    T alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_L, d_rhs, d_temp, policy_L,
                                         pBuffer));

    // Backward solve: L^T*soln = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_LT, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_LT, d_temp, d_soln,
                                         policy_LT, pBuffer));

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_LT);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}
}  // namespace CUSPARSE
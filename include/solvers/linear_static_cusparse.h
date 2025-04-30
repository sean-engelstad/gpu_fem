#pragma once

#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"

#include "../cuda_utils.h"
#include "utils/_utils.h"
#include "utils/_cusparse_utils.h"
#include "chrono"

namespace CUSPARSE {

template <typename T>
void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                     bool can_print = false) {
    // uses old CUSPARSE style because new CUSPARSE spMat objects don't support BSR
    DeviceVec<T> rhs_perm = bsr_pre_solve<DeviceVec<T>>(mat, rhs, soln);

    if (can_print) {
        printf("begin cusparse direct LU solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int blockDim = bsr_data.block_dim;
    index_t *d_bsrRowPtr = bsr_data.rowp;
    index_t *d_bsrColPtr = bsr_data.cols;
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_bsrVal = mat.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Constant scalar coefficienct
    const double alpha = 1.0;

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE, trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_LU_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb, nnzb, blockDim,
        d_bsrVal, d_bsrRowPtr, d_bsrColPtr, trans_L, trans_U, policy_L, policy_U, dir);

    // triangular solve L*z = x
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, 
        d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim, info_L, d_rhs, d_temp, policy_L, pBuffer));

    // triangular solve U*y = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U, 
        d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim, info_U, d_temp, d_soln, policy_U, pBuffer));

    // print out d_soln
    // cudaMemcpy

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    bsr_post_solve<DeviceVec<T>>(mat, rhs, soln);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

template <typename T>
void GMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                     bool can_print = false) {
    // TODO : add preconditioner, LU factorization, etc. use direct_LU_solve above as a baseline
}

};  // namespace CUSPARSE
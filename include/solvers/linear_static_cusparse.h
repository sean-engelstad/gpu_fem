#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../cuda_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"

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
    index_t *d_bsrRowPtr = bsr_data.rowPtr;
    index_t *d_bsrColPtr = bsr_data.colPtr;
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
    cusparseStatus_t status;

    // Constant scalar coefficienct
    const double alpha = 1.0;

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    bsrilu02Info_t info_M = 0;
    bsrsv2Info_t info_L = 0;
    bsrsv2Info_t info_U = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // step 1: create a descriptor which contains
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for bsrilu02 and two info's for bsrsv2
    cusparseCreateBsrilu02Info(&info_M);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and
    // allocate the buffer
    cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr, d_bsrColPtr,
                                 blockDim, info_M, &pBufferSize_M);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal, d_bsrRowPtr,
                               d_bsrColPtr, blockDim, info_L, &pBufferSize_L);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal, d_bsrRowPtr,
                               d_bsrColPtr, blockDim, info_U, &pBufferSize_U);
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete LU factorization on M
    //     perform analysis of triangular solve on L
    //     perform analysis of triangular solve on U
    //     The lower(upper) triangular part of M has the same sparsity
    //     pattern as L(U), we can do analysis of bsrilu0 and bsrsv2
    //     simultaneously.
    //
    // Notes:
    // bsrilu02_analysis() ->
    //   Executes the 0 fill-in ILU with no pivoting
    //
    // cusparseXbsrilu02_zeroPivot() ->
    //   is a blocking call. It calls
    //   cudaDeviceSynchronize() to make sure all previous kernels are done.
    //
    // cusparseDbsrsv2_analysis() ->
    //   output is the info structure filled with information collected
    //   during he analysis phase (that should be passed to the solve phase
    //   unchanged).
    //
    // The variable "info" contains the structural zero or numerical zero

    cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr, d_bsrColPtr,
                               blockDim, info_M, policy_M, pBuffer);
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // online forum suggested moving the L and U sparsity analysis
    // to after the M computation (tried it but didn't make a difference)
    cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal, d_bsrRowPtr,
                             d_bsrColPtr, blockDim, info_L, policy_L, pBuffer);
    cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal, d_bsrRowPtr,
                             d_bsrColPtr, blockDim, info_U, policy_U, pBuffer);

    // step 5: M = L * U
    cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim,
                      info_M, policy_M, pBuffer);
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_bsrVal, d_bsrRowPtr,
                          d_bsrColPtr, blockDim, info_L, d_rhs, d_temp, policy_L, pBuffer);

    // step 7: solve U*y = z
    cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U, d_bsrVal, d_bsrRowPtr,
                          d_bsrColPtr, blockDim, info_U, d_temp, d_soln, policy_U, pBuffer);

    // print out d_soln
    // cudaMemcpy

    // step 8: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrilu02Info(info_M);
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

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double *ptr;
} Vec;

#if defined(NDEBUG)
#define PRINT_INFO(var)
#else
#define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

template <typename T>
T get_resid(BsrMat<DeviceVec<T>> mat, DeviceVec<T> rhs, DeviceVec<T> soln) {
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int blockDim = bsr_data.block_dim;
    index_t *d_bsrRowPtr = bsr_data.rowPtr;
    index_t *d_bsrColPtr = bsr_data.colPtr;
    T *d_rhs = rhs.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();
    T *d_bsrVal = mat.getPtr();

    // init cublas handle, etc.
    // cudaError_t cudaStat;
    cublasStatus_t blas_stat;
    cublasHandle_t cublas_handle;
    blas_stat = cublasCreate(&cublas_handle);

    // also make cusparse handle
    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseStatus_t cusparse_status;

    // Descriptor for the BSR matrix
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Step 1: Perform the matrix-vector product: d_temp = K * u
    double alpha = 1.0, beta = 0.0;
    cusparse_status = cusparseDbsrmv(
        cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
        &alpha, descr, d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim, d_soln, &beta, d_temp);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE bsrmv failed!\n");
        return EXIT_FAILURE;
    }

    // Step 2: Compute the residual: d_temp = d_temp - f
    double alpha2 = -1.0;
    cublasDaxpy(cublas_handle, rhs.getSize(), &alpha2, d_rhs, 1, d_temp, 1);

    // Step 3: Compute max residual
    int maxIndex;
    double maxResidual;
    cublasIdamax(cublas_handle, rhs.getSize(), d_temp, 1, &maxIndex);

    int zeroBasedIndex = maxIndex - 1;  // Idamax uses 1-based for some reason..
    cudaMemcpy(&maxResidual, d_temp + zeroBasedIndex, sizeof(double), cudaMemcpyDeviceToHost);

    // Optionally zero out the temp array
    // cudaMemset(d_temp, 0, numRows * sizeof(float));

    // Free resources
    cudaFree(d_temp);
    cusparseDestroyMatDescr(descr);

    return maxResidual;
}
};  // namespace CUSPARSE
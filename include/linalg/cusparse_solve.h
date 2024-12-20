#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>
/*
nvcc bsr_ilu.cu -lcublas -lcusparse -lcudart -lcusolver && ./a.out
*/

/*
Assumes all variables required for the anlaysis are already memory allocated on
the gpu.

Parameters:
-----------
handle = cusparse handle object
status = cusparse status obejct
mb = number of block rows
nnzb = number of nonzero blocks
blockDim = dimension of the block
d_bsrVal = device bsr value array
d_bsrRowPts = device bsr row ptr array
d_bsrColInd = device bsr col indices array
d_x = device right hand side array
d_y = device solution array
d_z = device intermediate solution array
*/

void cusparse_solve(cusparseHandle_t handle, cusparseStatus_t status, int mb, int nnzb,
           int blockDim, double *d_bsrVal, int *d_bsrRowPtr, int *d_bsrColInd,
           double *d_x, double *d_y, double *d_z) {
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

  // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate the
  // buffer
  cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, d_bsrVal,
                               d_bsrRowPtr, d_bsrColInd, blockDim, info_M,
                               &pBufferSize_M);
  cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
                             &pBufferSize_L);
  cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_U,
                             &pBufferSize_U);
  pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void **)&pBuffer, pBufferSize);

  // step 4: perform analysis of incomplete LU factorization on M
  //     perform analysis of triangular solve on L
  //     perform analysis of triangular solve on U
  //     The lower(upper) triangular part of M has the same sparsity pattern as
  //     L(U), we can do analysis of bsrilu0 and bsrsv2 simultaneously.
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

  cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, d_bsrVal,
                             d_bsrRowPtr, d_bsrColInd, blockDim, info_M,
                             policy_M, pBuffer);
  status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
  }
  cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal,
                           d_bsrRowPtr, d_bsrColInd, blockDim, info_L, policy_L,
                           pBuffer);
  cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal,
                           d_bsrRowPtr, d_bsrColInd, blockDim, info_U, policy_U,
                           pBuffer);

  // step 5: M = L * U
  cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr,
                    d_bsrColInd, blockDim, info_M, policy_M, pBuffer);
  status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("block U(%d,%d) is not invertible\n", numerical_zero,
           numerical_zero);
  }

  // step 6: solve L*z = x
  cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
                        d_x, d_z, policy_L, pBuffer);

  // step 7: solve U*y = z
  cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
                        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_U,
                        d_z, d_y, policy_U, pBuffer);

  // step 6: free resources
  cudaFree(pBuffer);
  cusparseDestroyMatDescr(descr_M);
  cusparseDestroyMatDescr(descr_L);
  cusparseDestroyMatDescr(descr_U);
  cusparseDestroyBsrilu02Info(info_M);
  cusparseDestroyBsrsv2Info(info_L);
  cusparseDestroyBsrsv2Info(info_U);
  cusparseDestroy(handle);
}
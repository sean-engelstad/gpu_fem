#pragma once

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include <algorithm>

namespace CUSPARSE {

template <typename T>
void perform_LU_factorization(
    cusparseHandle_t handle,
    cusparseMatDescr_t &descr_L,
    cusparseMatDescr_t &descr_U,
    bsrsv2Info_t &info_L,
    bsrsv2Info_t &info_U,
    void **pBuffer,  // allows return of allocated pointer
    int mb, int nnzb, int blockDim,
    T *vals,
    const int *rowp,
    const int *cols,
    const cusparseOperation_t trans_L,
    const cusparseOperation_t trans_U,
    const cusparseSolvePolicy_t policy_L,
    const cusparseSolvePolicy_t policy_U,
    const cusparseDirection_t dir
) {
    // performs symbolic and numeric LU or ILU factorization in CUSPARSE

    // temp objects for the factorization
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;

    // create M matrix object (for full numeric factorization)
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseCreateBsrilu02Info(&info_M);

    // init L matrix objects (for triangular solve)
    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseCreateBsrsv2Info(&info_L);

    // init U matrix objects (for triangular solve)
    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseCreateBsrsv2Info(&info_U);

    // symbolic and numeric factorizations
    CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, 
        vals, rowp, cols, blockDim, info_M, &pBufferSize_M));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, 
        vals, rowp, cols, blockDim, info_L, &pBufferSize_L));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, 
        vals, rowp, cols, blockDim, info_U, &pBufferSize_U));
    pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
    // cudaMalloc((void **)&pBuffer, pBufferSize);
    cudaMalloc(pBuffer, pBufferSize);

    // perform ILU symbolic factorization on L

    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, 
        vals, rowp, cols, blockDim, info_M, policy_M, *pBuffer));
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // analyze sparsity patern of L for efficient triangular solves
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, 
        vals, rowp, cols, blockDim, info_L, policy_L, *pBuffer));

    // analyze sparsity pattern of U for efficient triangular solves
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, 
        vals, rowp, cols, blockDim, info_U, policy_U, *pBuffer));

    // perform ILU numeric factorization (with M policy)
    CHECK_CUSPARSE(cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, vals, 
        rowp, cols, blockDim, info_M, policy_M, *pBuffer));
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }
}

};


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
    index_t *rowp = bsr_data.rowp;
    index_t *cols = bsr_data.cols;
    T *d_rhs = rhs.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();
    T *vals = mat.getPtr();

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
        &alpha, descr, vals, rowp, cols, blockDim, d_soln, &beta, d_temp);

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
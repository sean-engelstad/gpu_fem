#pragma once

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <algorithm>

#include "_utils.h"
#include "cublas_v2.h"

namespace CUSPARSE {

template <typename T>
void perform_ilu0_factorization(cusparseHandle_t handle, cusparseMatDescr_t &descr_L,
                                cusparseMatDescr_t &descr_U, bsrsv2Info_t &info_L,
                                bsrsv2Info_t &info_U,
                                void **pBuffer,  // allows return of allocated pointer
                                int mb, int nnzb, int blockDim, T *vals, const int *rowp,
                                const int *cols, const cusparseOperation_t trans_L,
                                const cusparseOperation_t trans_U,
                                const cusparseSolvePolicy_t policy_L,
                                const cusparseSolvePolicy_t policy_U,
                                const cusparseDirection_t dir) {
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
    CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, vals, rowp, cols,
                                                blockDim, info_M, &pBufferSize_M));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, vals, rowp,
                                              cols, blockDim, info_L, &pBufferSize_L));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, vals, rowp,
                                              cols, blockDim, info_U, &pBufferSize_U));
    pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
    // cudaMalloc((void **)&pBuffer, pBufferSize);
    cudaMalloc(pBuffer, pBufferSize);

    // perform ILU symbolic factorization on L
    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, vals, rowp, cols,
                                              blockDim, info_M, policy_M, *pBuffer));
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // analyze sparsity patern of L for efficient triangular solves
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, vals, rowp,
                                            cols, blockDim, info_L, policy_L, *pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // analyze sparsity pattern of U for efficient triangular solves
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, vals, rowp,
                                            cols, blockDim, info_U, policy_U, *pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // perform ILU numeric factorization (with M policy)
    CHECK_CUSPARSE(cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, vals, rowp, cols, blockDim,
                                     info_M, policy_M, *pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }
}

template <typename T>
void mat_vec_mult(BsrMat<DeviceVec<T>> mat, DeviceVec<T> x, DeviceVec<T> y) {
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = x.getSize();
    T *d_x = x.getPtr();
    T *d_y = y.getPtr();
    T *d_vals = mat.getPtr();

    // permute data in soln if guess is not zero
    x.permuteData(block_dim, iperm);

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // here compute A*x => y
    T a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_y));
    CHECK_CUDA(cudaDeviceSynchronize());

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, x);  // permuted separately with new vec
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, y);

    // Free resources
    cudaFree(d_x);  // since copy vector
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);
}

template <typename T>
T get_resid(BsrMat<DeviceVec<T>> mat, DeviceVec<T> rhs, DeviceVec<T> soln) {
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();

    // permute data in soln if guess is not zero
    soln.permuteData(block_dim, iperm);
    T *d_vals = mat.getPtr();
    T *d_tmp = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_resid = DeviceVec<T>(soln.getSize()).getPtr();

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    T a = 1.0, b = 0.0;

    // apply precond to rhs if in use
    // copy b or rhs to resid
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

    // then subtract Ax from b
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    CHECK_CUDA(cudaDeviceSynchronize());
    // resid -= A * x
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
    CHECK_CUDA(cudaDeviceSynchronize());

    T true_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &true_resid));
    CHECK_CUDA(cudaDeviceSynchronize());

    // debugging
    // int NPRINT = 100;
    // printf("x: ");
    // printVec<T>(NPRINT, soln.createHostVec().getPtr());
    // printf("b: ");
    // printVec<T>(NPRINT, rhs_perm.createHostVec().getPtr());
    // printf("Ax-b: ");
    // printVec<T>(NPRINT, DeviceVec<T>(N, d_resid).createHostVec().getPtr());

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // Free resources
    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_tmp));
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);

    return true_resid;
}

};  // namespace CUSPARSE

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double *ptr;
} Vec;

#if defined(NDEBUG)
#define PRINT_INFO(var)
#else
#define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

#pragma once

#ifdef USE_GPU

#include "../../cuda_utils.h"
#include "cublas_v2.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>

namespace CUSPARSE {

// void direct_chol_solve(CsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs,
//                        DeviceVec<T> &soln) {
//     cusolverSpHandle_t handle = NULL;
//     cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
//     cudaStream_t stream = NULL;
//     cusparseMatDescr_t descrA = NULL;

//     CsrData csr_data = mat.getCsrData();

//     int rowsA = csr_data.num_global; // number of rows of A
//     int colsA = csr_data.num_global; // number of columns of A
//     int nnzA = csr_data.nnz;         // number of nonzeros of A
//     // int baseA = 0;  // base index in CSR format

//     int *d_rowPtr = csr_data.rowPtr;
//     int *d_colPtr = csr_data.colPtr;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();
//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     // note mat data will change and contain LU
//     // however, this is good, allows for faster repeated linear solves
//     // which is the whole benefit of direct solves
//     T *d_values = mat.getPtr();

//     // CSR(A)

//     double tol = 1.e-12;
//     // can change reordering types
//     // [symrcm (symamd or csrmetisnd)] if reorder is 1 (2, or 3),
//     // int reorder = 0;     // no reordering
//     int reorder = 1;
//     int singularity = 0; // -1 if A is invertible under tol.

//     CHECK_CUSOLVER(cusolverSpCreate(&handle));
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUDA(cudaStreamCreate(&stream));
//     CHECK_CUSOLVER(cusolverSpSetStream(handle, stream));
//     CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, stream));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
//     CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//     CHECK_CUSPARSE((cusparseSetMatIndexBase(descrA,
//     CUSPARSE_INDEX_BASE_ZERO)));

//     // // verify if A has symmetric pattern or not
//     // int issym = 0;
//     // CHECK_CUSOLVER(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
//     //                                        h_csrRowPtrA, h_csrRowPtrA + 1,
//     //                                        h_csrColIndA, &issym));
//     // if (!issym) {
//     //     printf("Error: A has no symmetric pattern, please use LU or QR
//     \n");
//     //     exit(EXIT_FAILURE);
//     // }

//     CHECK_CUSOLVER(cusolverSpDcsrlsvchol(handle, rowsA, nnzA, descrA,
//     d_values,
//                                          d_rowPtr, d_colPtr, d_rhs, tol,
//                                          reorder, d_x, &singularity));
//     CHECK_CUDA(cudaDeviceSynchronize());
//     if (0 <= singularity) {
//         printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
//                singularity, tol);
//     }

//     // Clean slate
//     if (handle) {
//         CHECK_CUSOLVER(cusolverSpDestroy(handle));
//     }
//     if (cusparseHandle) {
//         CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     }
//     if (stream) {
//         CHECK_CUDA(cudaStreamDestroy(stream));
//     }
//     if (descrA) {
//         CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
//     }
//     if (d_csrValA) {
//         CHECK_CUDA(cudaFree(d_csrValA));
//     }
//     if (d_csrRowPtrA) {
//         CHECK_CUDA(cudaFree(d_csrRowPtrA));
//     }
//     if (d_csrColIndA) {
//         CHECK_CUDA(cudaFree(d_csrColIndA));
//     }
//     if (d_x) {
//         CHECK_CUDA(cudaFree(d_x));
//     }
//     if (d_b) {
//         CHECK_CUDA(cudaFree(d_b));
//     }
// }

// void direct_chol_solve(CsrMat<HostVec<T>> &mat, HostVec<T> &rhs,
//                        HostVec<T> &soln) {
//     cusolverSpHandle_t handle = NULL;
//     cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
//     cudaStream_t stream = NULL;
//     cusparseMatDescr_t descrA = NULL;

//     CsrData csr_data = mat.getCsrData();

//     int rowsA = csr_data.num_global; // number of rows of A
//     int colsA = csr_data.num_global; // number of columns of A
//     int nnzA = csr_data.nnz;         // number of nonzeros of A
//     // int baseA = 0;  // base index in CSR format

//     int *h_rowPtr = csr_data.rowPtr;
//     int *h_colPtr = csr_data.colPtr;
//     T *h_rhs = rhs.getPtr();
//     T *h_soln = soln.getPtr();
//     HostVec<T> temp = HostVec<T>(soln.getSize());
//     T *h_temp = temp.getPtr();

//     // note mat data will change and contain LU
//     // however, this is good, allows for faster repeated linear solves
//     // which is the whole benefit of direct solves
//     T *h_values = mat.getPtr();

//     // CSR(A)

//     double tol = 1.e-12;
//     // can change reordering types
//     // [symrcm (symamd or csrmetisnd)] if reorder is 1 (2, or 3),
//     // int reorder = 0;     // no reordering
//     int reorder = 1;
//     int singularity = 0; // -1 if A is invertible under tol.

//     CHECK_CUSOLVER(cusolverSpCreate(&handle));
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUDA(cudaStreamCreate(&stream));
//     CHECK_CUSOLVER(cusolverSpSetStream(handle, stream));
//     CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, stream));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
//     CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//     CHECK_CUSPARSE((cusparseSetMatIndexBase(descrA,
//     CUSPARSE_INDEX_BASE_ZERO)));

//     // // verify if A has symmetric pattern or not
//     // int issym = 0;
//     // CHECK_CUSOLVER(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
//     //                                        h_csrRowPtrA, h_csrRowPtrA + 1,
//     //                                        h_csrColIndA, &issym));
//     // if (!issym) {
//     //     printf("Error: A has no symmetric pattern, please use LU or QR
//     \n");
//     //     exit(EXIT_FAILURE);
//     // }

//     CHECK_CUSOLVER(cusolverSpDcsrlsvcholHost(
//         handle, rowsA, nnzA, descrA, h_values, h_rowPtr, h_colPtr, h_rhs,
//         tol, reorder, h_soln, &singularity));

//     if (0 <= singularity) {
//         printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
//                singularity, tol);
//     }

//     // Clean slate
//     if (handle) {
//         CHECK_CUSOLVER(cusolverSpDestroy(handle));
//     }
//     if (cusparseHandle) {
//         CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     }
//     if (stream) {
//         CHECK_CUDA(cudaStreamDestroy(stream));
//     }
//     if (descrA) {
//         CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
//     }

//     // TODO : delete temp vecs down here
// }

template <typename T>
void direct_LU_solve_old(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs,
                         DeviceVec<T> &soln) {

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int blockDim = bsr_data.block_dim;
    int *d_bsrRowPtr = bsr_data.rowPtr;
    int *d_bsrColPtr = bsr_data.colPtr;
    T *d_rhs = rhs.getPtr();
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

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and allocate
    // the buffer
    cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, d_bsrVal,
                                 d_bsrRowPtr, d_bsrColPtr, blockDim, info_M,
                                 &pBufferSize_M);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L,
                               d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim,
                               info_L, &pBufferSize_L);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U,
                               d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim,
                               info_U, &pBufferSize_U);
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete LU factorization on M
    //     perform analysis of triangular solve on L
    //     perform analysis of triangular solve on U
    //     The lower(upper) triangular part of M has the same sparsity pattern
    //     as L(U), we can do analysis of bsrilu0 and bsrsv2 simultaneously.
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
                               d_bsrRowPtr, d_bsrColPtr, blockDim, info_M,
                               policy_M, pBuffer);
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, d_bsrVal,
                             d_bsrRowPtr, d_bsrColPtr, blockDim, info_L,
                             policy_L, pBuffer);
    cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, d_bsrVal,
                             d_bsrRowPtr, d_bsrColPtr, blockDim, info_U,
                             policy_U, pBuffer);

    // step 5: M = L * U
    cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, d_bsrVal, d_bsrRowPtr,
                      d_bsrColPtr, blockDim, info_M, policy_M, pBuffer);
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero,
               numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                          d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim, info_L,
                          d_rhs, d_temp, policy_L, pBuffer);

    // step 7: solve U*y = z
    cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
                          d_bsrVal, d_bsrRowPtr, d_bsrColPtr, blockDim, info_U,
                          d_temp, d_soln, policy_U, pBuffer);

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
}

// template <typename T>
// void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs,
//                      DeviceVec<T> &soln) {

//     // copy important inputs for Bsr structure out of BsrMat
//     // TODO : was trying to make some of these const but didn't accept it in
//     // final solve
//     BsrData bsr_data = mat.getBsrData();
//     int mb = bsr_data.nnodes;
//     int nnzb = bsr_data.nnzb;
//     int blockDim = bsr_data.block_dim;
//     int *d_rowPtr = bsr_data.rowPtr;
//     int *d_colPtr = bsr_data.colPtr;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();
//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     //
//     // https: //
//     // developer.nvidia.com/blog/accelerated-solution-sparse-linear-systems/

//     // copy kmat data vec since gets modified during LU
//     // otherwise we can't compute residual properly K * u - f
//     T *d_values = mat.getVec().copyVec().getPtr();

//     /*
//     Cusparse documentation
//     The function cusparseSpSM_bufferSize() returns the size of the workspace
//     needed by cusparseSpSM_analysis() and cusparseSpSM_solve(). The function
//     cusparseSpSM_analysis() performs the analysis phase, while
//     cusparseSpSM_solve() executes the solve phase for a sparse triangular
//     linear system. The opaque data structure spsmDescr is used to share
//     information among all functions. The function cusparseSpSM_updateMatrix()
//     updates spsmDescr with new matrix values.
//     */

//     // Initialize cuSPARSE handle
//     cusparseHandle_t handle;
//     CHECK_CUSPARSE(cusparseCreate(&handle));

//     // Create a cuSPARSE matrix descriptor
// cusparseSpMatDescr_t matA;
// CHECK_CUSPARSE(cusparseCreateBsr(
//     &matA, mb, mb, nnzb, blockDim, blockDim, d_rowPtr, d_colPtr,
//     d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F, CUSPARSE_ORDER_ROW));

//     // TODO : need to manually convert from BSR to CSR myself before doing
//     // this.. temporarily convert to CSR format
//     // cusparseSpMatDescr_t matA_CSR;
//     // CHECK_CUSPARSE(cusparseCreateCsr(&matA_CSR, brows, bcols, nnz,
//     //                                  csrRowOffsets, csrColInd, csrValues,
//     //                                  CUSPARSE_INDEX_32I,
//     CUSPARSE_INDEX_32I,
//     //                                  CUSPARSE_INDEX_BASE_ZERO,
//     CUDA_R_64F));

//     // temporarily convert the matrix to CSR for factorization?
//     // I suppose we could do that here instead of C++ later..
//     // cusparseSpMatDescr_t matA_CSR;

//     // Create a dense matrix descriptor for the right-hand side vector
//     cusparseDnMatDescr_t matB;
//     CHECK_CUSPARSE(cusparseCreateDnMat(&matB, mb, 1, mb, d_rhs, CUDA_R_64F,
//                                        CUSPARSE_ORDER_ROW));

//     // Create a dense matrix descriptor for the result vector
//     cusparseDnMatDescr_t matC;
//     CHECK_CUSPARSE(cusparseCreateDnMat(&matC, mb, 1, mb, d_soln, CUDA_R_64F,
//                                        CUSPARSE_ORDER_ROW));

//     // Create sparse matrix solve descriptor
//     cusparseSpSMDescr_t spsmDescr;
//     CHECK_CUSPARSE(cusparseSpSM_createDescr(&spsmDescr));

//     // Choose algorithm for sparse matrix solve
//     cusparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;

//     // create buffer size for LU factorization
//     size_t bufferSize;
//     double alpha = 1.0;
//     const void *alpha_ptr = reinterpret_cast<const void *>(&alpha);

//     CHECK_CUSPARSE(cusparseSpSM_bufferSize(
//         handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_ptr, matA, matB, matC,
//         CUDA_R_64F, alg, spsmDescr, &bufferSize));

//     // create buffer for sparse matrix solve
//     void *d_buffer;
//     CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

//     // do analysis to get in A in LU format
//     cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                           CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_ptr, matA,
//                           matB, matC, CUDA_R_64F, alg, spsmDescr, d_buffer);

//     CHECK_CUSPARSE(cusparseSpSM_solve(handle,
//     CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                       alpha_ptr, matA, matB, matC,
//                                       CUDA_R_64F, alg, spsmDescr));

//     // CHECK_CUSPARSE(cusparseSpSM_analysis(
//     //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //     CUSPARSE_OPERATION_NON_TRANSPOSE, matA, nullptr, CUDA_R_64F,
//     //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr,
//     //     &bufferSizeSM)); // spSMDescr, &bufferSizeSM) // nullptr,
//     //     &bufferSizeSM)

//     // // Allocate buffer for analysis
//     // void *d_bufferSM;
//     // CHECK_CUDA(cudaMalloc(&d_bufferSM, bufferSizeSM));

//     // // LU analysis step
//     // CHECK_CUSPARSE(cusparseSpSM_analysis(
//     //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //     CUSPARSE_OPERATION_NON_TRANSPOSE, matA, nullptr, CUDA_R_64F,
//     //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr, d_bufferSM));

//     // // Create descriptors for L and U (triangular structure)
//     // cusparseSpMatDescr_t matL, matU;

//     // // Lower triangular matrix (L)
//     // CHECK_CUSPARSE(cusparseCreateBlockedSparseMat(
//     //     &matL, mb, mb, nnzb, d_rowPtr, d_colPtr, d_values, blockDim,
//     //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
//     // CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL,
//     CUSPARSE_SPMAT_TRIANGULAR,
//         // CUSPARSE_SPMAT_TRIANGULAR_LOWER));

//         // // Upper triangular matrix (U)
//         // CHECK_CUSPARSE(cusparseCreateBlockedSparseMat(
//         //     &matU, mb, mb, nnzb, d_rowPtr, d_colPtr, d_values, blockDim,
//         //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
//         // CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU,
//         CUSPARSE_SPMAT_TRIANGULAR,
//         // CUSPARSE_SPMAT_TRIANGULAR_UPPER));

//         // // Solution for L*y = f  (y is d_temp, f is d_rhs)
//         // // Solution for U*x = y  (x is d_soln, y is d_rhs)

//         // // Perform LU factorization (in place update of d_values)
//         // CHECK_CUSPARSE(cusparseSpSM_solve(
//         //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         //     CUSPARSE_OPERATION_NON_TRANSPOSE, matL, d_rhs, d_temp,
//         //     CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr,
//         d_bufferSM));

//         // CHECK_CUSPARSE(cusparseSpSM_solve(
//         //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         //     CUSPARSE_OPERATION_NON_TRANSPOSE, matU, d_temp, d_soln,
//         CUDA_R_64F,
//         //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr, d_bufferSM));

//         // Cleanup
//         CHECK_CUSPARSE(cusparseSpSM_destroyDescr(spsmDescr));
//     CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
//     CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
//     CHECK_CUSPARSE(cusparseDestroySpMat(matA));
//     CHECK_CUDA(cudaFree(d_buffer));

//     // Destroy cuSPARSE handle
//     CHECK_CUSPARSE(cusparseDestroy(handle));
// }

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double *ptr;
} Vec;

#if defined(NDEBUG)
#define PRINT_INFO(var)
#else
#define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

// TODO : need to debug this before I compile with it..int
int _gpu_CG(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m,
            cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matL, Vec d_B,
            Vec d_X, Vec d_R, Vec d_R_aux, Vec d_P, Vec d_T, Vec d_tmp,
            void *d_bufferMV, int maxIterations, double tolerance) {

    // source:
    //
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/cg/cg_example.c

    const double zero = 0.0;
    const double one = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                          cudaMemcpyDeviceToDevice))
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
    //--------------------------------------------------------------------------
    // ### 2 ### R_i_aux = L^-1 L^-T R_i
    size_t bufferSizeL, bufferSizeLT;
    void *d_bufferL, *d_bufferLT;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
    //    (a) L^-1 tmp => R_i_aux    (triangular solver)
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, d_R.vec,
        d_tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL,
        &bufferSizeL))
    CHECK_CUDA(cudaMalloc(&d_bufferL, bufferSizeL))
    CHECK_CUSPARSE(
        cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                              CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL))
    CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
    CHECK_CUSPARSE(cusparseSpSV_solve(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, d_R.vec,
        d_tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))

    //    (b) L^-T R_i => tmp    (triangular solver)
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrLT))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, d_tmp.vec,
        d_R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT,
        &bufferSizeLT))
    CHECK_CUDA(cudaMalloc(&d_bufferLT, bufferSizeLT))
    CHECK_CUSPARSE(cusparseSpSV_analysis(
        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, d_tmp.vec,
        d_R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT,
        d_bufferLT))
    CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
    CHECK_CUSPARSE(cusparseSpSV_solve(
        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, d_tmp.vec,
        d_R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT))
    //--------------------------------------------------------------------------
    // ### 3 ### P0 = R0_aux
    CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                          cudaMemcpyDeviceToDevice))
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CHECK_CUBLAS(
        cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta))
    //--------------------------------------------------------------------------
    // ### 4 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 0; i < maxIterations; i++) {
        printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
        //     (a) T  = A * P_i
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    matA, d_P.vec, &zero, d_T.vec, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
        //     (b) denominator = (T, P_i)
        double denominator;
        CHECK_CUBLAS(
            cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1, &denominator))
        //     (c) alpha = delta / denominator
        double alpha = delta / denominator;
        PRINT_INFO(delta)
        PRINT_INFO(denominator)
        PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 6 ###  X_i+1 = X_i + alpha * P
        //    (a) X_i+1 = -alpha * T + X_i
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1, d_X.ptr, 1))
        //----------------------------------------------------------------------
        // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
        //    (a) R_i+1 = -alpha * T + R_i
        double minus_alpha = -alpha;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1, d_R.ptr, 1))
        //----------------------------------------------------------------------
        // ### 8 ###  check ||R_i+1|| < threshold
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
        PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
        //    (a) L^-1 R_i+1 => tmp    (triangular solver)
        CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
        CHECK_CUSPARSE(
            cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                               CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))
        //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
        CHECK_CUSPARSE(cusparseSpSV_solve(
            cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, d_tmp.vec,
            d_R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT))
        //----------------------------------------------------------------------
        // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
        //    (a) delta_new => (R_i+1, R_aux_i+1)
        double delta_new;
        CHECK_CUBLAS(
            cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta_new))
        //    (b) beta => delta_new / delta
        double beta = delta_new / delta;
        PRINT_INFO(delta_new)
        PRINT_INFO(beta)
        delta = delta_new;
        //----------------------------------------------------------------------
        // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
        //    (a) P = beta * P
        CHECK_CUBLAS(cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1))
        //    (b) P = R_aux + P
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, m, &one, d_R_aux.ptr, 1, d_P.ptr, 1))
    }
    //--------------------------------------------------------------------------
    printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                          cudaMemcpyDeviceToDevice))
    // R = -A * X + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
    // check ||R||
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrLT))
    CHECK_CUDA(cudaFree(d_bufferL))
    CHECK_CUDA(cudaFree(d_bufferLT))
    return EXIT_SUCCESS;
}

// template <typename T>
// void iterative_PCG_solve(BsrMat<DeviceVec<T>> mat, DeviceVec<T> rhs,
//                          DeviceVec<T> soln, int maxIterations = 10000,
//                          double tol = 1e-12) {

//     BsrData bsr_data = mat.getBsrData();
//     int mb = bsr_data.nnodes;
//     int nnzb = bsr_data.nnzb;
//     int blockDim = bsr_data.block_dim;
//     int *d_rowPtr = bsr_data.rowPtr;
//     int *d_colPtr = bsr_data.colPtr;
//     int m = mb * blockDim;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();

//     // make the kitchen sink of temporary vectors
//     DeviceVec<T> resid = DeviceVec<T>(soln.getSize());
//     T *d_resid = resid.getPtr();

//     DeviceVec<T> resid_aux = DeviceVec<T>(soln.getSize());
//     T *d_resid_aux = resid_aux.getPtr();

//     DeviceVec<T> p_vec = DeviceVec<T>(soln.getSize());
//     T *d_P = p_vec.getPtr();

//     DeviceVec<T> T_vec = DeviceVec<T>(soln.getSize());
//     T *d_T = T_vec.getPtr();

//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     // note this changes the mat data to be LU (but that's the whole point
//     // of LU solve is for repeated linear solves we now just do triangular
//     // solves)
//     T *d_values = mat.getPtr();

//     // cuSPARSE handle and matrix descriptor
//     cusparseHandle_t cusparseHandle;
//     cusparseMatDescr_t descr;
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
//     cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//     cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

//     cublasHandle_t cublasHandle = NULL;
//     CHECK_CUBLAS(cublasCreate(&cublasHandle));

//     // Setup ILU preconditioner
//     bsrilu02Info_t iluInfo;
//     CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&iluInfo));

//     cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
//     void *buffer;
//     size_t bufferSize;
//     CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
//         cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr, d_values,
//         d_rowPtr, d_colPtr, blockDim, iluInfo, &bufferSize));
//     CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

//     // Analyze and compute ILU
//     CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
//         cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr, d_values,
//         d_rowPtr, d_colPtr, blockDim, iluInfo, policy, buffer));

//     CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, CUSPARSE_DIRECTION_ROW,
//     mb,
//                                      nnzb, descr, d_values, d_rowPtr,
//                                      d_colPtr, blockDim, iluInfo, policy,
//                                      buffer));

//     cusparseIndexBase_t baseIdx = CUSPARSE_INDEX_BASE_ZERO;
//     cusparseSpMatDescr_t matA, matL;
//     int *d_L_rows = d_A_rows;
//     int *d_L_columns = d_A_columns;
//     cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
//     cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

//     cusparseSpMatDescr_t matA;
//     CHECK_CUSPARSE(cusparseCreateBsr(
//         &matA, mb, mb, nnzb, blockDim, blockDim, d_rowPtr, d_colPtr,
//         d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F, CUSPARSE_ORDER_ROW));

//     CHECK_CUSPARSE(cusparseCreateBsr(&matA, m, m, nnz, d_A_rows, d_A_columns,
//                                      d_A_values, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_32I, baseIdx,
//                                      CUDA_R_64F))
//     // L
//     CHECK_CUSPARSE(cusparseCreateBsr(&matL, m, m, nnz, d_L_rows, d_L_columns,
//                                      d_L_values, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_32I, baseIdx,
//                                      CUDA_R_64F))
//     CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
//                                              &fill_lower,
//                                              sizeof(fill_lower)))
//     CHECK_CUSPARSE(cusparseSpMatSetAttribute(
//         matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit,
//         sizeof(diag_non_unit)))

//     // resources,
//     // https://
//     //
//     docs.nvidia.com/cuda/cuda-samples/index.html#cusolversp-linear-solver-%5B/url%5D
//     // github repo #1 for PCG,
//     // https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/cg
//     // github repo #2 for PCG,
//     // https://
//     //
//     github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradientPrecond

//     printf("CG loop:\n");
//     _gpu_CG(cublasHandle, cusparseHandle, m, matA, matL, d_rhs, d_soln,
//     d_resid,
//             d_resid_aux, d_P, d_T, d_temp, d_bufferMV, maxIterations,
//             tolerance);

//     // Clean up
//     CHECK_CUDA(cudaFree(buffer));
//     CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
//     CHECK_CUBLAS(cublasDestroy(cublasHandle));
// }

template <typename T>
T get_resid(BsrMat<DeviceVec<T>> mat, DeviceVec<T> rhs, DeviceVec<T> soln) {

    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int blockDim = bsr_data.block_dim;
    int *d_bsrRowPtr = bsr_data.rowPtr;
    int *d_bsrColPtr = bsr_data.colPtr;
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
        cusparse_handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &alpha, descr, d_bsrVal,
        d_bsrRowPtr, d_bsrColPtr, blockDim, d_soln, &beta, d_temp);

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

    int zeroBasedIndex = maxIndex - 1; // Idamax uses 1-based for some reason..
    cudaMemcpy(&maxResidual, d_temp + zeroBasedIndex, sizeof(double),
               cudaMemcpyDeviceToHost);

    // Optionally zero out the temp array
    // cudaMemset(d_temp, 0, numRows * sizeof(float));

    // Free resources
    cudaFree(d_temp);
    cusparseDestroyMatDescr(descr);

    return maxResidual;
}
}; // namespace CUSPARSE

#endif // USE_GPU
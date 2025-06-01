#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <chrono>
#include <iostream>

#include "../cuda_utils.h"
#include "cublas_v2.h"
#include "utils/_cusparse_utils.h"
#include "utils/_utils.h"

namespace CUSPARSE {

template <typename T>
void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                     bool can_print = false) {
    /* direct LU solve => performs LU factorization then L and U triangular solves.
        Best for solving with same matrix and multiple rhs vectors such as aeroelastic + linear
       structures Although need to check LU factorization held in place correctly and boolean to not
       refactorize maybe */
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    if (can_print) {
        printf("begin cusparse direct LU solve\n");
    }

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    auto start_fact = std::chrono::high_resolution_clock::now();
    CUSPARSE::perform_LU_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb, nnzb,
                                       block_dim, d_vals, d_rowp, d_cols, trans_L, trans_U,
                                       policy_L, policy_U, dir);
    auto end_fact = std::chrono::high_resolution_clock::now();

    // triangular solve L*z = x
    const double alpha = 1.0;
    auto start_triangular = std::chrono::high_resolution_clock::now();
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_L, d_rhs, d_temp, policy_L,
                                         pBuffer));

    // triangular solve U*y = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U, d_vals,
                                         d_rowp, d_cols, block_dim, info_U, d_temp, d_soln,
                                         policy_U, pBuffer));
    auto end_triangular = std::chrono::high_resolution_clock::now();

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // print timing data
    std::chrono::duration<double> fact_time = end_fact - start_fact,
                                  triang_solve_time = end_triangular - start_triangular;
    if (can_print) {
        printf("\tLU factorization in %.4e sec, triang solve in %.4e sec\n", fact_time.count(),
               triang_solve_time.count());
    }
}

template <typename T, bool use_precond = true, bool right = false>
void GMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                 int _n_iter = 100, int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8,
                 bool can_print = false, bool debug = false, int print_freq = 10) {
    /* GMRES iterative solve using a BsrMat on GPU with CUDA / CuSparse
        only supports T = double right now, may add float at some point (but float won't converge as
       deeply the residual, only about 1e-7) */
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // which type of preconditioners
    constexpr bool left_precond = use_precond && !right;
    constexpr bool right_precond = use_precond && right;

    // if (can_print) {
    //     printf("begin cusparse GMRES solve\n");
    // }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;

    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    int n_iter = min(_n_iter, bsr_data.nnodes);
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();

    // permute data in soln if guess is not zero
    soln.permuteData(block_dim, iperm);

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();

    // also make a temporary array for the preconditioner values
    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    // ILU0 equiv to ILU(k) if sparsity pattern has ILU(k)
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

    // make temp vecs
    T *d_tmp = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_tmp2 = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_resid = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_w = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_xR = DeviceVec<T>(soln.getSize()).getPtr();  // for right preconditioning

    // create initial cusparse and cublas handles --------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create ILU(0) preconditioner
    // -----------------------------
    // [equiv to ILU(k) precondioner if ILU(k) sparsity pattern used in BsrData object]

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_factor = std::chrono::high_resolution_clock::now();

    CUSPARSE::perform_LU_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                       mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                       trans_U, policy_L, policy_U, dir);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> factor_time = end_factor - start_factor;
    if (can_print) printf("ILU(0) factor time in %.4e sec\n", factor_time.count());

    // main GMRES solve
    // ----------------

    // initialize GMRES data, some on host, some on GPU
    // host GMRES data
    T g[n_iter + 1], cs[n_iter], ss[n_iter];
    T H[(n_iter + 1) * (n_iter)];

    // GMRES device data
    T *d_Vmat;  // , *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter + 1) * N * sizeof(T)));
    // CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    // cusparseDnVecDescr_t vec_V;
    // CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_64F));

    bool converged = false;
    int total_iter = 0;

    double triang_time = 0.0, SpMV_time = 0.0, GS_time = 0.0;

    for (int iouter = 0; iouter < max_iter / n_iter; iouter++) {
        int jj = n_iter - 1;

        // apply precond to rhs if in use
        // copy b or rhs to resid
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;

        // then subtract Ax from rhs
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto start_mult = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto end_mult = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> spmv_time = end_mult - start_mult;
        SpMV_time += spmv_time.count();
        if (can_print && debug) {
            printf("\tSpMV on GPU in %.4e sec\n", spmv_time.count());
        }

        // resid -= A * x
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        T init_true_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_true_resid));

        // now apply precond to the resid
        if constexpr (left_precond) {
            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto start_triang = std::chrono::high_resolution_clock::now();

            // compute r0' = U^-1 L^-1 * r0
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_resid, policy_U, pBuffer));

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());  // take these out for speedup
            auto end_triang = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
            triang_time += triang_time_loc.count();
            if (can_print && debug)
                printf("\ttriang time loc %.4e, total %.4e\n", triang_time_loc.count(),
                       triang_time);
        }

        // temp debug
        // if (debug) {
        //     CHECK_CUDA(cudaMemcpy(d_x, d_resid, N * sizeof(T), cudaMemcpyDeviceToHost));
        //     // now also inverse permute the soln data
        //     permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);
        //     return;
        // }

        // GMRES initial residual
        // assumes here d_X is 0 initially => so r0 = b - Ax
        T beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        if (can_print)
            printf("GMRES init resid = true %.9e, precond %.9e\n", init_true_resid, beta);
        g[0] = beta;

        // set v0 = r0 / beta (unit vec)
        a = 1.0 / beta;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));

        // then begin main GMRES iteration loop!
        for (int j = 0; j < n_iter; j++, total_iter++) {
            if constexpr (right_precond) {
                // U^-1 L^-1 * vj => vj precond solve here
                a = 1.0;
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_triang = std::chrono::high_resolution_clock::now();
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, &d_Vmat[j * N], d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto end_triang = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
                triang_time += triang_time_loc.count();

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));
            }

            if constexpr (left_precond) {
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_mult = std::chrono::high_resolution_clock::now();

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                              descrA, d_vals, d_rowp, d_cols, block_dim,
                                              &d_Vmat[j * N], &b, d_w));
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());

                auto end_mult = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> spmv_time = end_mult - start_mult;
                SpMV_time += spmv_time.count();
                if (can_print && debug && j % print_freq == 0) {
                    printf("\tSpMV on GPU in %.4e sec, total %.4e\n", spmv_time.count(), SpMV_time);
                }

                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_triang = std::chrono::high_resolution_clock::now();

                // U^-1 L^-1 * w => w precond solve here
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));

                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto end_triang = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
                triang_time += triang_time_loc.count();
                if (can_print && debug && j % print_freq == 0) {
                    printf("\ttriang time loc %.4e, total %.4e\n", triang_time_loc.count(),
                           triang_time);
                }
            }

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto start_GS = std::chrono::high_resolution_clock::now();

            // now update householder matrix
            for (int i = 0; i < j + 1; i++) {
                // compute w -= <w, vi> * vi
                CHECK_CUBLAS(
                    cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &H[n_iter * i + j]));
                // if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[n_iter * i + j]);
                a = -H[n_iter * i + j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
            }

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto end_GS = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> GS_loc = end_GS - start_GS;
            GS_time += GS_loc.count();
            // if (can_print && debug && j % print_freq == 0) {
            //     printf("\tGS_time_loc %.4e, total %.4e\n", GS_loc.count(), GS_time);
            // }

            // norm of w => H_{j+1,j}
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[n_iter * (j + 1) + j]));

            // v_{j+1} column unit vec = w / H_{j+1,j}
            a = 1.0 / H[n_iter * (j + 1) + j];
            CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j + 1) * N], 1));

            // then givens rotations to elim householder matrix
            for (int i = 0; i < j; i++) {
                T temp = H[i * n_iter + j];
                H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i + 1) + j];
                H[n_iter * (i + 1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i + 1) + j];
            }

            T Hjj = H[n_iter * j + j], Hj1j = H[n_iter * (j + 1) + j];
            cs[j] = Hjj / sqrt(Hjj * Hjj + Hj1j * Hj1j);
            ss[j] = cs[j] * Hj1j / Hjj;

            T g_temp = g[j];
            g[j] *= cs[j];
            g[j + 1] = -ss[j] * g_temp;

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (can_print && (j % print_freq == 0))
                printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

            // if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            H[n_iter * j + j] = cs[j] * H[n_iter * j + j] + ss[j] * H[n_iter * (j + 1) + j];
            H[n_iter * (j + 1) + j] = 0.0;

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                if (can_print)
                    printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                jj = j;
                converged = true;
                break;
            }
        }  // end of inner loop

        // now solve Hessenberg triangular system
        // only up to size jj+1 x jj+1 where we exited on iteration jj
        T *Hred = new T[(jj + 1) * (jj + 1)];
        // TODO : this doesn't transpose, rewrite it like new deflated GMRES
        for (int i = 0; i < jj + 1; i++) {
            for (int j = 0; j < jj + 1; j++) {
                // in-place transpose to be compatible with column-major cublasDtrsv later on
                Hred[(jj + 1) * i + j] = H[n_iter * j + i];

                // Hred[(jj+1) * i + j] = H[n_iter * i + j];
            }
        }

        // now print out Hred
        // if (debug) {
        //     printf("Hred:");
        //     printVec<T>((jj + 1) * (jj + 1), Hred);
        //     printf("gred:");
        //     printVec<T>((jj + 1), g);
        // }

        // TODO : switch to solving this part on the GPU
        // now copy data from Hred host to device
        T *d_Hred;
        CHECK_CUDA(cudaMalloc(&d_Hred, (jj + 1) * (jj + 1) * sizeof(T)));
        CHECK_CUDA(
            cudaMemcpy(d_Hred, Hred, (jj + 1) * (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // also create gred vector on the device
        T *d_gred;
        CHECK_CUDA(cudaMalloc(&d_gred, (jj + 1) * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_gred, g, (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // now solve Hessenberg system H * y = g
        // T *d_y;
        // CHECK_CUDA(cudaMalloc(&d_y, (jj+1) * sizeof(T)));
        CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, jj + 1, d_Hred, jj + 1, d_gred, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        // writes g => y inplace

        // now copy back to the host
        T *h_y = new T[jj + 1];
        CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

        if constexpr (left_precond) {
            // now compute the matrix product soln = V * y one column at a time
            // zero solution (d_x is already zero)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_x, 1));
            }
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if constexpr (right_precond) {
            // zero xR
            CHECK_CUDA(cudaMemset(d_xR, 0.0, N * sizeof(T)));

            // now compute matrix product xR = V * y (the preconditioned solution first)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_xR, 1));
            }

            // then compute xR = M^-1 xR (un-preconditions it back to x space)
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_xR, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_xR, policy_U, pBuffer));

            // then update x = x_0 + xR
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_xR, 1, d_x, 1));
        }

        if (converged) break;

    }  // end of outer iterations

    if (debug)
        printf("GMRES: triang time %.4e, SpMV time %.4e, GS_time %.4e\n", triang_time, SpMV_time,
               GS_time);

    // check final residual
    // --------------------

    // copy rhs into resid again
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

    T final_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

    // debugging
    // int NPRINT = 100;
    // printf("x: ");
    // printVec<T>(NPRINT, soln.createHostVec().getPtr());
    // printf("b: ");
    // printVec<T>(NPRINT, rhs_perm.createHostVec().getPtr());
    // printf("A*x: ");
    // printVec<T>(NPRINT, DeviceVec<T>(N, d_tmp).createHostVec().getPtr());
    // printf("Ax-b: ");
    // printVec<T>(NPRINT, DeviceVec<T>(N, d_resid).createHostVec().getPtr());

    // now apply preconditioner to the residual
    if constexpr (left_precond) {
        // zero vec_tmp
        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));

        // ILU solve U^-1 L^-1 * b
        // L^-1 * b => tmp
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L,
                                             d_resid, d_tmp, policy_L, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_resid, policy_U, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    T final_precond_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_precond_resid));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (can_print)
        printf("GMRES converged to %.4e resid, %.4e precond resid in %d iterations\n", final_resid,
               final_precond_resid, total_iter);

    // cleanup and inverse permute the solution for exit
    // -------------------------------------------------

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements

    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_tmp));

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

template <typename T>
void HGMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln, int _m = 100,
                  int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false,
                  bool debug = false, int print_freq = 10) {
    /* householder orthog, right preconditioned GMRES solver, more numerically stable than MGS GMRES
            based off working python implementation in `cuda_examples/gmres/_gmres_util.py :
            gmres_householder() of p. 160 Saad, Sparse Linear Systems
     */

    auto start = std::chrono::high_resolution_clock::now();

    // apply permutations to incoming vecs (rhs and initial soln x0)
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    printf("here0\n");

    // setup GMRES storage
    // -------------------
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    int m = min(_m, bsr_data.nnodes);  // subspace size
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();
    T *d_vals = mat.getPtr();
    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));
    T *d_tmp = DeviceVec<T>(N).getPtr();
    T *d_tmp2 = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_h = DeviceVec<T>(N).getPtr();
    T *d_v = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();
    T one = 1.0;
    T *d_one = HostVec<T>(1, &one).createDeviceVec().getPtr();

    printf("here0.1\n");

    // permute soln with nodal reordering
    soln.permuteData(block_dim, iperm);

    // Hessenberg system and householder vec storage
    T *g = new T[m + 1];
    T *cs = new T[m];
    T *ss = new T[m];
    T *y = new T[m];
    T *HT = new T[(m + 1) * m];
    // zero out Hessenberg matrix and RHS
    memset(HT, 0.0, (m + 1) * m * sizeof(T));
    memset(g, 0.0, (m + 1) * sizeof(T));

    // storing HT this time, so each col of H consecutive now
    T *d_Wmat;  // householder vecs
    CHECK_CUDA(cudaMalloc((void **)&d_Wmat, (m + 1) * N * sizeof(T)));

    printf("here0.2\n");

    // create initial cusparse and cublas handles
    // ------------------------------------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create ILU(0) preconditioner
    // -----------------------------
    // [equiv to ILU(k) precondioner if ILU(k) sparsity pattern used in BsrData object]

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    if (debug) CHECK_CUDA(cudaDeviceSynchronize());
    auto start_factor = std::chrono::high_resolution_clock::now();

    CUSPARSE::perform_LU_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                       mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                       trans_U, policy_L, policy_U, dir);

    if (debug) CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> factor_time = end_factor - start_factor;
    if (can_print && debug) printf("ILU(0) factor time in %.4e sec\n", factor_time.count());

    printf("here1\n");

    // GMRES restart and inner loops begin
    // -----------------------------------

    bool converged = false;
    int total_iter = 0;

    double triang_time = 0.0, SpMV_time = 0.0, GS_time = 0.0;

    for (int irestart = 0; irestart < max_iter / m; irestart++) {
        // store final subspace size
        int mm = m;

        // compute initial residual r0
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto start_mult = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto end_mult = std::chrono::high_resolution_clock::now();
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        T beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        g[0] = beta;  // RHS orig beta * e1, will get Givens rotationed though

        // copy resid into z
        CHECK_CUDA(cudaMemcpy(d_z, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));

        if (can_print) printf("GMRES init resid = %.9e\n", beta);
        std::chrono::duration<double> spmv_time = end_mult - start_mult;
        SpMV_time += spmv_time.count();
        if (can_print && debug) {
            printf("\tSpMV on GPU in %.4e sec\n", spmv_time.count());
        }

        // printf("here2\n");

        // begin main GMRES Householder loop (one startup loop, diff than MGS method)
        for (int j = 0; j < m + 1; j++, total_iter++) {
            /* compute Householder unit vector w_j */
            T *d_w = &d_Wmat[j * N];  // hold w_j

            T sigma;  // sigma = norm2(z[j:])
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, &d_z[j], 1, &sigma));

            T vsub0;  // vsub0 = vsub[0] copied to host
            CHECK_CUDA(cudaMemcpy(&vsub0, &d_z[j], sizeof(T), cudaMemcpyDeviceToHost));
            T sign_vsub0 = vsub0 >= 0.0 ? 1.0 : -1.0;
            T alpha = -sign_vsub0 * sigma;

            // zero w tmp vec
            CHECK_CUDA(cudaMemset(d_w, 0.0, N * sizeof(T)));

            // printf("here3\n");

            // copy u = vsub - alpha * e1 into the w[j:] (from jth entry on so first j values 0)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_z[j], 1, &d_w[j], 1));
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, 1, &a, d_one, 1, &d_w[j], 1));

            // normalize w[j:] on to make it a unit vec (or just all of w)
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &nrm_w));
            T inrm_w = 1.0 / nrm_w;
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inrm_w, d_w, 1));

            // printf("here4\n");

            /* compute Hessenberg vector h_{j-1} */

            // h_{j-1} = P_j * z where P_j = I - 2 w_j w_j^T is a reflector
            //   or in short h_{j-1} = z - 2 <w_j, z> * w_j
            CHECK_CUDA(cudaMemcpy(d_h, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));
            T wz_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_z, 1, &wz_dot));
            // if (debug) printf("wz_dot %.4e\n", wz_dot);
            a = -2.0 * wz_dot;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_h, 1));

            // copy only up to 0:j or first j+1 vals of d_h into host H matrix (for col j-1)
            //  remember first one is startup iteration, should be ok to do device to host for now
            //  as long as small subspace size? TBD
            if (j > 0) {
                CHECK_CUDA(
                    cudaMemcpy(&HT[m * (j - 1)], d_h, (j + 1) * sizeof(T), cudaMemcpyDeviceToHost));

                // if (debug) {
                //     printf("H[:,%d] pre: ", j - 1);
                //     printVec<T>(j + 1, &HT[m * (j - 1)]);
                // }

                /* Givens rotations for Hessenberg matrix and RHS */

                // see cuda_examples/gmres/_gmres_utils.py : gmres_householder method Givens section
                for (int i = 0; i < j - 1; i++) {
                    T temp = HT[m * (j - 1) + i];
                    HT[m * (j - 1) + i] =
                        cs[i] * HT[m * (j - 1) + i] + ss[i] * HT[m * (j - 1) + (i + 1)];
                    HT[m * (j - 1) + i + 1] = -ss[i] * temp + ss[i] * HT[m * (j - 1) + (i + 1)];
                }

                T hx = HT[m * (j - 1) + j - 1] + 1e-12;
                T hy = HT[m * (j - 1) + j] + 1e-12;
                cs[j - 1] = hx / sqrt(hx * hx + hy * hy);
                ss[j - 1] = cs[j - 1] * hy / hx;

                // if (debug) printf("g[%d] pre Givens = %.4e\n", j - 1, g[j - 1]);
                // if (debug) printf("g[%d] pre Givens = %.4e\n", j, g[j]);
                // if (debug) {
                //     printf("H[:,%d]: ", j - 1);
                //     printVec<T>(j + 1, &HT[m * (j - 1)]);
                // }

                g[j] = -ss[j - 1] * g[j - 1];
                g[j - 1] *= cs[j - 1];

                HT[m * (j - 1) + (j - 1)] = cs[j - 1] * hx + ss[j - 1] * hy;
                HT[m * (j - 1) + j] = 0.0;

                // if (debug) printf("g[%d] = %.4e\n", j - 1, g[j - 1]);
                if (debug) printf("g[%d] = %.4e\n", j, g[j]);

                // check convergence of householder GMRES using Givens rotations of RHS
                if (abs(g[j]) < (abs_tol + beta * rel_tol)) {
                    if (can_print)
                        printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j]);
                    mm = j;
                    converged = true;
                    break;
                }  // end of conv if statement
            }      // end of Givens section

            /* apply Reflector backward orthogonalization to get v */
            // v = P1 * ... * Pj * ej
            // let ej stored in d_v device array
            CHECK_CUDA(cudaMemset(d_v, 0.0, N * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(&d_v[j], d_one, sizeof(T), cudaMemcpyDeviceToDevice));

            // perform reflector orthog first Pj then ... then P1
            for (int i = j; i >= 0; i--) {
                T *d_wi = &d_Wmat[i * N];
                T wv_dot;
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_v, 1, &wv_dot));
                a = -2.0 * wv_dot;
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_v, 1));
            }

            /* compute new Arnoldi vec and forward orthog */
            if (j <= m) {
                // z = Pj * ... * P1 * A * M^-1 * v

                /* first A * M^-1 * v into z */
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_v, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_z));

                /* then forward Householder reflectors */
                for (int i = 0; i < j + 1; i++) {
                    T *d_wi = &d_Wmat[i * N];
                    T wz_dot;
                    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_z, 1, &wz_dot));
                    a = -2.0 * wz_dot;
                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_z, 1));
                }
            }
        }  // end of inner subspace loop

        /* solve the Hessenberg system which is triangular thanks to the Givens rotations */

        // it has a size mm x mm where mm was the final iteration on exit of inner loop
        //      mm-1 to 0 inclusive is full size mm
        for (int i = mm - 1; i >= 0; i--) {
            T num = g[i];
            // subtract previous solved y terms
            for (int j = i + 1; j < mm; j++) {
                // numerator -= H[i,j] * y[j]
                num -= HT[m * j + i] * y[j];
            }
            y[i] = num / HT[m * i + i];
        }
        printf("y:");
        printVec<T>(mm, y);

        /* update solution z = Pi * (y[i] * ei + z) iteratively */
        // first set z to zero (full size N)
        CHECK_CUDA(cudaMemset(d_z, 0.0, N * sizeof(T)));
        for (int i = mm - 1; i >= 0; i--) {
            // d_tmp as ei
            CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(&d_tmp[i], d_one, sizeof(T), cudaMemcpyDeviceToDevice));

            // z += y[i] * ei
            a = y[i];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_z, 1));

            // z -= 2 * <z, wi> * wi (reflector Pi applied to z)
            T *d_wi = &d_Wmat[i * N];
            T wz_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_z, 1, &wz_dot));
            a = -2.0 * wz_dot;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_z, 1));
        }

        /* apply right precond z = M^-1 z */
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_z,
                                             d_tmp, policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_z, policy_U, pBuffer));

        /* final soln update x += z aka x += M^-1 * Vmm * ymm */
        a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_x, 1));

    }  // end of outer restart loop

    /* debug check final residual (should match final Givens rotation) */
    if (can_print && debug) {
        // copy rhs into resid again
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // then subtract Ax from b
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        // resid -= A * x
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        T final_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

        if (can_print)
            printf("GMRES converged to %.4e resid in %d iterations\n", final_resid, total_iter);
    }

    /* cleanup GMRES data and inv permute soln .. TODO : later persist some data in repeated solves?
     * maybe with class or struct? */

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements
    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_Wmat));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_tmp2));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_h));

    // print timing data
    CHECK_CUDA(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}  // end of HGMRES solve

};  // namespace CUSPARSE
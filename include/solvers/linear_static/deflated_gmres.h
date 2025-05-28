#pragma once

/* deflated GMRES linear solver for BSR matrices in CuSparse on the GPU, Sean Engelstad */

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../../cuda_utils.h"
#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"

/*
References the following papers:
// 1) GMRES deflated restart from "GMRES with adaptively deflated restarting and its performance on an
//        electromagnetic cavity problem" at https://www.sciencedirect.com/science/article/pii/S0168927411000705
// 2) thick restarting from "Thick-Restart Lanczos Method for Large 
//        Symmetric Eigenvalue problems" at https://epubs.siam.org/doi/epdf/10.1137/S0895479898334605

For deflated GMRES, you first perform m iterations of standard GMRES with:
    A * V_m = V_{m+1} * H_{m+1,m}         with A (NxN), V_m (Nxm), V_{m+1} (Nxm+1), H_{m+1,m} (m+1xm)
Then solve the update x = x_0 + V_m * y, with y an mx1 vector with:
    H_{m+1,m} * y = beta * e1
Here beta = |r_0| the init residual and this can be shown to be equivalent to min || b - Ax || in the Krylov subspace,
as Vm * (beta * e1) = r0 term. The system is solved using Givens rotations to make H_{m+1,m} upper triangular, with the
right hand side becoming a g vector after Givens rotations.
    H_{m+1,m} * y = g

After the standard GMRES solve of m iterations, the deflated GMRES strategy seeks to keep the lowest eigenvalue modes in 
A * x = lambda * x of A, which will help prevent stalling in the next restart. The eigenvalue problem can be solved in 
the Krylov subspace basis as follows:
    (H_{m+1,m}^T H_{m+1,m}) Y = Lambda * H_{m,m}^T Y
Where we take only the first Y_k deflated reduced eigenvectors. The full eigenvectors are then Phi_k = V_m * Y_k.
Common choices for the subspace size m and deflation size k are (m,k) = (30,10). When we do the restart, the Arnoldi matrix
is updated with a k+1 x k+1 nonzero block from the deflation eigenvectors. Namely, this part is:
    A * V_k' = V_{k+1}' * H_{k+1,k}'
with the primed or new bases the eigenvectors with:
    V_k' = Phi_k = V_m * Y_k and V_{k+1}' = [V_k', v_{m+1}]
    H_{k+1,k}' = [Y_k^T H_{m,m} Y_k; h_{m+1,m} e_m^T Y]
Then, the new Arnoldi system continues filling in the k+2 to m+1 H_{m+1,m}' matrix and adding new search directions to V_m' starting from k+2 entry.
Namely, the new full system is also,
    A * V_m' = V_{m+1}' * H_{m+1,m}'
The update on x = x_0 + V_m' * y minimizes the residual ||b - Ax|| = || r_0 - A * dx|| like before except now the k eigenvectors may be in the same direction
as r_0 (as they are l.c. of previous search directions). The min norm update is then given by || V_{m+1}' * V_{m+1}'^T * (r0 - A * dx) || like before, with now
    || r0 - A * V_m' * y || = || V_{m+1}' * (V_{m+1}'^T r0 - V_{m+1}'^T A * V_m' y) || = || c - H_{m+1,m} y ||
So the reduced basis update y is given by:
    H_{m+1,m}' * y = c
And Givens rotations can still be applied to H_{m+1,m}' and c to give an upper triangular system:
    H_{m+1,m}'' * y = g'
With final update x = x_0 + V_m' * y and then repeat. The deflated restart is supposed to be especially good at reducing solve time by less orthogonalization work
for especially stiff linear systems. Since last m+1,m entry is zero after final Givens rotation, becomes an mxm triangular system.
*/

namespace CUSPARSE {

template <typename T, bool right = false, bool modifiedGS = true, bool use_precond = true>
void GMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                 int subspace_size = 30, int deflation_size = 10, int max_iter = 300, 
                 T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false, 
                 bool debug = false, int print_freq = 10) {
                    
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
    CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, mat.get_nnz() * sizeof(T)));
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
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                         mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // main GMRES solve
    // ----------------
    int m = subspace_size, k = deflation_size;

    // Hessenberg linear system data (solved on host because so small and can be done separately on host
    // without any copying needed)
    T *H = new T[(m + 1) * m]; // hessenberg matrix
    T *g = new T[m+1]; // hessenberg RHS (Givens rotated)
    T *cs = new T[m]; // Givens cosines
    T *ss = new T[m]; // Givens sines
    T *y = new T[m];  // solution of Hessenberg system for update x = x0 + Vm * y
    // final Hessenberg system will become mxm (but need extra storage for final Givens application on m+1 row)
    // H matrix stored with typical row-major format so that mxm matrix is just first m^2 values

    // dense Krylov subspace vectors stored on device however
    T *d_V = DeviceVec<T>((m + 1) * N).getPtr(); // Kryov search directions
    T *d_Phi = DeviceVec<T>(k * N).getPtr(); // temporary storage for deflation eigenvecs (quickly stored back in d_V)

    bool converged = false;
    int total_iter = 0;

    for (int irestart = 0; irestart < max_iter / subspace_size; irestart++) {

        // compute r0 = b - A * x
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // compute xR0 = L * U * x0 for preconditioned initial guess (if right precond)
        if constexpr (right_precond) {
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim, d_tmp, &b, d_xR));
        }

        // then subtract Ax from rhs
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        // get initial residual beta = || r_0 ||, preconditioned if left precond
        T beta;
        if constexpr (left_precond) {
            // if left precond, compute M^-1 r0 = U^-1 L^-1 r0
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_resid, policy_U, pBuffer));
        }
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));

        // reset Hessenberg RHS g vec
        memset(g, 0.0, (m+1) * sizeof(T));

        // zero out V past k+1 for restart cases
        if (irestart > 0) CHECK_CUDA(cudaMemset(&d_V[(k+1) * N], 0.0, (m-k) * N * sizeof(T)));

        // report initial residual
        if (can_print)
            printf("GMRES init resid = %.9e\n", beta);
        g[0] = beta;

        // standard vs deflated restart section
        if (irestart == 0) {
            // standard restart, only for first startup
            // set v0 = r0 / beta and initial g[0] = beta for RHS of Hessenberg system
            a = 1.0 / beta;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));
        } else {
            // deflated restart, TBD
            // compute k many initial g vectors of RHS since previous eigenvectors may be in same
            // direction as r0. Namely g0 = V_{m+1}'^T r0, or first k values g[0:k] = V_k'^T * r0
            // where V_k' are the eigenvectors of previous subspace
            for (int i = 0; i < k; i++) {
                // store in g[i] = <v_i, r0>
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, &d_V[i * N], 1, d_resid, 1, &g[i]));
            }
        }

        // start past k+1 if restart, less new iterations with restarted cases
        int start = irestart == 0 ? 0 : k+1;

        // then begin main GMRES iteration loop!
        for (int j = start; j < m; j++, total_iter++) {

            // current subspace size if exit early
            mm = j;

            // compute w = A * v
            // -----------------
            if constexpr (right_precond) {
                // first compute U^-1 L^-1 vj => tmp
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, &d_V[j * N], d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));

                // then compute A * tmp => w, so in full w = A * M^-1 * vj
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));
            }

            if constexpr (left_precond) {
                // compute A * vj => w
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                              descrA, d_vals, d_rowp, d_cols, block_dim,
                                              &d_V[j * N], &b, d_w));

                // compute U^-1 * L^-1 * w => w
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));
            }

            // orthogonalize w against previous search directions (Gram-Schmidt)
            // -----------------------------------------------------------------

            // TODO : could add householder and modified GS with reorthogonalization here too
            if constexpr (modifiedGS) {
                // modified GS is more numerically stable but more flops than classical
                for (int i = 0; i < j + 1; i++) {
                    // compute H_{i,j} = <vi, w>
                    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, &d_V[i * N], 1, &H[m * i + j]));
                    if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[m * i + j]);

                    // w -= H_{i,j} * vi
                    a = -H[n_iter * i + j];
                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[i * N], 1, d_w, 1));
                }
            } else {
                // TODO : could implement classical GS like in regular GMRES, but then I need
                // H on the device.. prob not worth it (classical GS was slower because orthog less accurate, despite less flops)
                printf("classical GS not implemented in deflated GMRES yet..\n");
            }

            // compute H_{j+1,j} = || w ||
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[m * (j+1) + j]));

            // compute v_{j+1} = w / || w ||
            a = 1.0 / H[m * (j + 1) + j]; // fine to add in because V mat zeroed out each restart
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, &d_V[i * N], 1));

            // apply Givens rotations to Hessenberg matrix
            for (int i = 0; i < j; i++) {
                T temp = H[m * i + j];
                H[m * i + j] = cs[i] * H[m * i + j] + ss[i] * H[m * (i+1) + j];
                H[m * (i+1) + j] = -ss[i] * temp + cs[i] * H[m * (i+1) + j];
            }

            // apply Givens rotations to Hessenberg RHS
            T H1 = H[m * j + j], H2 = H[m * (j+1) + j];
            cs[j] = H1 / sqrt(H1 * H1 + H2 * H2);
            ss[j] = cs[j] * H2 / H1;
            g[j+1] = -ss[j] * g[j];
            g[j] *= cs[j];

            // compute new entries in the Hessenberg matrix for vj
            H[m * j + j] = cs[j] * H[m * j + j] + ss[j] * H[m * (j+1) + j];
            H[m * (j+1) + j] = 0.0;

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (can_print && (j % print_freq == 0))
                printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

            if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                if (can_print)
                    printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                jj = j;
                converged = true;
                break;
            }
        }  // end of inner loop for Krylov subspace

        // now solve Householder triangular system H * y = g of size mm x mm 
        //     (where mm might be less than m if exit early)
        for (int i = 0; i < mm; i++) {
            T num = g[i];
            // subtract previous solved y terms
            for (int j = i+1; j < mm; j++) {
                num -= H[mm * i + j] * y[j];
            }
            y[i] = num / H[mm * i + i];
        }

        // compute update x = x0 + V * y
        // -----------------------------

        if constexpr (left_precond) {
            for (int j = 0; j < mm + 1; j++) {
                a = y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[j * N], 1, d_x, 1));
            }
        }

        if constexpr (right_precond) {
            // now compute matrix product xR = V * y (the preconditioned solution first)
            for (int j = 0; j < mm + 1; j++) {
                a = y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[j * N], 1, d_xR, 1));
            }

            // compute U^-1 L^-1 * xR => x
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_xR, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_x, policy_U, pBuffer));
        }

        if (converged) break;

        // compute Ritz eigenvectors for deflated GMRES
        // --------------------------------------------

        // compute H^T H matrix and also H^T

    }  // end of outer iterations

    // check final residual
    // --------------------

    // compute final residual r(x) = b - Ax
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
    T final_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

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

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_resid, policy_U, pBuffer));
    }

    T final_precond_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_precond_resid));

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

};  // namespace CUSPARS





/* develop fast block-GS multicolor using small matrix for testing purposes.. */
// 7 node test matrix with 2x2 block dim for multicoloring (aka 14 DOF).. that I've made up

#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"

template <typename T>
void baseline_solve_single_color(const int icolor, T *h_defect, T *h_new_defect) {
    /* baseline to solve update for a single color with full LU pattern */
    // we'll want to match each color to the new fast version (but we'll only call one at a time)

    // start by setting up cusparse objects..
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // copy the defect to the device
    T *d_defect = HostVec<T>(14, h_defect).createDeviceVec().getPtr();

    // make a new temp color defect array, and only set this color part NZ
    int start, stop;
    if (icolor == 0) {
        start = 0, stop = 6;
    } else if (icolor == 1) {
        start = 6, stop = 10;
    } else if (icolor == 2) {
        start = 10, stop = 14; 
    }
    int ncolor_vals = stop - start;

    // compare this result with python also..
    T *d_defect_color = DeviceVec<T>(ncolor_vals).getPtr();
    cudaMemcpy(d_defect_color, &d_defect[start], ncolor_vals * sizeof(T), cudaMemcpyDeviceToDevice);

    // now make temp for soln change
    T *d_dsoln = DeviceVec<T>(14).getPtr();
    // get pointer for the color part of soln change
    T *d_dsoln_color = &d_dsoln[start];

    // define the matrix..
    const int nnodes = 7;
    int h_rowp[nnodes + 1] = {0, 3, 4, 5, 8, 11, 14, 16};
    const int nnzb = 16;
    int h_colors[nnzb] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    int h_rows[nnzb] =   {0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6};
    int h_cols[nnzb] =   {0, 3, 4, 1, 2, 0, 3, 6, 0, 4, 5, 0, 4, 5, 3, 6};
    const int block_dim = 2;
    const int nnz = nnzb * 4;

    T *h_vals = new T[nnz];
    memset(h_vals, 0.0, nnz * sizeof(T));
    int nnzb_diag = 7;
    int nnz_diag = nnzb_diag * 4;
    T *h_diag_vals = new T[nnz_diag];
    for (int inz = 0; inz < nnz; inz++) {
        int inzb = inz / 4; // block index (like on CSR pattern)
        int inner = inz % 4;
        int ix = inner / 2, iy = inner % 2; // inside each 2x2 block

        int inner_diag = ix == iy;
        T inner_val = 1.0 + 1.0 * inner_diag;

        int color = h_colors[inzb];
        int off_diag = h_rows[inzb] != h_cols[inzb];
        T csr_val = 3.0 - color - 0.4 * off_diag;

        h_vals[inz] = csr_val * inner_val;

        int block_row = h_rows[inzb], block_col = h_cols[inzb];
        if (block_row == block_col) {
            int diag_nnz = 4 * block_row + inner;
            h_diag_vals[diag_nnz] = h_vals[inz];
        }
    }

    // the diag matrix only..
    int h_diag_rowp[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int h_diag_cols[7] = {0, 1, 2, 3, 4, 5, 6};
    int diag_nnzb = 7;

    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    void *pBuffer = 0;
    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // for kmat SpMV
    cusparseMatDescr_t descr_kmat_L = 0, descr_kmat_U = 0;
    bsrsv2Info_t info_kmat_L = 0, info_kmat_U = 0;
    void *kmat_pBuffer = 0;

    // pass onto device
    int *d_diag_rowp = HostVec<int>(8, h_diag_rowp).createDeviceVec().getPtr();
    int *d_diag_cols = HostVec<int>(7, h_diag_cols).createDeviceVec().getPtr();
    T *d_temp = DeviceVec<T>(14).getPtr();
    T *d_temp2 = DeviceVec<T>(14).getPtr();
    T *d_temp3 = DeviceVec<T>(14).getPtr();
    T *d_kmat_vals = HostVec<T>(nnz, h_vals).createDeviceVec().getPtr();
    T *d_diag_LU_vals = HostVec<T>(nnz, h_diag_vals).createDeviceVec().getPtr();
    // printf("h_vals:");
    // printVec<T>(nnz, h_vals);

    // ILU(0) factor of the diag matrix..
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_nnzb, block_dim,
                                                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                 trans_U, policy_L, policy_U, dir);


    // now do LU solves
    T a = 1.0, b = 0.0;
    const double alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(
        cusparseHandle, dir, trans_L, nnodes, diag_nnzb, &alpha, descr_L,
        d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2, d_temp3,
        policy_L, pBuffer));  
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                    diag_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                        d_diag_rowp, d_diag_cols, block_dim, info_U,
                                        d_temp3, d_temp, policy_U, pBuffer));

    // // 2) update soln x_color += dx_color
    // int nrows_color = nblock_rows_color * block_dim;
    // T *d_soln_color = &d_soln.getPtr()[block_dim * start];
    // a = omega;
    // CHECK_CUBLAS(
    //     cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

    // a = -omega,
    // b = 1.0;  // so that defect := defect - mat*vec
    // CHECK_CUSPARSE(cusparseDbsrmv(
    //     cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
    //     block_dim, d_temp, &b, d_defect.getPtr()));
}

int main() {
    using T = double;
    T *h_defect = new T[14];
    T *h_new_defect = new T[14];
    for (int i = 0; i < 14; i++) {
        h_defect[i] = 1.234 - 2.189 * i;
    }

    baseline_solve_single_color<T>(0, h_defect, h_new_defect);
};
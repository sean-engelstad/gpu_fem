// geom multigrid for the shells
#pragma once
#include <cusparse_v2.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/bsr_mat.h"

// local includes for shell multigrid
#include "grid.cuh"

class ShellGrid {
   public:
    using T = double;
    using I = long long int;

    ShellGrid(int N_, BsrMat<DeviceVec<T>> Kmat_, DeviceVec<T> d_rhs_, HostVec<int> h_color_rowp_)
        : N(N_) {
        Kmat = Kmat_;
        d_rhs = d_rhs_;
        h_color_rowp = h_color_rowp_;
        block_dim = 6;
        nnodes = N / 6;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        // init helper methods
        buildColorLocalRowPointers();
        initCuda();
        buildDiagInvMat();
    }

    void initCuda() {
        // init handles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_defect = DeviceVec<T>(N);
        d_soln = DeviceVec<T>(N);
        d_temp = DeviceVec<T>(N).getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();

        // copy rhs into defect
        cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));
    }

    void buildDiagInvMat() {
        // first need to construct rowp and cols for diagonal (fairly easy)
        int *h_diag_rowp = new int[nnodes + 1];
        diag_inv_nnzb = nnodes;
        int *h_diag_cols = new int[nnodes];
        h_diag_rowp[0] = 0;

        for (int i = 0; i < nnodes; i++) {
            h_diag_rowp[i + 1] = i + 1;
            h_diag_cols[i] = i;
        }

        // printf("diag inv mat step 1:\n");
        // printf("h_diag_rowp: ");
        // printVec<int>(nnodes + 1, h_diag_rowp);
        // printf("h_diag_cols: ");
        // printVec<int>(nnodes, h_diag_cols);
        // printf("\n\n");
        // return;

        // on host, get the pointer locations in Kmat of the block diag entries..
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // printf("h_kmat_rowp: ");
        // printVec<int>(nnodes + 1, h_kmat_rowp);
        // printf("h_kmat_cols: ");
        // printVec<int>(min(100, kmat_nnzb), h_kmat_cols);

        // now copy to device
        d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
        d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

        // create the bsr data object on device
        auto d_diag_bsr_data =
            BsrData(nnodes, 6, diag_inv_nnzb, d_diag_rowp, d_diag_cols, nullptr, nullptr, false);

        // printf("h_diag_rowp: ");
        // printVec<int>(nnodes + 1, h_diag_rowp);
        // printf("h_diag_cols: ");
        // printVec<int>(nnodes, h_diag_cols);
        // printf("\n\n");

        // auto h_diag_bsr_data =
        //     BsrData(nnodes, 6, diag_inv_nnzb, h_diag_rowp, h_diag_cols, nullptr, nullptr, true);
        // auto d_diag_bsr_data = h_diag_bsr_data.createDeviceBsrData();
        delete[] h_diag_rowp;
        delete[] h_diag_cols;

        // now allocate DeviceVec for the values
        int ndiag_vals = block_dim * block_dim * nnodes;
        auto d_diag_vals = DeviceVec<T>(ndiag_vals);

        // printf("2: h_kmat_rowp: ");
        // printVec<int>(nnodes + 1, h_kmat_rowp);
        // printf("h_kmat_cols: ");
        // printVec<int>(min(100, kmat_nnzb), h_kmat_cols);

        int *h_kmat_diagp = new int[nnodes];
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
                int block_col = h_kmat_cols[jp];
                // printf("row %d, col %d\n", block_row, block_col);
                if (block_row == block_col) {
                    h_kmat_diagp[block_row] = jp;
                }
            }
        }

        // DEBUG
        // printf("step 2\n");
        // printf("-----\n");
        // printf("h_kmat_diagp: ");
        // printVec<int>(nnodes, h_kmat_diagp);
        // return;

        int *d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

        // call the kernel to copy out diag vals first
        dim3 block(32);
        int nblocks = (ndiag_vals + 31) / 32;
        dim3 grid(nblocks);
        k_copyBlockDiagFromBsrMat<T>
            <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_vals.getPtr());

        // DEBUG
        // printf("step 3\n");
        // printf("-----\n");
        // T *h_kmat_vals = Kmat.getVec().createHostVec().getPtr();
        // for (int block_row = 0; block_row < 3; block_row++) {
        //     for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
        //         int block_col = h_kmat_cols[jp];
        //         if (block_row == block_col) {
        //             printf("h_kmat[%d,%d] : ", block_row, block_col);
        //             printVec<T>(36, &h_kmat_vals[36 * jp]);
        //         }
        //     }
        // }
        // printf("h_kmat_vals: ");
        // printVec<T>(100, h_kmat_vals);
        // T *h_diag_vals1 = d_diag_vals.createHostVec().getPtr();
        // for (int inode = 0; inode < 3; inode++) {
        //     printf("h_diag_vals[%d]: ", inode);
        //     printVec<T>(36, &h_diag_vals1[36 * inode]);
        // }
        // return;

        delete[] h_kmat_rowp;
        delete[] h_kmat_cols;

        // bool use_cublas = true;  // getriBatched very slow for in-place, LU batched not working
        // rn
        bool use_cublas = false;

        // auto d_d = DeviceVec<T>(ndiag_vals);
        // d_diag_inv_vals = d_diag_inv_vals_vec.getPtr();

        // copy original D for DEBUG  (since LU decomp messes up comparison..)
        // auto d_diag_vals_copy = DeviceVec<T>(ndiag_vals);
        // cudaMemcpy(d_diag_vals_copy.getPtr(), d_diag_vals.getPtr(), ndiag_vals * sizeof(T),
        //            cudaMemcpyDeviceToDevice);

        if (use_cublas) {
            // make ptr-ptr objects.. that point to the D and Dinv single ptrs
            d_diag_LU_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            nblocks = (nnodes + 31) / 32;
            dim3 grid2(nblocks);
            k_singleToDoublePointer<T>
                <<<grid2, block>>>(nnodes, block_dim, d_diag_vals.getPtr(), d_diag_LU_batch_ptr);

            d_temp_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            k_singleToDoublePointerVec<T>
                <<<grid2, block>>>(nnodes, block_dim, d_temp, d_temp_batch_ptr);

            // T **d_diag_inv_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            // k_singleToDoublePointer<T>
            //     <<<grid2, block>>>(nnodes, block_dim, d_diag_inv_vals, d_diag_inv_batch_ptr);

            // // get row scaling..
            // T *d_diag_scales = DeviceVec<T>(N).getPtr();
            // // divide D by the row scales
            // nblocks = (N + 31) / 32;
            // dim3 grid3(nblocks);
            // k_computeDiagRowScales<T>
            //     <<<grid3, block>>>(nnodes, block_dim, d_diag_vals.getPtr(), d_diag_scales);

            // // then we'll do local 6x6 inverses D => Dinv of the block diag matrix into
            // // d_diag_inv_vals
            // cudaMemcpy(d_diag_inv_vals, d_diag_vals.getPtr(), ndiag_vals * sizeof(T),
            //            cudaMemcpyDeviceToDevice);

            // now use cublas to do diag inv in batch (on diag), other option is to use cusparse
            // if this is slow.. first we do in-place LU decomp,
            // https://docs.nvidia.com/cuda/cublas/ first an in-place LU decomp P*A = L*U (with
            // pivots on each 6x6 nodal block) in-place on d_diag_vals
            d_piv = DeviceVec<int>(nnodes * block_dim).getPtr();
            d_info = DeviceVec<int>(nnodes).getPtr();
            cublasDgetrfBatched(cublasHandle, block_dim, d_diag_LU_batch_ptr, block_dim, d_piv,
                                d_info, nnodes);  // LU decomp in place here..

            // really singular block diag => need to use LU batched can't get accurate full D^-1
            // directly

            // then do an inversion from d_diag_vals LU decomp => d_diag_inv_vals ptr
            // get ri batched is really inaccurate..
            // this is really inaccurate on first call especially if matrices ill-conditioned, often
            // are.. NOTE : could do newton-schulz refinement for matrix inversion:
            //    X_{k+1} = X_k * (2I - D * X_k)
            // OR like here I'm just going to set D = S * A where S is scaling 6x6 diag matrix
            // and A has ones on diag, so scaled to O(1), thus each row is normalized by diag(S)
            // cublasDgetriBatched(cublasHandle, block_dim, d_diag_batch_ptr, block_dim, d_piv,
            //                     d_diag_inv_batch_ptr, block_dim, d_info, nnodes);

            // // undo the row scalings..
            // k_reapplyDiagRowScales<T><<<grid3, block>>>(nnodes, block_dim, d_diag_scales,
            //                                             d_diag_vals.getPtr(), d_diag_inv_vals);

        }  // end of cublas

        if (!use_cublas) {
            // use cusparse to get Dinv as LU factor and then we just do triang solves on
            // block - diag perform_ilu0_factorization();
            d_diag_LU_vals = d_diag_vals.getPtr();  // just copy these pointers..
            // printf("performing ILU(0) factor\n");

            CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                 trans_U, policy_L, policy_U, dir);
            // printf("did ILU(0) factor on block-diag D(K)\n");
        }

        // // DEBUG: check the 6x6 diag and diag inv matrices..
        // T *h_diag_vals = d_diag_vals.createHostVec().getPtr();
        // // printf("step 4\n");
        // for (int inode = 0; inode < 3; inode++) {
        //     printf("node %d D vs Dinv\n", inode);

        //     for (int icol = 0; icol < 6; icol++) {
        //         cudaMemset(d_temp, 0.0, N * sizeof(T));
        //         k_setSingleVal<<<1, 1>>>(N, 6 * inode + icol, 1.0, d_temp);

        //         printf("\tD[:,%d]: ", icol);
        //         // sym so actually row printout here
        //         printVec<T>(6, &h_diag_vals[36 * inode + 6 * icol]);

        //         // test the Dinv using getrsbatched on unit vecs
        //         cublasDgetrsBatched(cublasHandle, CUBLAS_OP_N, block_dim, 1, d_diag_LU_batch_ptr,
        //                             block_dim, d_piv, d_temp_batch_ptr, block_dim, d_info,
        //                             nnodes);

        //         T *h_temp = new T[6];
        //         cudaMemcpy(h_temp, &d_temp[6 * inode], 6 * sizeof(T), cudaMemcpyDeviceToHost);

        //         printf("\tLU=>Dinv[:,%d]: ", icol);
        //         // sym so actually row printout here
        //         printVec<T>(6, h_temp);
        //     }
        // }

        // test t

        // printf("here1\n");
        // d_diag_vals.free();
        // // delete[] d_diag_batch_ptr;
        // // delete[] d_diag_inv_batch_ptr;
        // printf("here2\n");

        // and make a BsrMat for it..
        D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_diag_vals);
        // printf("here3\n");
    }

    void buildColorLocalRowPointers() {
        // build local row pointers for row-slicing by color (of Kmat)
        // int *h_color_vals_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

        // init the color pointers
        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();  // says which rows in d_kmat_rowp are each color
        h_color_bnz_ptr =
            new int[num_colors + 1];  // says which block nz bounds for each color in cols, Kmat
        h_color_local_rowp_ptr =
            new int[num_colors + 1];  // pointer for bounds of d_color_local_rowps
        int *h_color_local_rowps = new int[nnodes + num_colors];

        // copy kmat pointers to host
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // build each pointer..
        h_color_bnz_ptr[0] = 0;
        h_color_local_rowp_ptr[0] = 0;
        int offset = 0;
        for (int icolor = 0; icolor < num_colors; icolor++) {
            int brow_start = color_rowp[icolor], brow_end = color_rowp[icolor + 1];
            int bnz_start = h_kmat_rowp[brow_start], bnz_end = h_kmat_rowp[brow_end];

            int nnzb_color = bnz_end - bnz_start;
            h_color_bnz_ptr[icolor + 1] = h_color_bnz_ptr[icolor] + nnzb_color;

            // now set the local rowp arrays for this color
            int nbrows_color = brow_end - brow_start;
            h_color_local_rowp_ptr[icolor + 1] = h_color_local_rowp_ptr[icolor] + nbrows_color + 1;
            h_color_local_rowps[offset] = 0;
            for (int local_row = 0; local_row < nbrows_color; local_row++) {
                int row_diff =
                    h_kmat_rowp[brow_start + local_row + 1] - h_kmat_rowp[brow_start + local_row];
                h_color_local_rowps[local_row + 1 + offset] =
                    h_color_local_rowps[local_row + offset] + row_diff;
            }
            offset += nbrows_color + 1;
        }

        // DEBUG:
        // printf("nnodes %d, N %d, Kmat nnzb %d\n", nnodes, N, kmat_nnzb);
        // printf("h_kmat_rowp: ");
        // printVec<int>(nnodes + 1, h_kmat_rowp);
        // printf("h_color_rowp (prev): ");
        // printVec<int>(num_colors + 1, color_rowp);
        // printf("h_kmat_cols: ");
        // printVec<int>(min(100, kmat_nnzb), h_kmat_cols);
        // printf("\n----\nnew color ptrs\n-----\n");
        // printf("h_color_bnz_ptr: ");
        // printVec<int>(num_colors + 1, h_color_bnz_ptr);
        // printf("h_color_local_rowp_ptr: ");
        // printVec<int>(num_colors + 1, h_color_local_rowp_ptr);
        // printf("h_color_local_rowps: ");
        // printVec<int>(nnodes + num_colors, h_color_local_rowps);

        delete[] h_kmat_rowp;
        delete[] h_kmat_cols;

        d_color_local_rowps =
            HostVec<int>(nnodes + num_colors, h_color_local_rowps).createDeviceVec().getPtr();
    }

    void multicolorBlockGaussSeidel(int n_iters, bool print = false, int print_freq = 10) {
        // do multicolor BSRmat block gauss-seidel on the defect

        T init_defect_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();

        for (int iter = 0; iter < n_iters; iter++) {
            for (int icolor = 0; icolor < num_colors; icolor++) {
                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int block_dim2 = block_dim * block_dim;  // 36

                // printf("iter %d, color %d : block rows [%d,%d)\n", iter, icolor, start, end);

                // 1) compute Dinv_c * defect_c => dx_c  (c indicates color subset)
                int color_Dinv_nnzb = nblock_rows_color;
                // can use same rowp, cols here (0,1,...,nrows)
                int *d_color_diag_cols = &d_diag_cols[block_dim * start];
                // T *d_Dinv_vals_color = &d_diag_inv_vals[block_dim2 * start];
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                T *d_temp_color2 = &d_temp2[block_dim * start];

                T a = 1.0, b = 0.0;
                // CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                //                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                //                               nblock_rows_color, nblock_rows_color,
                //                               color_Dinv_nnzb, &a, descrDinvMat,
                //                               d_Dinv_vals_color, d_diag_rowp, d_color_diag_cols,
                //                               block_dim, d_defect_color, &b, d_temp_color));

                // couldn't get accurate block-diag inv directly, so doing block diag LU here.. TBD
                // or I'll come back to that..
                // cudaMemcpy(d_temp_color, d_defect_color, nblock_rows_color * block_dim *
                // sizeof(T),
                //            cudaMemcpyDeviceToDevice);
                // T **d_temp_batch_color = &d_temp_batch_ptr[start];
                // T **d_diag_LU_batch_color = &d_diag_LU_batch_ptr[start];
                // cublasDgetrsBatched(cublasHandle, CUBLAS_OP_N, block_dim, 1,
                // d_diag_LU_batch_color,
                //                     block_dim, d_piv, d_temp_batch_color, block_dim, d_info,
                //                     nnodes);

                // T defect_nrm_0;
                // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm_0));
                // printf("defect nrm0 %.2e\n", defect_nrm_0);

                const double alpha = 1.0;
                // try normal triang solve first.. DEBUG
                // printf("try normal triang solve first (DEBUG), get rid of this\n");
                // CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                //                                      diag_inv_nnzb, &alpha, descr_U,
                //                                      d_diag_LU_vals, d_diag_rowp, d_diag_cols,
                //                                      block_dim, info_U, d_defect.getPtr(),
                //                                      d_temp, policy_U, pBuffer));
                // printf("here0\n");

                // cusparse LU solve on block diag matrix D(K) : giving D^-1 * d_temp

                T *d_diag_LU_vals_color = &d_diag_LU_vals[block_dim2 * start];
                // // NOTE : I don't think I need this call.. since this is block U part
                // // // and agbove diagonal and this is only diagonal here
                // // // CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                // // //     cusparseHandle, dir, trans_L, nblock_rows_color, color_Dinv_nnzb,
                // &alpha,
                // // //     descr_L, d_diag_LU_vals_color, d_diag_rowp, d_color_diag_cols,
                // block_dim,
                // // //     info_L, d_defect_color, d_temp_color, policy_L, pBuffer));
                // // // printf("here1\n");

                // T defect_nrm_1;
                // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm_1));
                // printf("defect nrm1 %.2e\n", defect_nrm_1);

                // triangular solve U*y = z,
                // only need this call with block diag I think..
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, nblock_rows_color, color_Dinv_nnzb, &alpha,
                    descr_U, d_diag_LU_vals_color, d_diag_rowp, d_color_diag_cols, block_dim,
                    info_U, d_defect_color, d_temp_color, policy_U, pBuffer));
                // printf("here2, nnodes = %d\n", nnodes);

                // T defect_nrm_2;
                // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm_2));
                // printf("defect nrm2 %.2e\n", defect_nrm_2);

                // 2) update soln x_color += dx_color
                int nrows_color = nblock_rows_color * block_dim;
                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = 1.0;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));
                printf("here3\n");

                // 3) update defect, defect -= K[color,:]^T * dx_color, with KT_color = N x
                // nrows_color matrix
                int kmat_bnz_start = h_color_bnz_ptr[icolor];
                int kmat_bnz_end = h_color_bnz_ptr[icolor + 1];
                int color_kmat_nnzb = kmat_bnz_end - kmat_bnz_start;
                T *d_color_kmat_vals = &d_kmat_vals[36 * kmat_bnz_start];
                int local_color_rowp_start = h_color_local_rowp_ptr[icolor];
                printf("kmat bnz %d to %d, local rowp start %d\n", kmat_bnz_start, kmat_bnz_end,
                       local_color_rowp_start);
                int *d_kmat_color_local_rowp = &d_color_local_rowps[local_color_rowp_start];
                int *d_kmat_color_cols = &d_kmat_cols[kmat_bnz_start];
                a = -1.0,
                b = 1.0;  // so that defect := defect - mat*vec
                // CHECK_CUSPARSE(cusparseDbsrmv(
                //     cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_TRANSPOSE,
                //     nblock_rows_color, nnodes, color_kmat_nnzb, &a, descrKmat, d_color_kmat_vals,
                //     d_kmat_color_local_rowp, d_kmat_color_cols, block_dim, d_temp_color, &b,
                //     d_defect.getPtr()));

                // DEBUG try this instead..
                // printf(
                //     "try Sparse MV debug, remove this later.. and just do row-sliced transpose "
                //     "version\n");
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));
                // printf("here4\n");

            }  // next color iteration

            printf("iter %d, done with color iterations\n", iter);

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    // data
    int N;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_rhs, d_defect, d_soln;
    T *d_temp, *d_temp2, *d_resid;
    HostVec<int> h_color_rowp;

   private:  // private data for cusparse and cublas
    // private data
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // color rowp and nnzb pointers data for row-slicing
    int *h_color_bnz_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

    // for general matrices..
    int block_dim, nnodes;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    T *d_diag_vals, *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
};
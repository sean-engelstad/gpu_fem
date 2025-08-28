// geom multigrid for the shells
#pragma once
#include <cusparse_v2.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/bsr_mat.h"
#include "solvers/linear_static/bsr_direct_LU.h"

// local includes for shell multigrid
#include "grid.cuh"
#include "prolongation/_prolong.h"  // for prolongations
#include "vtk.h"

enum SMOOTHER : short {
    MULTICOLOR_GS,
    LEXIGRAPHIC_GS,
};

template <class Assembler, class Prolongation, SMOOTHER smoother>
class ShellGrid {
   public:
    using T = double;
    using I = long long int;

    ShellGrid() = default;

    ShellGrid(Assembler &assembler_, int N_, BsrMat<DeviceVec<T>> Kmat_, DeviceVec<T> d_rhs_,
              HostVec<int> h_color_rowp_, bool full_LU_ = false)
        : N(N_), full_LU(full_LU_) {
        Kmat = Kmat_;
        d_rhs = d_rhs_;
        h_color_rowp = h_color_rowp_;
        block_dim = 6;
        nnodes = N / 6;

        assembler = assembler_;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        // init helper methods
        if (smoother == MULTICOLOR_GS) {
            buildColorLocalRowPointers();
        }
        initCuda();
        if (smoother == MULTICOLOR_GS) {
            buildDiagInvMat();
        }
        if (smoother == LEXIGRAPHIC_GS && !full_LU) {
            initLowerMatForGaussSeidel();
        }
    }

    static ShellGrid *buildFromAssembler(Assembler &assembler, T *h_loads, bool full_LU = false,
                                         bool reorder = true) {
        // only do full LU factor on coarsest grid..

        // BSR symbolic factorization
        // must pass by ref to not corrupt pointers
        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;
        if (smoother == MULTICOLOR_GS) {
            if (reorder) {
                bsr_data.multicolor_reordering(num_colors,
                                               _color_rowp);  // TODO : add this method.. (I guess I
                                                              // can just do host for now..)
            } else {
                num_colors = 1;
                _color_rowp = new int[2];
                _color_rowp[0] = 0;
                _color_rowp[1] = assembler.get_num_nodes();
            }
        } else if (smoother == LEXIGRAPHIC_GS) {
            if (reorder) {
                bsr_data.RCM_reordering(1);
            }
            // default or no colors..
            num_colors = 1;
            _color_rowp = new int[2];
            _color_rowp[0] = 0;
            _color_rowp[1] = assembler.get_num_nodes();
        }
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);

        if (full_LU) {
            // only do this on coarsest grid (full LU fillin pattern for direct solve)
            bsr_data.compute_full_LU_pattern(10.0, false);
        } else {
            bsr_data.compute_nofill_pattern();
        }

        assembler.moveBsrDataToDevice();
        auto loads = assembler.createVarsVec(h_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto soln = assembler.createVarsVec();
        auto res = assembler.createVarsVec();
        auto vars = assembler.createVarsVec();
        int N = vars.getSize();

        // assemble the kmat
        assembler.add_jacobian(res, kmat);
        assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);

        return new ShellGrid(assembler, N, kmat, loads, h_color_rowp, full_LU);
    }

    void initCuda() {
        // init handles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_defect = DeviceVec<T>(N);
        d_soln = DeviceVec<T>(N);
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_weights = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();
        d_int_temp = DeviceVec<int>(N).getPtr();

        // copy rhs into defect
        cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        // and inv permute the rhs
        d_perm = Kmat.getPerm();
        d_iperm = Kmat.getIPerm();
        auto d_bsr_data = Kmat.getBsrData();
        d_elem_conn = d_bsr_data.elem_conn;
        nelems = d_bsr_data.nelems;
        d_defect.permuteData(block_dim, d_iperm);

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

        // also init kmat for direct LU solves..
        if (full_LU) {
            d_kmat_lu_vals = DeviceVec<T>(Kmat.get_nnz()).getPtr();
            CHECK_CUDA(cudaMemcpy(d_kmat_lu_vals, d_kmat_vals, Kmat.get_nnz() * sizeof(T),
                                  cudaMemcpyDeviceToDevice));

            // ILU(0) factor on full LU pattern
            CUSPARSE::perform_ilu0_factorization(
                cusparseHandle, descr_kmat_L, descr_kmat_U, info_kmat_L, info_kmat_U, &kmat_pBuffer,
                nnodes, kmat_nnzb, block_dim, d_kmat_lu_vals, d_kmat_rowp, d_kmat_cols, trans_L,
                trans_U, policy_L, policy_U, dir);
        }
    }

    void initLowerMatForGaussSeidel() { /* init L+D matrix for lexigraphic or RCM Gauss-seidel */

        // init kmat descriptor for L+D matrix (no ilu0 factor, this is just the matrix itself
        // nofill)
        cusparseCreateMatDescr(&descr_kmat_L);
        cusparseSetMatIndexBase(descr_kmat_L, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr_kmat_L, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatFillMode(descr_kmat_L, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr_kmat_L, CUSPARSE_DIAG_TYPE_NON_UNIT);  // includes diag here..
        cusparseCreateBsrsv2Info(&info_kmat_L);

        // get buffer size
        int pbufferSize;
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
            cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, descr_kmat_L, d_kmat_vals, d_kmat_rowp,
            d_kmat_cols, block_dim, info_kmat_L, &pbufferSize));
        cudaMalloc(&kmat_pBuffer, pbufferSize);

        // compute symbolic analysis for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, nnodes, kmat_nnzb,
                                                descr_kmat_L, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                                block_dim, info_kmat_L, policy_L, kmat_pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
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

        // on host, get the pointer locations in Kmat of the block diag entries..
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // printf("h_color_rowp:");
        // printVec<int>(h_color_rowp.getSize() + 1, h_color_rowp.getPtr());
        // printf("nnodes %d\n", nnodes);
        // printf("h_kmat_rowp:");
        // printVec<int>(20, h_kmat_rowp);
        // printf("h_kmat_cols:");
        // printVec<int>(100, h_kmat_cols);

        // now copy to device
        d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
        d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

        // create the bsr data object on device
        auto d_diag_bsr_data =
            BsrData(nnodes, 6, diag_inv_nnzb, d_diag_rowp, d_diag_cols, nullptr, nullptr, false);
        delete[] h_diag_rowp;
        delete[] h_diag_cols;

        // now allocate DeviceVec for the values
        int ndiag_vals = block_dim * block_dim * nnodes;
        auto d_diag_vals = DeviceVec<T>(ndiag_vals);

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

        int *d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

        // call the kernel to copy out diag vals first
        dim3 block(32);
        int nblocks = (ndiag_vals + 31) / 32;
        dim3 grid(nblocks);
        k_copyBlockDiagFromBsrMat<T>
            <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_vals.getPtr());
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

        delete[] h_kmat_rowp;
        delete[] h_kmat_cols;

        d_color_local_rowps =
            HostVec<int>(nnodes + num_colors, h_color_local_rowps).createDeviceVec().getPtr();
    }

    void direct_solve(bool print = false) {
        // T defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
        // printf("\tdirect solve, ||defect|| = %.4e\n", defect_nrm);

        // do a direct solve on the coarsest grid, with current defect.. (multicolor GS not
        // reliable) we keep permuted form, so I have to undo that from direct solve..
        // bool permute_inout = false;
        // CUSPARSE::direct_LU_solve<T>(Kmat, d_defect, d_soln, print, permute_inout);
        // this routine destroys and recreates handles .. may cause problems if calling multiple
        // times I think

        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, nnodes, kmat_nnzb,
                                             &alpha, descr_kmat_L, d_kmat_lu_vals, d_kmat_rowp,
                                             d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(),
                                             d_temp, policy_L, kmat_pBuffer));

        // triangular solve U*y = z
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes, kmat_nnzb,
                                             &alpha, descr_kmat_U, d_kmat_lu_vals, d_kmat_rowp,
                                             d_kmat_cols, block_dim, info_kmat_U, d_temp,
                                             d_soln.getPtr(), policy_U, kmat_pBuffer));

        // T soln_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_soln.getPtr(), 1, &soln_nrm));
        // printf("\tdirect solve, ||soln|| = %.4e\n", soln_nrm);
    }

    T getDefectNorm() {
        T def_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &def_nrm));
        return def_nrm;
    }

    void smoothDefect(int n_iters, bool print = false, int print_freq = 10, T omega = 1.0,
                      bool rev_colors = false) {
        /* calls either multicolor smoother or lexigraphic GS depending on tempalte smoother type */
        if (smoother == MULTICOLOR_GS) {
            multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega, rev_colors);
        } else if (smoother == LEXIGRAPHIC_GS) {
            lexigraphicBlockGS(n_iters, print, print_freq);
        }
    }

    void multicolorBlockGaussSeidel_slow(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool rev_colors = false) {
        // slower version of do multicolor BSRmat block gauss-seidel on the defect
        // slower in the sense that it uses full mat-vec and full triang solves (does work right)
        // would like a faster version with color slicing next..

        // T init_defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        // if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();

        // DEBUG
        // n_solns = n_iters * num_colors;
        // h_solns = new T *[n_solns];

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                // printf("\t\titer %d, color %d\n", iter, icolor);

                int _icolor2 = (_icolor + iter) % 4;  // permute order as you go
                // int _icolor2 = _icolor;  // no permutations.. about the same either way..

                int icolor = rev_colors ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                T *d_temp_color2 = &d_temp2[block_dim * start];
                cudaMemset(d_temp2, 0.0, N * sizeof(T));  // DEBUG
                cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * block_dim * sizeof(T),
                           cudaMemcpyDeviceToDevice);

                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, &alpha, descr_L,
                    d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2, d_resid,
                    policy_L, pBuffer));  // prob only need U^-1 part for block diag.. TBD
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                                     diag_inv_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                                     d_diag_rowp, d_diag_cols, block_dim, info_U,
                                                     d_resid, d_temp, policy_U, pBuffer));

                // 2) update soln x_color += dx_color
                int nrows_color = nblock_rows_color * block_dim;
                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega,
                b = 1.0;  // so that defect := defect - mat*vec
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

            }  // next color iteration

            // printf("iter %d, done with color iterations\n", iter);

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    void lexigraphicBlockGS(int n_iters, bool print = false, int print_freq = 10) {
        // this is lexigraphic or RCM GS (RCM if more general mesh..)

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        T a, b;

        for (int iter = 0; iter < n_iters; iter++) {
            // 1) (L+D)*dx = defect with triang solve
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, &alpha, descr_kmat_L, d_kmat_vals,
                d_kmat_rowp, d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(), d_temp,
                policy_L, kmat_pBuffer));  // prob only need U^-1 part for block diag.. TBD

            // 2) update d_soln += d_temp (aka dx)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            // 3) compute new defect = prev_defect - A * dx
            a = -1.0,
            b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    // void multicolorBlockGaussSeidel_fast(int n_iters, bool print = false, int print_freq = 10) {
    //     // do multicolor BSRmat block gauss-seidel on the defect
    //     // work in progress here for faster or more scalable version with color slicing..
    //     // and hopefully block_diag_inv mat-vec products

    //     T init_defect_nrm;
    //     CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
    //     if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

    //     int num_colors = h_color_rowp.getSize() - 1;
    //     int *color_rowp = h_color_rowp.getPtr();

    //     // DEBUG
    //     n_solns = n_iters * num_colors;
    //     h_solns = new T *[n_solns];

    //     for (int iter = 0; iter < n_iters; iter++) {
    //         for (int icolor = 0; icolor < num_colors; icolor++) {
    //             // get active rows / cols for this color
    //             int start = color_rowp[icolor], end = color_rowp[icolor + 1];
    //             int nblock_rows_color = end - start;
    //             // int block_dim2 = block_dim * block_dim;  // 36

    //             // printf("iter %d, color %d : block rows [%d,%d)\n", iter, icolor, start, end);

    //             // 1) compute Dinv_c * defect_c => dx_c  (c indicates color subset)
    //             // int color_Dinv_nnzb = nblock_rows_color;
    //             // can use same rowp, cols here (0,1,...,nrows)
    //             // int *d_color_diag_cols = &d_diag_cols[block_dim * start];
    //             // T *d_Dinv_vals_color = &d_diag_inv_vals[block_dim2 * start];
    //             T *d_defect_color = &d_defect.getPtr()[block_dim * start];
    //             cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
    //             T *d_temp_color = &d_temp[block_dim * start];
    //             T *d_temp_color2 = &d_temp2[block_dim * start];
    //             cudaMemset(d_temp2, 0.0, N * sizeof(T));  // DEBUG
    //             cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * block_dim *
    //             sizeof(T),
    //                        cudaMemcpyDeviceToDevice);

    //             T a = 1.0, b = 0.0;
    //             // CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
    //             //                               CUSPARSE_OPERATION_NON_TRANSPOSE,
    //             //                               nblock_rows_color, nblock_rows_color,
    //             //                               color_Dinv_nnzb, &a, descrDinvMat,
    //             //                               d_Dinv_vals_color, d_diag_rowp,
    //             d_color_diag_cols,
    //             //                               block_dim, d_defect_color, &b, d_temp_color));

    //             // couldn't get accurate block-diag inv directly, so doing block diag LU here..
    //             TBD
    //             // or I'll come back to that..
    //             // cudaMemcpy(d_temp_color, d_defect_color, nblock_rows_color * block_dim *
    //             // sizeof(T),
    //             //            cudaMemcpyDeviceToDevice);
    //             // T **d_temp_batch_color = &d_temp_batch_ptr[start];
    //             // T **d_diag_LU_batch_color = &d_diag_LU_batch_ptr[start];
    //             // cublasDgetrsBatched(cublasHandle, CUBLAS_OP_N, block_dim, 1,
    //             // d_diag_LU_batch_color,
    //             //                     block_dim, d_piv, d_temp_batch_color, block_dim, d_info,
    //             //                     nnodes);

    //             // T defect_nrm_0;
    //             // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1,
    //             &defect_nrm_0));
    //             // printf("defect nrm0 %.2e\n", defect_nrm_0);

    //             const double alpha = 1.0;
    //             // try normal triang solve first.. DEBUG
    //             // printf("try normal triang solve first (DEBUG), get rid of this\n");
    //             CHECK_CUSPARSE(cusparseDbsrsv2_solve(
    //                 cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, &alpha, descr_L,
    //                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2,
    //                 d_resid, policy_L, pBuffer));  // prob only need U^-1 part for block diag..
    //                 TBD
    //             CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
    //                                                  diag_inv_nnzb, &alpha, descr_U,
    //                                                  d_diag_LU_vals, d_diag_rowp, d_diag_cols,
    //                                                  block_dim, info_U, d_resid, d_temp,
    //                                                  policy_U, pBuffer));
    //             // printf("here0\n");

    //             // cusparse LU solve on block diag matrix D(K) : giving D^-1 * d_temp

    //             // T *d_diag_LU_vals_color = &d_diag_LU_vals[block_dim2 * start];
    //             // // NOTE : I don't think I need this call.. since this is block U part
    //             // // // and agbove diagonal and this is only diagonal here
    //             // // // CHECK_CUSPARSE(cusparseDbsrsv2_solve(
    //             // // //     cusparseHandle, dir, trans_L, nblock_rows_color, color_Dinv_nnzb,
    //             // &alpha,
    //             // // //     descr_L, d_diag_LU_vals_color, d_diag_rowp, d_color_diag_cols,
    //             // block_dim,
    //             // // //     info_L, d_defect_color, d_temp_color, policy_L, pBuffer));
    //             // // // printf("here1\n");

    //             // T defect_nrm_1;
    //             // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1,
    //             &defect_nrm_1));
    //             // printf("defect nrm1 %.2e\n", defect_nrm_1);

    //             // triangular solve U*y = z,
    //             // only need this call with block diag I think..
    //             // CHECK_CUSPARSE(cusparseDbsrsv2_solve(
    //             //     cusparseHandle, dir, trans_U, nblock_rows_color, color_Dinv_nnzb, &alpha,
    //             //     descr_U, d_diag_LU_vals_color, d_diag_rowp, d_color_diag_cols, block_dim,
    //             //     info_U, d_defect_color, d_temp_color, policy_U, pBuffer));
    //             // printf("here2, nnodes = %d\n", nnodes);

    //             // T defect_nrm_2;
    //             // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1,
    //             &defect_nrm_2));
    //             // printf("defect nrm2 %.2e\n", defect_nrm_2);

    //             // 2) update soln x_color += dx_color
    //             int nrows_color = nblock_rows_color * block_dim;
    //             T *d_soln_color = &d_soln.getPtr()[block_dim * start];
    //             a = 1.0;
    //             CHECK_CUBLAS(
    //                 cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color,
    //                 1));
    //             printf("here3\n");

    //             // 3) update defect, defect -= K[color,:]^T * dx_color, with KT_color = N x
    //             // nrows_color matrix
    //             // int kmat_bnz_start = h_color_bnz_ptr[icolor];
    //             // int kmat_bnz_end = h_color_bnz_ptr[icolor + 1];
    //             // int color_kmat_nnzb = kmat_bnz_end - kmat_bnz_start;
    //             // T *d_color_kmat_vals = &d_kmat_vals[36 * kmat_bnz_start];
    //             // int local_color_rowp_start = h_color_local_rowp_ptr[icolor];
    //             // printf("kmat bnz %d to %d, local rowp start %d\n", kmat_bnz_start,
    //             kmat_bnz_end,
    //             //    local_color_rowp_start);
    //             // int *d_kmat_color_local_rowp = &d_color_local_rowps[local_color_rowp_start];
    //             // int *d_kmat_color_cols = &d_kmat_cols[kmat_bnz_start];
    //             a = -1.0,
    //             b = 1.0;  // so that defect := defect - mat*vec
    //             // CHECK_CUSPARSE(cusparseDbsrmv(
    //             //     cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_TRANSPOSE,
    //             //     nblock_rows_color, nnodes, color_kmat_nnzb, &a, descrKmat,
    //             d_color_kmat_vals,
    //             //     d_kmat_color_local_rowp, d_kmat_color_cols, block_dim, d_temp_color, &b,
    //             //     d_defect.getPtr()));

    //             // DEBUG try this instead..
    //             // printf(
    //             //     "try Sparse MV debug, remove this later.. and just do row-sliced transpose
    //             "
    //             //     "version\n");
    //             CHECK_CUSPARSE(cusparseDbsrmv(
    //                 cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                 nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
    //                 d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));
    //             // printf("here4\n");

    //             // auto h_soln = d_soln.createHostVec(); // DEBUG
    //             // int i_soln = n_iters * num_colors + icolor;
    //             // h_solns[i_soln] = new T[N];
    //             // memcpy(h_solns[i_soln], h_soln.getPtr(), N * sizeof(T));

    //         }  // next color iteration

    //         printf("iter %d, done with color iterations\n", iter);

    //         /* report progress of defect nrm if printing.. */
    //         T defect_nrm;
    //         CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
    //         if (print && iter % print_freq == 0)
    //             printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

    //     }  // next block-GS iteration

    //     // NOTE : solution and defect are permuted here..
    // }  // end of multicolor block GS fast

    void prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in, bool debug = false,
                    std::string file_prefix = "", std::string file_suffix = "") {
        // prolongate from coarser grid to this fine grid
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        // T soln_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, coarse_soln_in.getSize(), coarse_soln_in.getPtr(),
        // 1,
        //                          &soln_nrm));
        // printf("\tprolong coarse soln in, ||soln|| = %.4e\n", soln_nrm);

        Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                 d_weights);
        CHECK_CUDA(cudaDeviceSynchronize());

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_temp_vec);
        d_temp_vec.permuteData(block_dim, d_iperm);

        // rescale coarse-fine using 1DOF min energy step
        // since FEA restrict and prolong operations are not energy minimally scaled
        // if u = u0 + omega * s, with s the proposed d_temp or du here (or line search)
        // then min energy omega from 1DOF galerkin is omega = <s, defect> / <s, Ks>
        // so need 2 dot prods, one SpMV, see 'multigrid/_python_demos/4_gmg_shell/1_mg.py' also
        T sT_defect;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_defect.getPtr(), 1, d_temp, 1, &sT_defect));

        T a = 1.0, b = 0.0;  // K * d_temp + 0 * d_temp2 => d_temp2
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));
        T sT_Ks;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));
        T omega = sT_defect / sT_Ks;
        if (debug) printf("omega = %.2e\n", omega);
        // printf("sT_defect %.2e, sT_Ks %.2e\n", sT_defect, sT_Ks);

        // now add coarse-fine dx into soln and update defect (with u = u0 + omega * d_temp)
        a = omega, b = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        a = -omega, b = 1.0;
        // a = -omega, b = 0.0;  // DEBUG
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_defect.getPtr()));

        // DEBUG : write out the cf update, defect update and before and after defects
        if (debug) {
            auto h_cf_update = d_temp_vec.createPermuteVec(6, d_perm).createHostVec();
            T xpts_shift[3] = {0.0, -1.5, 1.5};
            printToVTKDEBUG<Assembler, HostVec<T>>(
                assembler, h_cf_update, file_prefix + "post2_cf_soln" + file_suffix, xpts_shift);

            auto h_cf_loads = DeviceVec<T>(N, d_temp2).createPermuteVec(6, d_perm).createHostVec();
            T xpts_shift2[3] = {0.0, -1.5, 3.0};
            printToVTKDEBUG<Assembler, HostVec<T>>(
                assembler, h_cf_loads, file_prefix + "post3_cf_loads" + file_suffix, xpts_shift2);

            auto h_defect2 = d_defect.createPermuteVec(6, d_perm).createHostVec();
            T xpts_shift3[3] = {0.0, -1.5, 4.5};
            printToVTKDEBUG<Assembler, HostVec<T>>(
                assembler, h_defect2, file_prefix + "post4_cf_fin_defect" + file_suffix,
                xpts_shift3);
        }
    }

    void restrict_defect(int nelems_fine, int *d_iperm_fine, DeviceVec<T> fine_defect_in) {
        // transfer from finer mesh to this coarse mesh
        cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T));  // reset defect

        Prolongation::restrict_defect(nelems_fine, d_iperm, d_iperm_fine, fine_defect_in, d_defect,
                                      d_weights);

        // auto h_defect2 = d_defect.createPermuteVec(6, d_perm).createHostVec();
        // printToVTK<Assembler, HostVec<T>>(assembler, h_defect2, "out/4_fin_defect.vtk");
        // T defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
        // printf("\trestrict to coarse, ||defect|| = %.4e\n", defect_nrm);

        // apply bcs to the defect again (cause it will accumulate on the boundary by backprop)
        // apply bcs is on un-permuted data
        d_defect.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_defect);
        d_defect.permuteData(block_dim, d_iperm);

        // reset soln (with bcs zero here, TBD others later)
        cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T));
    }

    // data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_rhs, d_defect, d_soln, d_temp_vec;
    T *d_temp, *d_temp2, *d_resid, *d_weights;
    int *d_perm, *d_iperm;
    const int *d_elem_conn;
    HostVec<int> h_color_rowp;
    int *d_int_temp;

    // DEBUG
    int n_solns;
    T **h_solns;

    // turn off private during debugging
   private:  // private data for cusparse and cublas
    // private data
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // color rowp and nnzb pointers data for row-slicing
    int *h_color_bnz_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    T *d_diag_vals, *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // and simiarly for Kmat a few differences
    bool full_LU;  // full LU only for coarsest mesh
    cusparseMatDescr_t descr_kmat_L = 0, descr_kmat_U = 0;
    bsrsv2Info_t info_kmat_L = 0, info_kmat_U = 0;
    void *kmat_pBuffer = 0;
};
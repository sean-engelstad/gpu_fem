// geom multigrid for the shells
#pragma once
#include <cusparse_v2.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/bsr_mat.h"
#include "solvers/linear_static/bsr_direct_LU.h"

// local includes for shell multigrid
#include <chrono>
#include <set>

#include "grid.cuh"
#include "prolongation/_prolong.h"  // for prolongations
#include "prolongation/unstruct_utils.h"
#include "vtk.h"

// for unstructured multigrid

enum SMOOTHER : short {
    MULTICOLOR_GS,
    MULTICOLOR_GS_FAST,
    MULTICOLOR_GS_FAST2,
    MULTICOLOR_GS_FAST2_JUNCTION,
    DAMPED_JACOBI,
    LEXIGRAPHIC_GS,
};

enum SCALER : short {
    ZERO, // doesn't add directly into defect
    NONE,
    LINE_SEARCH,
    PCG,
};

template <class Assembler, class Prolongation, SMOOTHER smoother, SCALER scaler>
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
        const bool startup = true;
        initCuda<startup>();
        if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST ||
            smoother == MULTICOLOR_GS_FAST2 || smoother == MULTICOLOR_GS_FAST2_JUNCTION ||
            smoother == DAMPED_JACOBI) {
            buildDiagInvMat<startup>();
            if ((smoother == MULTICOLOR_GS_FAST2 || smoother == MULTICOLOR_GS_FAST2_JUNCTION) &&
                !full_LU)
                buildTransposeColorMatrices<startup>();
        }
        if (smoother == LEXIGRAPHIC_GS && !full_LU) {
            initLowerMatForGaussSeidel();
        }
    }

    void update_after_assembly() {
        // update dependent ILU and other matrices from new assembly
        const bool startup = false;  // false here, just assembly no new startup pieces
        initCuda<startup>();
        if constexpr (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST ||
                      smoother == MULTICOLOR_GS_FAST2 || smoother == MULTICOLOR_GS_FAST2_JUNCTION ||
                      smoother == DAMPED_JACOBI) {
            buildDiagInvMat<startup>();
            if ((smoother == MULTICOLOR_GS_FAST2 || smoother == MULTICOLOR_GS_FAST2_JUNCTION) &&
                !full_LU)
                buildTransposeColorMatrices<startup>();
        }
    }

    

    double get_memory_usage_mb() {
        // get memory usage for kmat in megabytes
        size_t kmat_nnz = block_dim * block_dim * kmat_nnzb;
        size_t bytes_per_double = sizeof(double);
        size_t mem_bytes = kmat_nnz * bytes_per_double;
        double mem_MB = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
        // printf("kmat nnz = %d, mem_bytes %d\n", kmat_nnz, mem_bytes);
        return mem_MB;
    }

    template <bool startup = true>
    void initCuda() {
        // init handles
        if constexpr (startup) {
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

            // get perm pointers
            d_perm = Kmat.getPerm();
            d_iperm = Kmat.getIPerm();
            auto d_bsr_data = Kmat.getBsrData();
            d_elem_conn = d_bsr_data.elem_conn;
            nelems = d_bsr_data.nelems;

            // copy rhs into defect
            d_rhs.permuteData(block_dim, d_iperm);  // permute rhs to permuted form
            cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
            // d_defect.permuteData(block_dim, d_iperm);

            // make mat handles for SpMV
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
            CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

            CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
            CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

            if (full_LU) {
                d_kmat_lu_vals = DeviceVec<T>(Kmat.get_nnz()).getPtr();
            }
        }

        // also init kmat for direct LU solves.. this is for re-assembly here
        if (full_LU) {
            CHECK_CUDA(cudaMemcpy(d_kmat_lu_vals, d_kmat_vals, Kmat.get_nnz() * sizeof(T),
                                  cudaMemcpyDeviceToDevice));

            // ILU(0) factor on full LU pattern
            CUSPARSE::perform_ilu0_factorization(
                cusparseHandle, descr_kmat_L, descr_kmat_U, info_kmat_L, info_kmat_U, &kmat_pBuffer,
                nnodes, kmat_nnzb, block_dim, d_kmat_lu_vals, d_kmat_rowp, d_kmat_cols, trans_L,
                trans_U, policy_L, policy_U, dir);
        }
    }

    T getDefectNorm() {
        T def_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &def_nrm));
        return def_nrm;
    }

    void zeroSolution() { cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T)); }
    void zeroDefect() { cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T)); }

    void setDefect(DeviceVec<T> new_defect, bool perm = true) {
        // set the defect on the finest grid
        new_defect.copyValuesTo(d_defect);
        if (perm) d_defect.permuteData(block_dim, d_iperm);  // unperm to permuted
    }

    void getDefect(DeviceVec<T> defect_out, bool perm = true) {
        // copy solution to another device vec outside this class
        d_defect.copyValuesTo(defect_out);
        if (perm) defect_out.permuteData(block_dim, d_perm);  // permuted to unperm order
    }

    void getSolution(DeviceVec<T> soln_out, bool perm = true) {
        // copy solution to another device vec outside this class
        d_soln.copyValuesTo(soln_out);
        if (perm) soln_out.permuteData(block_dim, d_perm);  // permuted to unperm order
    }

    T getResidNorm() {
        /* double check the linear system is actually solved */

        // copy rhs into the resid
        cudaMemcpy(d_resid, d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

        // subtract A * x where A = Kmat
        T a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_soln.getPtr(), &b, d_resid));

        // get resid nrm now
        T resid_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_nrm));
        return resid_nrm;
    }

    void smoothDefect(int n_iters, bool print = false, int print_freq = 10, T omega = 1.0,
                      bool symmetric = false) {
        /* calls either multicolor smoother or lexigraphic GS depending on tempalte smoother type */
        if (smoother == MULTICOLOR_GS) {
            multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega, symmetric);
        } else if (smoother == MULTICOLOR_GS_FAST) {
            multicolorBlockGaussSeidel_fast(n_iters, print, print_freq, omega, symmetric);
        } else if (smoother == MULTICOLOR_GS_FAST2 || smoother == MULTICOLOR_GS_FAST2_JUNCTION) {
            multicolorBlockGaussSeidel_fast2(n_iters, print, print_freq, omega, symmetric);
        } else if (smoother == DAMPED_JACOBI) {
            dampedJacobi(n_iters, print, print_freq, omega);
        } else if (smoother == LEXIGRAPHIC_GS) {
            lexigraphicBlockGS(n_iters, print, print_freq);
        }
    }

    

    void prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in) {
        // prolongate from coarser grid to this fine grid
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        if constexpr (Prolongation::structured) {
            Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                     d_weights);
        } else {
            if constexpr (Prolongation::assembly) {
                Prolongation::prolongate(cusparseHandle, descrP, P_mat, coarse_soln_in, d_temp_vec);
            } else {
                // slower version
                Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                         d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
            }
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_temp_vec);
        d_temp_vec.permuteData(block_dim, d_iperm);

        // compute rhs update
        T a = 1.0, b = 0.0;  // K * d_temp + 0 * d_temp2 => d_temp2
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));

        
        if constexpr (scaler == ZERO) {
            return; // no update to solution, just keep soln update in temp and use that in outer K-cycle
        }
        T omega;
        if constexpr (scaler == NONE) {
            // no rescaling
            omega = 1.0;
        } else if (scaler == LINE_SEARCH) {
            // one DOF, min energy scaling (one DOF line search)
            // so need 2 dot prods, one SpMV, see 'multigrid/_python_demos/4_gmg_shell/1_mg.py' also
            T sT_defect;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_defect.getPtr(), 1, d_temp, 1, &sT_defect));

            T sT_Ks;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));
            omega = sT_defect / sT_Ks;
        }

        // now add coarse-fine dx into soln and update defect (with u = u0 + omega * d_temp)
        a = omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        a = -omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_defect.getPtr(), 1));
    }

    void restrict_defect(int nelems_fine, int *d_iperm_fine, DeviceVec<T> fine_defect_in) {
        // transfer from finer mesh to this coarse mesh
        cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T));  // reset defect
        // printf("start restrict defect\n");

        if constexpr (Prolongation::structured) {
            Prolongation::restrict_defect(nelems_fine, d_iperm, d_iperm_fine, fine_defect_in,
                                          d_defect, d_weights);
        } else {
            if constexpr (Prolongation::assembly) {
                Prolongation::restrict_defect(cusparseHandle, restrict_descrPT, restrict_PT_mat,
                                              fine_defect_in, d_defect);
            } else {
                // slower restriction operator (not using sparse matrix-vec kernels)
                Prolongation::restrict_defect(restrict_nnodes_fine, d_elem_conn, restrict_d_n2e_ptr,
                                              restrict_d_n2e_elems, restrict_d_n2e_xis, d_iperm,
                                              d_iperm_fine, fine_defect_in, d_defect, d_weights);
            }
        }

        // apply bcs to the defect again (cause it will accumulate on the boundary by backprop)
        // apply bcs is on un-permuted data
        d_defect.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_defect);
        d_defect.permuteData(block_dim, d_iperm);

        // printf("end restrict defect\n");

        // reset soln (with bcs zero here, TBD others later)
        cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T));
    }

    void free() {
        // TBD::
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

    // unstruct prolong data at the finer mesh level (for prolong)
    // this is in someways describing the P matrix operator.. (could reformulate as that)
    int *d_n2e_ptr, *d_n2e_elems, *d_coarse_conn, n2e_nnz, ncoarse_elems;
    T *d_n2e_xis;
    cusparseMatDescr_t descrP = 0;
    BsrMat<DeviceVec<T>> P_mat;

    // unstruct prolong at the coarser mesh level (for restrict)
    int *restrict_d_n2e_ptr, *restrict_d_n2e_elems, restrict_nnodes_fine, restrict_n2e_nnz;
    T *restrict_d_n2e_xis;
    cusparseMatDescr_t restrict_descrPT = 0;
    BsrMat<DeviceVec<T>> restrict_PT_mat;

    // turn off private during debugging
    //    private:  // private data for cusparse and cublas
    // ----------------------------------------------------

    // private data
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // color rowp and nnzb pointers data for row-slicing
    int *h_color_submat_nnzb;
    int **d_color_submat_rowp, **d_color_submat_rows, **d_color_submat_cols;
    T **d_color_submat_vals;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    DeviceVec<T> d_diag_vals;
    T *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;
    bool build_lu_inv_operator;
    int *d_kmat_diagp;
    BsrData d_diag_bsr_data;
    DeviceVec<T> d_dinv_vals;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
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
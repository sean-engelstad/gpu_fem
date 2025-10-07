// geom multigrid for the shells
#pragma once
#include <cusparse_v2.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/bsr_mat.h"

// local includes for shell multigrid
#include <chrono>
#include <set>

enum SCALER : short {
    ZERO, // doesn't add directly into defect
    NONE,
    LINE_SEARCH,
    PCG,
};

template <class Assembler, class Prolongation, class Smoother, SCALER scaler>
class SingleGrid {
    /* single grid class for multigrid, that manages smoothing, prolong and soln + defect of a given level */

   public:
    using T = double;

    SingleGrid() = default;

    SingleGrid(Assembler &assembler_, Prolongation *prolongation_, Smoother *smoother_, 
        BsrMat<DeviceVec<T>> Kmat_, DeviceVec<T> d_rhs_) : assembler(assembler_), prolongation(prolongation_), smoother(smoother_),
        Kmat(Kmat_), d_rhs(d_rhs_) {
        
        N = assembler.get_num_vars();
        block_dim = 6;
        nnodes = N / 6;
        
        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        initCuda();
    }

    void update_after_assembly() {
        // update dependent ILU and other matrices from new assembly
        if (prolongation) prolongation->update_after_assembly();
        if (restriction) restriction->update_after_assembly();
        if (smoother) smoother->update_after_assembly();
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
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_defect = DeviceVec<T>(N);
        d_soln = DeviceVec<T>(N);
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();

        // get perm pointers
        d_perm = Kmat.getPerm();
        d_iperm = Kmat.getIPerm();
        auto d_bsr_data = Kmat.getBsrData();
        nelems = d_bsr_data.nelems;

        // copy rhs into defect
        d_rhs.permuteData(block_dim, d_iperm);  // permute rhs to permuted form
        cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        // d_defect.permuteData(block_dim, d_iperm);

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));
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

    void smoothDefect(int n_iters, bool print = false, int print_freq = 10) {
        /* call the smoother */
        smoother->smoothDefect(d_defect, d_soln, n_iters, print, print_freq);
    }

    void prolongate(DeviceVec<T> coarse_soln_in) {
        /* prolongate from a coarser grid to this fine grid */

        // call prolong and store dx_fine => d_temp
        cudaMemset(d_temp, 0.0, N * sizeof(T));
        prolongation->prolongate(coarse_soln_in, d_temp_vec);

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);
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

    void restrict_defect(DeviceVec<T> fine_defect_in) {
        /* transfer defect from a finer mesh to THIS coarse mesh */

        // zero this coarse defect + restrict the finer defect to this coarse grid
        cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T));  // reset defect
        restriction->restrict_defect(fine_defect_in, d_defect);

        // apply bcs to the defect again (cause it will accumulate on the boundary by backprop)
        // apply bcs is on un-permuted data
        d_defect.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_defect);
        d_defect.permuteData(block_dim, d_iperm);

        // reset soln (with bcs zero here, TBD others later)
        cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T));
    }

    void free() {
        // TBD::
    }

    // public data
    Smoother *smoother;
    Prolongation *prolongation;
    Prolongation *restriction;

    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    int *d_perm, *d_iperm;
    DeviceVec<T> d_rhs, d_defect, d_soln, d_temp_vec;
    BsrMat<DeviceVec<T>> Kmat;
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;

  private:
    T *d_temp, *d_temp2, *d_resid;

    // private data
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
};
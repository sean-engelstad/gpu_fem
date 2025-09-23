#pragma once

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

template <typename T, class GRID, class Precond>
class PCGSolver {
public:

    PCGSolver(GRID &grid) {

        // get matrix and init other temp data for PCG solve
        mat = grid.Kmat;
        soln = grid.d_soln;
        rhs = grid.d_rhs;

        auto &bsr_data = mat.getBsrData();
        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;
        block_dim = bsr_data.block_dim;
        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        iperm = bsr_data.iperm;
        d_vals = mat.getPtr();

        cublasHandle = grid.cublasHandle;
        cusparseHandle = grid.cusparseHandle;
        
        N = soln.getSize();
        d_rhs = rhs.getPtr();
        d_x = soln.getPtr(); // soln starts out at zero

        // description of the K matrix
        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // make temp vecs
        d_tmp = DeviceVec<T>(N).getPtr();
        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();
        d_p = DeviceVec<T>(N).getPtr();
        d_w = DeviceVec<T>(N).getPtr();
        d_z = DeviceVec<T>(N).getPtr();
    }

    void solve() {
        
    }


    // main matrix and linear system data
    BsrMat<DeviceVec<T>> mat;
    DeviceVec<T> soln, rhs, d_resid;
    int N, mb, nb, nnzb, block_dim;
    int *d_rowp, *d_cols, *iperm;
    T *d_vals;

    // cusparse and cublas handles
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

    // description of K matrix
    cusparseMatDescr_t descrK;

    // temp vecs for PCG algorithm
    T *d_tmp, *d_p, *d_w, *d_z;

    // temp scalars for PCG
    T rho, rho_prev, a, alpha, b;

    

    /* 2) begin PCG solve with GMG preconditioner (no restarts in this version, not much point in PCG cause low # temp vecs) */
    int n_iter = 100;
    bool can_print = true;
    int print_freq = 1;
    // int print_freq = 5;
    // int n_cycles = 2;
    // int n_cycles = 4;
    // int n_cycles = 10;

    // NOTE : I'm implementing first here a left-precond flexible PCG
    // no need for right-precond, this is the true residual despite left-precond (nice feature in PCG)

    // compute r_0 = b - Ax
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
    T a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                    d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

    // compute |r_0|
    T init_resid_norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
    if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

    // copy z => p
    CHECK_CUDA(cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));

    T rho_prev, rho; // coefficients that we need to remember
    bool converged = false;


    // if constexpr (pcg_method == 1) {
        // from this document, https://www.netlib.org/templates/templates.pdf
        // doesn't include corrections if 

        // inner loop
        for (int j = 0; j < n_iter; j++) {

            /* inner 1) solve Mz = r for z (precond) */
            // ----------------------------------------

            // set the defect of Vcycle to the residual (permuted), and set soln to zero
            cudaMemcpy(mg.grids[0].d_defect.getPtr(), d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemset(mg.grids[0].d_soln.getPtr(), 0.0, N * sizeof(T));

            // only so many steps of outer V-cycle
            bool inner_print = false, double_smooth = true;
            int print_freq = 1;
            // bool symmetric = true;
            bool symmetric = false; // this is tsronger smoother and doesn't really help PCG? some reason
            mg.vcycle_solve(0, pre_smooth, post_smooth, ncycles, inner_print, atol, rtol, omega, double_smooth, print_freq, symmetric);

            // copy out of fine grid temp vec into z the prolong solution
            cudaMemcpy(d_z, mg.grids[0].d_soln.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

            // // write precond solution
            // int *d_perm = mg.grids[0].d_perm;
            // auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
            // printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");

            /* 2) compute dot products, and p vec */
            // -------------------------------------
            
            // if fletcher-reeves method
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));

            if (j == 0) {
                // first iteration, p = z
                cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // compute beta
                T beta = rho / rho_prev;

                // p_new = z + beta * p in two steps
                a = beta;  // p *= beta scalar
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
                a = 1.0;  // p += z
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));
            }

            // store rho for next iteration (prev), only used in this part
            rho_prev = rho;

            /* 3) compute w = A * p mat-vec product */
            // ----------------------------------------

            // w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                            descrA, d_vals, d_rowp, d_cols, block_dim, d_p, &b, d_w));

            /* 4) update x and r using dot products */
            // ---------------------------------------

            // alpha = <r,z> / <w,p> = rho / <w,p>
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            T alpha = rho / wp0;

            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

            // r -= alpha * w
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));

            // copy z into zprev (for polak-riberre formula)
            cudaMemcpy(d_zprev, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);

            // check for convergence
            T resid_norm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
            if (can_print && (j % print_freq == 0)) printf("PCG [%d] = %.8e\n", j, resid_norm);

            if (abs(resid_norm) < (atol + init_resid_norm * rtol)) {
                converged = true;
                if (can_print)
                    printf("\nPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
                break;
            }
        }

    private:
};
#pragma once
#include "assembler.cuh"
#include "linalg/bsr_mat.h"
#include "solvers/linear_static/bsr_direct_LU.h"
#include "solvers/linear_static/bsr_gmres.h"

/* main assembler for the DKT plate elements..*/

class DKTPlateAssembler {
    using T = double;

   public:
    DKTPlateAssembler(int nxe_, T E_, T thick_, T nu_, T load_mag_)
        : nxe(nxe_), E(E_), nu(nu_), thick(thick_), load_mag(load_mag_) {
        nx = nxe + 1;
        nelems = nxe * nxe * 2;
        nnodes = nx * nx;
        ndof = nnodes * 3;
        N = ndof;

        init_nz_pattern();
    }

    void assemble() {
        // set ordering and then call initialize and assemble..
        // here we first move the host bsr data to device (since ordered now..)
        auto d_bsr_data = h_bsr_data.createDeviceBsrData();
        nnzb = h_bsr_data.nnzb;
        int nvals = nnzb * 9;
        d_vals = DeviceVec<T>(nvals);
        Kmat = BsrMat<DeviceVec<T>>(d_bsr_data, d_vals);

        // assembly helper methods
        assemble_lhs();
        assemble_rhs();
        initCuda();
    }

    void init_nz_pattern() {
        // init sparsity pattern or nz pattern of lhs / matrix
        int nodes_per_elem = 3;
        block_dim = 3;
        int n_conn = nelems * 3;
        int32_t *conn = new int32_t[n_conn];

        for (int ielem = 0; ielem < nelems; ielem++) {
            int iquad = ielem / 2;
            int itri = ielem % 2;

            int ixe = iquad % nxe, iye = iquad / nxe;
            int n1 = nx * iye + ixe;
            int n2 = n1 + 1, n3 = n1 + nx, n4 = n1 + nx + 1;
            int32_t *nodes = &conn[3 * ielem];
            if (itri == 0) {
                nodes[0] = n1, nodes[1] = n2, nodes[2] = n3;
            } else {
                nodes[0] = n4, nodes[1] = n3, nodes[2] = n2;
            }
        }

        h_bsr_data = BsrData(nelems, nnodes, nodes_per_elem, block_dim, conn);
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
        d_resid = DeviceVec<T>(N).getPtr();
        // d_int_temp = DeviceVec<int>(N).getPtr();

        // copy rhs into defect
        cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        // and inv permute the rhs
        d_perm = Kmat.getPerm();
        d_iperm = Kmat.getIPerm();
        auto d_bsr_data = Kmat.getBsrData();
        d_elem_conn = d_bsr_data.elem_conn;
        nelems = d_bsr_data.nelems;
        // d_defect.permuteData(block_dim, d_iperm);

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        // CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        // CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        // CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

        // // also init kmat for direct LU solves..
        // if (full_LU) {
        //     d_kmat_lu_vals = DeviceVec<T>(Kmat.get_nnz()).getPtr();
        //     CHECK_CUDA(cudaMemcpy(d_kmat_lu_vals, d_kmat_vals, Kmat.get_nnz() * sizeof(T),
        //                           cudaMemcpyDeviceToDevice));

        //     // ILU(0) factor on full LU pattern
        //     CUSPARSE::perform_ilu0_factorization(
        //         cusparseHandle, descr_kmat_L, descr_kmat_U, info_kmat_L, info_kmat_U,
        //         &kmat_pBuffer, nnodes, kmat_nnzb, block_dim, d_kmat_lu_vals, d_kmat_rowp,
        //         d_kmat_cols, trans_L, trans_U, policy_L, policy_U, dir);
        // }
    }

    void assemble_lhs() {
        // assemble the kmat
        int ncols = nelems * 27;
        dim3 block(32);
        dim3 grid((ncols + 31) / 32);

        k_assemble_stiffness_matrix<T><<<grid, block>>>(nxe, E, thick, nu, Kmat);

        // get the row from rowp.. on host (quite easily actually + we need this for bcs to make col
        // bcs also)
        int *h_rowp = new int[nnodes + 1];
        auto bsr_data = Kmat.getBsrData();
        cudaMemcpy(h_rowp, bsr_data.rowp, (nnodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        nnzb = bsr_data.nnzb;
        int *h_rows = new int[nnzb];
        for (int brow = 0; brow < nnodes; brow++) {
            for (int jp = h_rowp[brow]; jp < h_rowp[brow + 1]; jp++) {
                h_rows[jp] = brow;
            }
        }
        d_rows = HostVec<int>(nnzb, h_rows).createDeviceVec().getPtr();
        // delete[] h_rowp;
        // delete[] h_rows;

        // apply bcs..
        int nvals = 9 * nnzb;
        dim3 grid2((nvals + 31) / 32);
        d_perm = bsr_data.perm, d_iperm = bsr_data.iperm;
        d_rowp = bsr_data.rowp, d_cols = bsr_data.cols;

        k_applyBCsLHS<T><<<grid2, block>>>(nxe, nnzb, d_rows, d_cols, d_perm, d_vals.getPtr());
    }

    void assemble_rhs() {
        // assemble rhs..
        dim3 block(32);
        dim3 grid((ndof + 31) / 32);
        d_rhs = DeviceVec<T>(N);

        k_assembleRHS<T><<<grid, block>>>(load_mag, nxe, d_iperm, d_rhs.getPtr());
    }

    void direct_solve(bool print = false) {
        bool permute_inout = false;
        CUSPARSE::direct_LU_solve<T>(Kmat, d_defect, d_soln, print, permute_inout);
        // TODO : do direct solve manually like with MITC4 shells..

        // check the residual also
        T resid_nrm = getResidNorm();
        printf("direct solve completed with resid nrm %.2e\n", resid_nrm);
    }

    void gmres_solve(bool print = false) {
        // right precond, modified GS solve
        int n_iter = 50, max_iter = 300;
        T atol = 1e-7, rtol = 1e-7;
        bool can_print = print, debug = false,
             permute_inout = false;  // no permute inout because we have already permuted it?
        int print_freq = 10;
        CUSPARSE::GMRES_solve<T>(Kmat, d_defect, d_soln, n_iter, max_iter, atol, rtol, can_print,
                                 debug, print_freq, permute_inout);
        // TODO : how to handle repeated solves like direct poisson solver?

        // check the residual also
        T resid_nrm = getResidNorm();
        printf("GMRES solve completed with resid nrm %.2e\n", resid_nrm);
    }

    void printToVTK(DeviceVec<T> vec, std::string filename) {
        // to VTK method for the DKT plate element
        // NOTE : better to use F5 binary for large cases, we will handle that

        // later
        using namespace std;
        string sp = " ";
        string dataType = "double64";

        ofstream myfile;
        myfile.open(filename);
        myfile << "# vtk DataFile Version 2.0\n";
        myfile << "TACS GPU shell writer\n";
        myfile << "ASCII\n";

        // make an unstructured grid even though it is really structured
        myfile << "DATASET UNSTRUCTURED_GRID\n";
        myfile << "POINTS " << nnodes << sp << dataType << "\n";

        // print all the xpts coordinates
        T dx = 1.0 / nxe;
        for (int inode = 0; inode < nnodes; inode++) {
            int ix = inode % nx, iy = inode / nx;
            T x = ix * dx, y = iy * dx;

            myfile << x << sp << y << sp << 0.0 << "\n";
        }

        // print all the cells
        int num_elems = nelems;
        int nodes_per_elem = 3;
        int num_elem_nodes = num_elems * (nodes_per_elem + 1);  // repeats here
        myfile << "CELLS " << num_elems << " " << num_elem_nodes << "\n";

        const int *h_elem_conn = h_bsr_data.elem_conn;

        for (int ielem = 0; ielem < nxe * nxe * 2; ielem++) {
            const int *loc_elem_conn = &h_elem_conn[3 * ielem];
            myfile << 3;
            for (int inode = 0; inode < 3; inode++) {
                myfile << sp << loc_elem_conn[inode];
            }
            myfile << "\n";
        }

        // cell type 6 is for CTRIA3 basically
        myfile << "CELL_TYPES " << nelems << "\n";
        for (int ielem = 0; ielem < nelems; ielem++) {
            myfile << 6 << "\n";
        }

        // disp vector field now
        vec.permuteData(3, d_perm);  // perm for visualization
        CHECK_CUDA(cudaDeviceSynchronize());
        T *h_vec = vec.createHostVec().getPtr();

        myfile << "POINT_DATA " << nnodes << "\n";
        string scalarName = "disp";
        myfile << "VECTORS " << scalarName << " double64\n";
        for (int inode = 0; inode < nnodes; inode++) {
            myfile << 0.0 << sp;
            myfile << 0.0 << sp;
            myfile << h_vec[3 * inode] << "\n";
        }

        myfile << "VECTORS "
               << "rot"
               << " double64\n";
        for (int inode = 0; inode < nnodes; inode++) {
            myfile << h_vec[3 * inode + 1] << sp;
            myfile << h_vec[3 * inode + 2] << sp;
            myfile << 0.0 << "\n";
        }
        vec.permuteData(3, d_iperm);  // unperm after vis
        CHECK_CUDA(cudaDeviceSynchronize());
        myfile.close();
    }

    // void apply_near_identity_gmres_map(int n_iters, std::string baseName) {
    //     // apply near identity AM^-1 ILU(0) map during GMRES iteration
    //     // this method is for debugging the iteration itself..
    //     // d_defect holds normal component after the application
    //     // d_resid holds the previous defect
    //     // and d_temp holds the new defect (without orthog)

    //     // write init defect first
    // }

    T getResidNorm() {
        // get the residual nrm of the linear system R = b - Ax
        // reset resid to zero
        cudaMemcpy(d_resid, d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

        T a = -1.0, b = 1.0;  // resid = 1.0 * rhs - 1.0 * A * x
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, nnzb, &a,
                                      descrKmat, d_vals.getPtr(), d_rowp, d_cols, block_dim,
                                      d_soln.getPtr(), &b, d_resid));

        T resid_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_nrm));

        return resid_nrm;
    }

    // data
    BsrData h_bsr_data;
    int nxe, nx, nnodes, nelems, ndof, N;
    int block_dim, nnzb;
    T E, thick, nu, load_mag;
    int *d_rows, *d_rowp, *d_cols, *d_perm, *d_iperm;
    const int *d_elem_conn;
    BsrMat<DeviceVec<T>> Kmat;
    DeviceVec<T> d_rhs, d_defect, d_soln, d_temp_vec, d_vals;
    T *d_temp, *d_temp2, *d_resid;

    // cusparse utils
    cusparseMatDescr_t descrKmat = 0;
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
};
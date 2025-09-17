    auto h_soln = grid.d_defect.createPermuteVec(grid.block_dim, grid.Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_0.vtk");
    damping the colored GS (that's like damped jacobi)
    // T omega = 0.7;
    T omega = 1.0;
    DEBUG
    int color_order[] = {0,3,1,2};
    for (int iter = 0; iter < n_iters; iter++) {
        for (int _icolor = 0; _icolor < num_colors; _icolor++) {
            // don't go 0,1,2,3 colors go 0,3,2,1 order.. can also do damping or reverse order after that..
            // int icolor = color_order[_icolor];
            int icolor = _icolor;

            // get active rows / cols for this color
            int start = color_rowp[icolor], end = color_rowp[icolor + 1];
            int nblock_rows_color = end - start;
            // int block_dim2 = grid.block_dim * grid.block_dim;  // 36

            printf("iter %d, color %d : block rows [%d,%d)\n", iter, icolor, start, end);

            // 1) compute Dinv_c * defect_c => dx_c  (c indicates color subset)
            // int color_Dinv_nnzb = nblock_rows_color;
            // can use same rowp, cols here (0,1,...,nrows)
            // int *d_color_diag_cols = &grid.d_diag_cols[grid.block_dim * start];
            // T *d_Dinv_vals_color = &d_diag_inv_vals[block_dim2 * start];
            T *d_defect_color = &grid.d_defect.getPtr()[grid.block_dim * start];
            cudaMemset(grid.d_temp, 0.0, N * sizeof(T));  // holds dx_color
            T *d_temp_color = &grid.d_temp[grid.block_dim * start];
            T *d_temp_color2 = &grid.d_temp2[grid.block_dim * start];
            cudaMemset(grid.d_temp2, 0.0, N * sizeof(T)); // DEBUG
            cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * grid.block_dim * sizeof(T), cudaMemcpyDeviceToDevice);

            T a = 1.0, b = 0.0;

            const double alpha = 1.0;
            // try normal triang solve first.. DEBUG
            // printf("try normal triang solve first (DEBUG), get rid of this\n");
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(grid.cusparseHandle, grid.dir, grid.trans_L, grid.nnodes,
                                                    grid.diag_inv_nnzb, &alpha, grid.descr_L,
                                                    grid.d_diag_LU_vals, grid.d_diag_rowp, grid.d_diag_cols,
                                                    grid.block_dim, grid.info_L, grid.d_temp2,
                                                    grid.d_resid, grid.policy_L, grid.pBuffer)); // prob only need U^-1 part for block diag.. TBD
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(grid.cusparseHandle, grid.dir, grid.trans_U, grid.nnodes,
                                                    grid.diag_inv_nnzb, &alpha, grid.descr_U,
                                                    grid.d_diag_LU_vals, grid.d_diag_rowp, grid.d_diag_cols,
                                                    grid.block_dim, grid.info_U, grid.d_resid,
                                                    grid.d_temp, grid.policy_U, grid.pBuffer));

            // 2) update soln x_color += dx_color
            int nrows_color = nblock_rows_color * grid.block_dim;
            T *d_soln_color = &grid.d_soln.getPtr()[grid.block_dim * start];
            // a = 1.0;
            a = omega;
            CHECK_CUBLAS(
                cublasDaxpy(grid.cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));
            // printf("here3\n");

            // 3) update defect, defect -= K[color,:]^T * dx_color, with KT_color = N x
            // nrows_color matrix
            // int kmat_bnz_start = grid.h_color_bnz_ptr[icolor];
            // int kmat_bnz_end = grid.h_color_bnz_ptr[icolor + 1];
            // int color_kmat_nnzb = kmat_bnz_end - kmat_bnz_start;
            // T *d_color_kmat_vals = &grid.d_kmat_vals[36 * kmat_bnz_start];
            // int local_color_rowp_start = grid.h_color_local_rowp_ptr[icolor];
            // printf("kmat bnz %d to %d, local rowp start %d\n", kmat_bnz_start, kmat_bnz_end,
            //         local_color_rowp_start);
            // int *d_kmat_color_local_rowp = &grid.d_color_local_rowps[local_color_rowp_start];
            // int *d_kmat_color_cols = &grid.d_kmat_cols[kmat_bnz_start];
            // a = -1.0,
            a = -omega,
            b = 1.0;  // so that defect := defect - mat*vec

            CHECK_CUSPARSE(cusparseDbsrmv(
                grid.cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                grid.nnodes, grid.nnodes, grid.kmat_nnzb, &a, grid.descrKmat, grid.d_kmat_vals, grid.d_kmat_rowp, grid.d_kmat_cols,
                grid.block_dim, grid.d_temp, &b, grid.d_defect.getPtr()));
            // printf("here4\n");

            // auto h_soln = grid.d_soln.createHostVec(); // DEBUG
            auto h_soln = grid.d_defect.createPermuteVec(grid.block_dim, grid.Kmat.getPerm()).createHostVec();
             printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_" + std::to_string(num_colors * iter + icolor + 1) + ".vtk");

        }  // next color iteration

        printf("iter %d, done with color iterations\n", iter);

        /* report progress of defect nrm if printing.. */
        T defect_nrm;
        CHECK_CUBLAS(cublasDnrm2(grid.cublasHandle, N, grid.d_defect.getPtr(), 1, &defect_nrm));
        if (print && iter % print_freq == 0)
            printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

    }  // next block-GS iteration




// TODO : would like to just rewrite the full assembler later, let's do less intrusive wrappers for
// assembler from BDF, plate, cylinder or other geometries like what I did in the python for mitc
// shell..

// #include <cusparse_v2.h>
// #include <thrust/device_vector.h>

// #include "cublas_v2.h"
// #include "cuda_utils.h"
// #include "linalg/vec.h"
// #include "element.cuh"

// #include "element/shell/shell_elem_group.h"
// #include "element/shell/shell_elem_group.cuh"

// class ShellMultigridAssembler {
//     // lighter weight assembler for geometric multigrid (single grid here)
//     // TODO : would like to rewrite this from scratch again.. so more parallel, NZ pattern on GPU
//   public:

//     using T = double;
//     using Quad = QuadLinearQuadrature<T>;
//     using Basis = ShellQuadBasis<T, Quad, 2>;
//     using Geo = Basis::Geo;
//     using Data = ShellIsotropicData<T, false>;
//     using Physics = IsotropicShell<T, Data, false>;
//     using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;

//     static constexpr int xpts_per_node = 3;
//     static constexpr int vars_per_node = 6;

//     ShellMultigridAssembler(int num_nodes_, int num_elements_, DeviceVec<int> d_elem_conn,
//         DeviceVec<int> d_bcs, DeviceVec<T> d_xpts, DeviceVec<Data> d_data) :
//         num_nodes(num_nodes_), num_elements(num_elements_) {

//         elem_conn = d_elem_conn;
//         bcs = d_bcs;
//         xpts = d_xpts;
//         data = d_data;

//         num_dof = num_nodes * vars_per_node;

//         // zero vars
//         vars = DeviceVec<T>(num_dof);

//         // construct initial bsr data
//         h_bsr_data = BsrData(num_elements, num_nodes, 4, 6, elem_conn.getPtr());
//         d_bsr_data = h_bsr_data.createDeviceBsrData();
//     }

//     void compute_nofill_pattern() {
//         // compute nofill pattern (with desired ordering as well..) for coloring

//     }

//     void assembleJacobian() {

//         // assemble the Kmat
//         dim3 block(1, 24, 4); // (1 elems_per_block, 24 DOF per elem, 4 quadpts per elem)
//         int nblocks = (num_elements + block.x - 1) / block.x;
//         dim3 grid(nblocks);

//         add_jacobian_gpu<T, ElemGroup, Data, 1, DeviceVec, BsrMat><<<grid, block>>>(
//             num_nodes, num_elements, elem_conn, xpts, vars, physData, res, K_mat);

//         CHECK_CUDA(cudaDeviceSynchronize());
//     }

//     // public data
//     int num_nodes, num_dof, num_elements;
//     DeviceVec<int> bcs, elem_conn;
//     DeviceVec<T> xpts, vars;
//     DeviceVec<Data> data;
//     BsrData bsr_data;
//     BsrMat K_mat, Dinv_mat;
// };

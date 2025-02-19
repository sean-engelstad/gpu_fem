#include "../plate_demo/_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "mesh/vtk_writer.h"
#include "shell/shell.h"
#include <iostream>

/**
 solve on CPU with cusparse for debugging
 **/

int main(void)
{
    using T = double;

    std::ios::sync_with_stdio(false); // always flush print immediately

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 10; // 100
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler =
        createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    // bool perform_fillin = false;
    // double fillin = 10.0; // 10.0
    // bool print = true;
    // if (perform_fillin) {
    //     assembler.symbolic_factorization(fillin, print);
    // }

    auto d_bsr_data = assembler.getBsrData();
    // move it back onto the host for these manipulations
    // auto bsr_data = d_bsr_data.createHostBsrData(); 
    // // no longer needed only copies to device after symbolic factorization
    auto bsr_data = d_bsr_data;
    int *orig_rowPtr = bsr_data.rowPtr;
    int *orig_colPtr = bsr_data.colPtr;
    int nnodes = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int orig_nnzb = nnzb;

    printf("post host bsr data transfer rowPtr: ");
    printVec<int>(nnodes + 1, orig_rowPtr);

    // RCM reordering // based on 1585 of TACSAssembler.cpp
    bool perform_rcm = true;
    int *rowPtr, *colPtr;
    int *perm = new int[nnodes];
    double fill = 10.0;
    
    if (perform_rcm)
    {
        int *_new_perm = new int[nnodes];
        int root_node = 0;
        int num_rcm_iters = 1;
        TacsComputeRCMOrder(nnodes, orig_rowPtr, orig_colPtr, _new_perm,
                            root_node, num_rcm_iters);
        // also do we need to reverse the order here? (see line 1585 of
        // TACSAssembler.cpp)
        if (perm)
        {
            for (int k = 0; k < nnodes; k++)
            {
                perm[_new_perm[k]] = k;
            }
        }

        printf("perm: ");
        printVec<int>(nnodes, perm);

        // copy back so perm doesn't get deleted with su_mat
        // for (int i = 0; i < nnodes; i++) {
        //     _new_perm[i] = perm[i];
        // }

        // printVec<int>(nnodes, _new_perm);

        // then do the symbolic factorization
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(
            nnodes, nnodes, nnzb, orig_rowPtr, orig_colPtr, nullptr);
        // su_mat.perm = perm;
        printf("pre-reorder rowPtr: ");
        printVec<int>(nnodes + 1, orig_rowPtr);

        auto su_mat2 =
            SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, perm, fill);

        nnzb = su_mat2->nnz;
        rowPtr = su_mat2->rowp;
        colPtr = su_mat2->cols;

        // perm = su_mat2->perm;

        printf("rowPtr: ");
        printVec<int>(nnodes + 1, rowPtr);
        // printf("colPtr: ");
        // printVec<int>(nnzb, colPtr);

        printf("perm0: ");
        printVec<int>(nnodes, perm);

        // get reverse perm as iperm, iperm as perm from convention in
        // sparse-utils is flipped from my definition of reordering matrix..
        // (realized this after code was written)
        // int *perm = su_mat2->iperm;
        // int *iperm = su_mat2->perm;

        write_to_csv<int>(rowPtr, nnodes + 1, "csv/RCM_rowPtr.csv");
        write_to_csv<int>(colPtr, nnzb, "csv/RCM_colPtr.csv");
    }

    printf("perm1: ");
    printVec<int>(nnodes, perm);

    bool perform_qordering = true;
    int *qorder_perm; //, *qorder_iperm;
    if (perform_qordering)
    {
        // compute bandwidth of RCM reordered sparsity
        int bandwidth = getBandWidth(nnodes, nnzb, rowPtr, colPtr);

        // choose a p factor (1/p * bandwidth => # rows for random reordering in
        // q-ordering)
        double p = 1.0; // example values 2.0, 1.0, 0.5, 0.25 (lower values are
                        // more random and higher bandwidth)
        int prune_width = (int)1.0 / p * bandwidth;

        printf("bandwidth %d, prune_width %d\n", bandwidth, prune_width);
        int num_prunes = (nnodes + prune_width - 1) / prune_width;
        // random number generator
        std::random_device rd;
        std::mt19937 g(rd());

        // build new permutation of q-ordering from old permutation
        std::vector<int> q_perm(
            perm, perm + nnodes); // TODO : define it relative to perm
        // perform random reordering on each prune_width size
        for (int iprune = 0; iprune < num_prunes; iprune++)
        {
            // TODO : how to best apply extra permutation on top of current one?
            int lower = prune_width * iprune;
            int upper = std::min(lower + prune_width, nnodes);
            std::shuffle(q_perm.begin() + lower, q_perm.begin() + upper, g);
        }

        printf("perm: ");
        printVec<int>(nnodes, perm);

        printf("q_perm: ");
        printVec<int>(nnodes, q_perm.data());

        // now compute the new rowPtr, colPtr after fillin (this is full
        // fillin here) later we'll do ILU with it
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(
            nnodes, nnodes, nnzb, orig_rowPtr, orig_colPtr, nullptr);
        // su_mat.perm = q_perm.data();
        auto su_mat2 = SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(
            su_mat, q_perm.data(), fill);

        nnzb = su_mat2->nnz;
        rowPtr = su_mat2->rowp;
        colPtr = su_mat2->cols;

        // copy q-ordering permutation into q_perm array
        qorder_perm = new int[nnodes];
        std::copy(q_perm.begin(), q_perm.end(), qorder_perm);

        

        write_to_csv<int>(rowPtr, nnodes + 1, "csv/qorder_rowPtr.csv");
        write_to_csv<int>(colPtr, nnzb, "csv/qorder_colPtr.csv");

    }

    // if (perform_rcm || perform_qordering) {
    //     // write out matrix sparsity to debug (after reordering)
    //     // this is full fillin here though
    //     write_to_csv<int>(rowPtr, nnodes + 1, "csv/plate_rowPtr.csv");
    //     write_to_csv<int>(colPtr, nnzb, "csv/plate_colPtr.csv");
    // }

    // compute also the qorder_iperm
    int *final_perm, *final_iperm;
    if (perform_qordering) {
        final_perm = qorder_perm;
    } else {
        final_perm = perm; // assumes RCM fillin here
    }
    final_iperm = new int[nnodes];
    for (int inode = 0; inode < nnodes; inode++) {
        final_iperm[final_perm[inode]] = inode;
    }

    // compute ILU(k)
    bool perform_ilu = true;
    int *kmat_rowPtr, *kmat_colPtr, kmat_nnzb;
    if (perform_ilu)
    {
        // use orig rowPtr, colPtr here so that you don't use full fillin
        auto A = SparseUtils::BSRMat<double,1,1>(nnodes, nnodes, orig_nnzb, orig_rowPtr, orig_colPtr, nullptr);
        // A.perm = final_perm;
        // A.iperm = final_iperm; // this may delete perm here, may want to put in ApplyPerm directly
        auto A2 = SparseUtils::BSRMatApplyPerm<double,1>(A, final_perm, final_iperm);
        int levFill = 3;
        int *levels;

        // get permuted no-fill pattern for Kmat
        kmat_rowPtr = A2->rowp;
        kmat_colPtr = A2->cols;
        kmat_nnzb = A2->rowp[nnodes];

        computeILUk(nnodes, nnzb, A2->rowp, A2->cols, levFill, fill, &levels);

        // copy the new rowptr, colptr out
        rowPtr = A2->rowp;
        colPtr = A2->cols;
        nnzb = rowPtr[nnodes];

        // write out matrix sparsity to debug
        write_to_csv<int>(rowPtr, nnodes + 1, "csv/ILU_rowPtr.csv");
        write_to_csv<int>(colPtr, nnzb, "csv/ILU_colPtr.csv");

    }

    // check if perm was destroyed here..
    printf("check perm: ");
    printVec<int>(nnodes, final_perm);

    
    // TODO: clean up all of the above and move into main repos

    // make the bsr data for the preconditioner
    auto precond_bsr_data = BsrData(nnodes, 6, nnzb, rowPtr, colPtr, final_perm, final_iperm);
    // update kmat bsr data with reordering
    bsr_data.post_reordering_update(kmat_nnzb, kmat_rowPtr, kmat_colPtr, final_perm, final_iperm);

    // move bsr data onto the device
    auto d_precond_bsr_data = precond_bsr_data.createDeviceBsrData();
    auto d_bsr_data = bsr_data.createDeviceBsrData();

    // now make the kmat and preconditioner
    auto precond_mat = BsrMat<VecType<T>>(d_precond_bsr_data);
    auto kmat = BsrMat<VecType<T>>(d_bsr_data);

    // build the preconditioner with ILU(k) and reordered rowPtr, colPtr (qordering)
    auto precond = Preconditioner<VecType<T>>(precond_mat, kmat);

    // cusparse solve section
    // -------------------------------------------------

    // (1) compute the ILU(0) preconditioner on GPU 
    // in the ILU(k) fill pattern (so equiv to ILU(k) preconditioner)
    // -------------------

    precond.copyValues();
    double *precond_vals = precond_mat.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseStatus_t status;

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    bsrilu02Info_t info_M = 0;
    bsrsv2Info_t info_L = 0;
    bsrsv2Info_t info_U = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // step 1: create a descriptor which contains
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for bsrilu02 and two info's for bsrsv2
    cusparseCreateBsrilu02Info(&info_M);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and
    // allocate the buffer
    cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, precond_vals,
                                 rowPtr, colPtr, blockDim, info_M,
                                 &pBufferSize_M);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L,
                               precond_vals, rowPtr, colPtr, blockDim,
                               info_L, &pBufferSize_L);
    cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U,
                               precond_vals, rowPtr, colPtr, blockDim,
                               info_U, &pBufferSize_U);
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // Step 4.1: ILU(0) Symbolic Analysis
    cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, precond_vals,
        rowPtr, colPtr, blockDim, info_M,
        policy_M, pBuffer);

    // Step 4.2: Check for Structural Zero Pivot
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // Step 4.3: Perform ILU(0) Numeric Factorization (M ≈ L * U)
    cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, precond_vals, rowPtr,
    colPtr, blockDim, info_M, policy_M, pBuffer);

    // Step 4.4: Check for Numerical Zero Pivot in U
    status = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("block U(%d,%d) is not invertible\n", numerical_zero,
    numerical_zero);
    }

    // Step 4.5: Analyze the Sparsity Pattern of L and U for Triangular Solves
    cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, precond_vals,
        rowPtr, colPtr, blockDim, info_L,
        policy_L, pBuffer);
    cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, precond_vals,
        rowPtr, colPtr, blockDim, info_U,
        policy_U, pBuffer);

    // (2) implement GMRES solver
    // -------------------

    
    
    // implement matrix-vec products (K un-filled in), preconditioner solve M^-1 x
    // write GMRES algorithm (running with data on GPU)

    return 0;
};

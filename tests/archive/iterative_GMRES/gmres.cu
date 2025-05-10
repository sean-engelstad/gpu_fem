<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
#include "../../plate_demo/_plate_utils.h"
========
#include <iostream>

#include "../plate_demo/_plate_utils.h"
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
#include "chrono"
#include "linalg/linalg.h"
#include "mesh/vtk_writer.h"
#include "shell/shell.h"

/**
Important links in cublas and cusparse:
(you can ctrl+F for the routines and their syntax/args in here)

cublas: https://docs.nvidia.com/cuda/cublas/
cusparse: https://docs.nvidia.com/cuda/cusparse/
*/

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    // problem size (computational size)
    // ---------------------
    int nxe = 100; // 10

    // reordering inputs
    // ---------------------

    bool perform_rcm = true;
    bool perform_qordering = true;
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    double qorder_p = 0.5; // lower p value is more nnz : 0.5, 1.0, 2.0
    bool perform_ilu = true;
    int levFill = 3; // ILU(k) fill level 0,1,2,3,...

    // load and other utils/settings
    // --------------------
    bool swap_perm = false; // correct is false, true makes solve fast but that's cause the matrix is wrong
    bool random_load = true;

    bool scale_matrix = false;
    double matrix_scale = 1e-4;

    // GMRES inputs
    // ---------------------
    int m = 100; // number of GMRES iterations
    double tolerance = 1e-14;
========
    double qorder_p = 2.0;  // lower p value is more nnz : 0.5, 1.0, 2.0
    bool perform_ilu = true;
    int levFill = 2;  // ILU(k) fill level 0,1,2,3,...

    // GMRES inputs
    // ---------------------
    int m = 30;  // number of GMRES iterations
    double tolerance = 1e-11;
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    // ----------------------

    using T = double;

    std::ios::sync_with_stdio(false);  // always flush print immediately

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // int nxe = 30; // 100
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    // bool perform_fillin = false;
    // double fillin = 10.0; // 10.0
    // bool print = true;
    // if (perform_fillin) {
    //     assembler.symbolic_factorization(fillin, print);
    // }

    auto bsr_data = assembler.getBsrData();
    // move it back onto the host for these manipulations
    // auto bsr_data = d_bsr_data.createHostBsrData();
    // // no longer needed only copies to device after symbolic factorization
    // auto bsr_data = d_bsr_data;
    int *orig_rowPtr = bsr_data.rowPtr;
    int *orig_colPtr = bsr_data.colPtr;
    int nnodes = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int orig_nnzb = nnzb;

    // printf("post host bsr data transfer rowPtr: ");
    // printVec<int>(nnodes + 1, orig_rowPtr);

    // RCM reordering // based on 1585 of TACSAssembler.cpp
    // bool perform_rcm = true;
    int *rowPtr, *colPtr;
    int *perm = new int[nnodes];
    double fill = 10.0;

    if (perform_rcm) {
        int *_new_perm = new int[nnodes];
        int root_node = 0;
        int num_rcm_iters = 1;
        TacsComputeRCMOrder(nnodes, orig_rowPtr, orig_colPtr, _new_perm, root_node, num_rcm_iters);
        // also do we need to reverse the order here? (see line 1585 of
        // TACSAssembler.cpp)
        if (perm) {
            for (int k = 0; k < nnodes; k++) {
                perm[_new_perm[k]] = k;
            }
        }

        // printf("perm: ");
        // printVec<int>(nnodes, perm);

        // copy back so perm doesn't get deleted with su_mat
        // for (int i = 0; i < nnodes; i++) {
        //     _new_perm[i] = perm[i];
        // }

        // printVec<int>(nnodes, _new_perm);

        // then do the symbolic factorization
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, orig_rowPtr,
                                                        orig_colPtr, nullptr);
        // su_mat.perm = perm;
        // printf("pre-reorder rowPtr: ");
        // printVec<int>(nnodes + 1, orig_rowPtr);

        auto su_mat2 = SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, perm, fill);

        nnzb = su_mat2->nnz;
        rowPtr = su_mat2->rowp;
        colPtr = su_mat2->cols;

        // perm = su_mat2->perm;

        // printf("rowPtr: ");
        // printVec<int>(nnodes + 1, rowPtr);
        // printf("colPtr: ");
        // printVec<int>(nnzb, colPtr);

        // printf("perm0: ");
        // printVec<int>(nnodes, perm);

        // get reverse perm as iperm, iperm as perm from convention in
        // sparse-utils is flipped from my definition of reordering matrix..
        // (realized this after code was written)
        // int *perm = su_mat2->iperm;
        // int *iperm = su_mat2->perm;

        write_to_csv<int>(rowPtr, nnodes + 1, "csv/RCM_rowPtr.csv");
        write_to_csv<int>(colPtr, nnzb, "csv/RCM_colPtr.csv");
    }

    // printf("perm1: ");
    // printVec<int>(nnodes, perm);

    // bool perform_qordering = true;
    int *qorder_perm;  //, *qorder_iperm;
    if (perform_qordering) {
        // compute bandwidth of RCM reordered sparsity
        int bandwidth = getBandWidth(nnodes, nnzb, rowPtr, colPtr);

        // choose a p factor (1/p * bandwidth => # rows for random reordering in
        // q-ordering)
        // double p = 1.0; // example values 2.0, 1.0, 0.5, 0.25 (lower values are
        // more random and higher bandwidth)
        int prune_width = (int)1.0 / qorder_p * bandwidth;

        printf("bandwidth %d, prune_width %d\n", bandwidth, prune_width);
        int num_prunes = (nnodes + prune_width - 1) / prune_width;
        // random number generator
        std::random_device rd;
        std::mt19937 g(rd());

        // build new permutation of q-ordering from old permutation
        std::vector<int> q_perm(perm, perm + nnodes);  // TODO : define it relative to perm
        // perform random reordering on each prune_width size
        for (int iprune = 0; iprune < num_prunes; iprune++) {
            // TODO : how to best apply extra permutation on top of current one?
            int lower = prune_width * iprune;
            int upper = std::min(lower + prune_width, nnodes);
            std::shuffle(q_perm.begin() + lower, q_perm.begin() + upper, g);
        }

        // printf("perm: ");
        // printVec<int>(nnodes, perm);

        // printf("q_perm: ");
        // printVec<int>(nnodes, q_perm.data());

        // now compute the new rowPtr, colPtr after fillin (this is full
        // fillin here) later we'll do ILU with it
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, orig_rowPtr,
                                                        orig_colPtr, nullptr);
        // su_mat.perm = q_perm.data();
        auto su_mat2 =
            SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, q_perm.data(), fill);

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
        final_perm = perm;  // assumes RCM fillin here
    }
    final_iperm = new int[nnodes];
    for (int inode = 0; inode < nnodes; inode++) {
        final_iperm[final_perm[inode]] = inode;
    }

    // compute ILU(k)
    // bool perform_ilu = true;
    int *kmat_rowPtr, *kmat_colPtr, kmat_nnzb;
    if (perform_ilu) {
        // use orig rowPtr, colPtr here so that you don't use full fillin
        auto A = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, orig_nnzb, orig_rowPtr,
                                                   orig_colPtr, nullptr);
        // A.perm = final_perm;
        // A.iperm = final_iperm; // this may delete perm here, may want to put in ApplyPerm
        // directly
        auto A2 = SparseUtils::BSRMatApplyPerm<double, 1>(A, final_perm, final_iperm);
        // int levFill = 3;
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
    // printf("check perm: ");
    // printVec<int>(nnodes, final_perm);

    // NOTE : flips the order of perm, iperm when storing in the kmat (because I use reverse
    // convention in my code)

    // TODO : do I need to swap lperm, rperm? maybe to get correct answer
    int *my_perm, *my_iperm;
    // swap_perm is true gets the wrong kmat with only on diagonal mostly, and then solves easily because nearly diagonally dominant
    // swap_perm is false gets correct kmat, but my GMRES algorithm has givens rotations oscillating a lot +-1 rapidly (so preconditioner)
    // problem or some other kind of problem
    if (!swap_perm) {
        // I think this one is correct as this is what I did in verifying the linear solves with AMD sparse_utils reordering => bsr_utils..
        my_perm = final_iperm;
        my_iperm = final_perm;
    } else {
        // but this one makes the preconditioner norms seem to make more sense
        my_perm = final_perm;
        my_iperm = final_iperm;
    }
    

    // TODO: clean up all of the above and move into main repos
    // update kmat bsr data with reordering (flipped perm, iperm here bc flipped convention)
    bsr_data.post_reordering_update(kmat_nnzb, kmat_rowPtr, kmat_colPtr, my_perm, my_iperm);
    // printf("checkpt0.5\n");

    // make the bsr data for the preconditioner (flipped perm, iperm here bc flipped convention)
    auto precond_bsr_data = BsrData(nnodes, 6, nnzb, rowPtr, colPtr, my_perm, my_iperm);

    // printf("kmat_nnzb = %d\n", kmat_nnzb);
    // printf("kmat rowPtr:");
    // printVec<int>(nnodes+1, kmat_rowPtr);

    // printf("checkpt1\n");

    // move bsr data onto the device
    auto d_precond_bsr_data = precond_bsr_data.createDeviceBsrData();
    // printf("checkpt1.5\n");
    auto d_bsr_data = bsr_data.createDeviceBsrData();

    // printf("checkpt2\n");

    // now make the kmat and preconditioner
    auto precond_mat = BsrMat<VecType<T>>(d_precond_bsr_data);
    auto kmat = BsrMat<VecType<T>>(d_bsr_data);

    // printf("checkpt3\n");

    // also get soln and rhs vectors
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    double Q = 1000.0; // load magnitude
========
    double Q = 1.0;  // load magnitude
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
    // sine distributed loads
    T *my_loads;
    if (!random_load) {
        my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    } else {
        int nnx = nxe + 1;
        int nny = nye + 1;
        int num_nodes = nnx * nny;
        int num_dof = 6 * num_nodes;

        my_loads = new T[num_dof];
        memset(my_loads, 0.0, num_dof * sizeof(T));
        for (int i = 0; i < num_dof; ++i) {
            my_loads[i] = static_cast<double>(std::rand()) / RAND_MAX;  // Scale to [0, 1]
        }
    }

    // printf("my_loads: ");
    // printVec<double>(100, my_loads);

    // add jacobian and apply bcs
    double print = true;
    assembler.add_jacobian(res, kmat, print);
    assembler.apply_bcs(res, print);
    assembler.apply_bcs(kmat, print);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads, true);

    // scale the matrix
    if (scale_matrix) {
        kmat.scale(matrix_scale);
        loads.scale(matrix_scale);
    }

    // printf("checkpt4\n");

    // build the preconditioner with ILU(k) and reordered rowPtr, colPtr (qordering)
    auto precond = ILUk_Preconditioner<VecType<T>>(precond_mat, kmat);
    precond.copyValues();

    // check whether kmat to preconditioner copying worked

    // print kmat
    auto h_kmat = kmat.getVec().createHostVec();
    write_to_csv<int>(kmat_rowPtr, nnodes + 1, "csv/kmat_rowp.csv");
    write_to_csv<int>(kmat_colPtr, kmat_nnzb, "csv/kmat_cols.csv");
    write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(), "csv/kmat_vals.csv");

    // print preconditioner
    auto h_precond_mat = precond_mat.getVec().createHostVec();
    write_to_csv<int>(rowPtr, nnodes + 1, "csv/precond_rowp.csv");
    write_to_csv<int>(colPtr, rowPtr[nnodes], "csv/precond_cols.csv");
    write_to_csv<double>(h_precond_mat.getPtr(), h_precond_mat.getSize(), "csv/precond_vals.csv");

    // return 0;

    // printf("checkpt5\n");

    // cusparse solve section
    // -------------------------------------------------

    // (1) compute the ILU(0) preconditioner on GPU
    // in the ILU(k) fill pattern (so equiv to ILU(k) preconditioner)
    // -------------------

    double *precond_vals = precond_mat.getPtr();
    double *kmat_vals = kmat.getPtr();
    int mb = bsr_data.nnodes;
    int block_dim = 6;

    // device row and col ptrs
    int *dp_rowp = d_precond_bsr_data.rowPtr;
    int *dp_cols = d_precond_bsr_data.colPtr;
    int *dk_rowp = d_bsr_data.rowPtr;
    int *dk_cols = d_bsr_data.colPtr;

    // int ct = 0;
    // ct++;
    // printf("checkpt%d\n", ct);

    // auto h_precond_vals = precond_mat.getVec().createHostVec();
    // printf("precond vals\n");
    // printVec<double>(h_precond_vals.getSize(), h_precond_vals.getPtr());

    cusparseHandle_t cusparseHandle;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // Create original matrix descriptor A
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // initialize L and U precond matrices
    // descr_M = 0,
    cusparseMatDescr_t  descr_L = 0, descr_U = 0;
    // bsrilu02Info_t info_M = 0;
    bsrilu02Info_t info_L_fct = 0, info_U_fct = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    // int pBufferSize_M, 
    int pBufferSize_L, pBufferSize_U, pBufferSize;
    void *pBuffer = 0;
    int structural_zero, numerical_zero;
    // do this CUSPARSE_SOLVE_POLICY_USE_LEVEL so that it parallelizes the triangular solves better
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    // const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    // const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL; // CUSPARSE_SOLVE_POLICY_NO_LEVEL, CUSPARSE_SOLVE_POLICY_USE_LEVEL
    // const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
========
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L =
        CUSPARSE_SOLVE_POLICY_NO_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                         // CUSPARSE_SOLVE_POLICY_USE_LEVEL
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // step 1: create a descriptor which contains
    // cusparseCreateMatDescr(&descr_M);
    // cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    // cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL); // need general for ilu
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL); // need general for ilu
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for bsrilu02 and two info's for bsrsv2
    cusparseCreateBsrilu02Info(&info_L_fct);
    cusparseCreateBsrilu02Info(&info_U_fct);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);

    // step 3: query how much memory used in bsrilu02 and bsrsv2, and
    // allocate the buffer
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    // CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M, precond_vals,
    //                              dp_rowp, dp_cols, block_dim, info_M,
    //                              &pBufferSize_M))
========
    CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M,
                                                precond_vals, dp_rowp, dp_cols, block_dim, info_M,
                                                &pBufferSize_M))
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                              precond_vals, dp_rowp, dp_cols, block_dim, info_L,
                                              &pBufferSize_L))
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
        precond_vals, dp_rowp, dp_cols, block_dim,
                               info_U, &pBufferSize_U))
    // pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    pBufferSize = max(pBufferSize_L, pBufferSize_U);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // Perform ILU numerical factorization directly into L and U
    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_L, precond_vals, dp_rowp, dp_cols, block_dim, info_L_fct, policy_L, pBuffer));
    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_U, precond_vals, dp_rowp, dp_cols, block_dim, info_U_fct, policy_U, pBuffer));

    cusparseStatus_t status_L = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_L_fct, &structural_zero);
    if (status_L == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("L factor has a zero pivot at row %d\n", structural_zero);
    }

    cusparseStatus_t status_U = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_U_fct, &structural_zero);
    if (status_U == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("U factor has a zero pivot at row %d\n", structural_zero);
========
                                              precond_vals, dp_rowp, dp_cols, block_dim, info_U,
                                              &pBufferSize_U))
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void **)&pBuffer, pBufferSize);

    // analyze ILU structure of M
    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_M, precond_vals,
                                              dp_rowp, dp_cols, block_dim, info_M, policy_M,
                                              pBuffer))
    cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // switched the order here.. so Dbsrilu02 is after L and U analyses
    // perform ILU numerical factorization in M
    CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, precond_vals, dp_rowp,
                                     dp_cols, block_dim, info_M, policy_M, pBuffer))
    int numerical_zero;
    status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
    }

    // Now perform the actual ILU factorization into L and U
    CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_L, precond_vals, dp_rowp, dp_cols, block_dim, info_L_fct, policy_L, pBuffer));
    CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_U, precond_vals, dp_rowp, dp_cols, block_dim, info_U_fct, policy_U, pBuffer));

    status_L = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_L_fct, &numerical_zero);
    if (status_L == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("block L(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    status_U = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_U_fct, &numerical_zero);
    if (status_U == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    // ---- Change Matrix Type to TRIANGULAR for bsrsv2 (solves) ----
    // cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    // cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_TRIANGULAR);

    // analyze sparsity pattern of L and U for later triangular solves
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L, 
        precond_vals, dp_rowp, dp_cols, block_dim, info_L,
         policy_L, pBuffer))
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U, 
            precond_vals, dp_rowp, dp_cols, block_dim, info_U,
            policy_U, pBuffer))

    CHECK_CUDA(cudaDeviceSynchronize());

    // check preconditioner after LU factorization
    auto d_precond_vec2 = DeviceVec<T>(h_precond_mat.getSize(),precond_vals);
    auto h_precond_mat2 = d_precond_vec2.createHostVec();
    write_to_csv<int>(rowPtr,nnodes+1, "csv/precond2_rowp.csv");
    write_to_csv<int>(colPtr, rowPtr[nnodes], "csv/precond2_cols.csv");
    write_to_csv<double>(h_precond_mat2.getPtr(), h_precond_mat2.getSize(), "csv/precond2_vals.csv");
    
========
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                            precond_vals, dp_rowp, dp_cols, block_dim, info_L,
                                            policy_L, pBuffer))
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                            precond_vals, dp_rowp, dp_cols, block_dim, info_U,
                                            policy_U, pBuffer))
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    // (2) implement GMRES solver
    // -------------------

    // need to permute loads to rhs (so we account for reordering)
    DeviceVec<T> rhs = bsr_pre_solve<DeviceVec<T>>(kmat, loads, soln);

    // create vector descriptor types
    // cusparseDnVecDescr_t d_X, d_B, d_tmp1, d_tmp2, d_tmp3, d_R;
    auto resid = assembler.createVarsVec();
    auto tmp1 = assembler.createVarsVec();
    auto tmp2 = assembler.createVarsVec();
    auto tmp3 = assembler.createVarsVec();
    auto w = assembler.createVarsVec();
    int nvars = resid.getSize();

    double *resid_ptr = resid.getPtr();
    double *tmp1_ptr = tmp1.getPtr();
    double *tmp2_ptr = tmp2.getPtr();
    double *tmp3_ptr = tmp3.getPtr();
    double *w_ptr = w.getPtr();
    double *soln_ptr = soln.getPtr();

    // double one = 1.0, zero = 0.0, minus_one = -1.0;

    // compute initial resid = b - A * x0
    rhs.copyValuesTo(resid);  // copies b into resid

    // compute resid = - A * x0 + resid
    double a = -1.0, b = 1.0;
    // computes y = a * A * x + b * y
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  kmat_vals, dk_rowp, dk_cols, block_dim, soln_ptr, &b, resid_ptr));
    // can't do cusparseSPMV because that only works for CSR matrices
    // can only do cusparseDbsrmv which they will soon deprecate :(

    // then copy resid into tmp1
    resid.copyValuesTo(tmp1);

    // then compute resid = U^-1 * L^-1 * tmp (M^-1 on the init resid for preconditioner)
    // compute resid = U^-1 * L^-1 * b (with preconditioner)

    //    (a) L^-1 tmp1 => tmp2    (triangular solver)
    a = 1.0;
    // printf("checkpt1\n");
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                         precond_vals, dp_rowp, dp_cols, block_dim, info_L,
                                         tmp1_ptr, tmp2_ptr, policy_L, pBuffer))

    // printf("checkpt2\n");

    //    (b) U^-1 tmp2 => resid    (triangular solver)
    a = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                         precond_vals, dp_rowp, dp_cols, block_dim, info_U,
                                         tmp2_ptr, resid_ptr, policy_U, pBuffer))

    // printf("checkpt3\n");

    // we have now computed resid = M^-1* (resid - A * x0)

    // now compute beta = || r_0 || the initial resid norm
    double nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, resid_ptr, 1, &nrm_R))
    double threshold = tolerance * nrm_R;
    double beta = nrm_R;
    printf("GMRES Initial Residual: Norm %e, threshold %e\n", nrm_R, threshold);

    // return 0;

    // setup initial dense matrix data for V the Arnoldi basis
    std::vector<DeviceVec<T>> V;  // create an nvars x m matrix
    for (int i = 0; i < m; i++) {
        V.push_back(DeviceVec<T>(nvars));
    }

    //  and H the householder matrix
    std::vector<double> H((m + 1) * m, 0.0);  // Row-major storage of Hessenberg matrix

    // define givens rotations
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    HostVec<T> g(m+1), cs(m), ss(m);
    std::vector<T> err;
========
    HostVec<T> g(m + 1), cs(m), ss(m);
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    // start Arnoldi process
    // --------------

    // set V[:,0] = r / beta
    a = 1.0 / beta, b = 0.0;  // a * resid + b * tmp1 => into tmp1
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, resid_ptr, 1, V[0].getPtr(), 1));
    // then copy values from tmp1 into V[:,0]
    // (avoids needing cusparse matrix descriptors for V)
    // tmp1.copyValuesTo(V[0]);

    // set g[0] = beta (initial givens rotation)
    g[0] = beta;
    double nrm_debug;
    int jj = m - 1; // j at checkout 
    err.push_back(abs(g[0]));

    // loop over GMRES iterations (later can implement restarts too)
    for (int j = 0; j < m; j++) {
        // zero out temporary vecs
        w.zeroValues();

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, V[j].getPtr(), 1, &nrm_debug))
        if (j == 0 || j == (m - 1)) {
            printf("\t||v[%d]|| = %.4e\n", j, nrm_debug);
        }

        // printf("checkpt4\n");

        // compute w_j = a * A * v_j + b * w_j (v_j is tmp1 into w_j as tmp2)
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
            &a, descrA, kmat_vals, dk_rowp, dk_cols, block_dim, V[j].getPtr(), &b, w_ptr))

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, w_ptr, 1, &nrm_debug))
        if (j == 0 || j == (m - 1)) {
            printf("\t||A*v[%d]|| = %.4e\n", j, nrm_debug);
        }
        // printf("checkpt5\n");

        // compute w_j = U^-1 L^-1 w_j
        // L^-1 * w => tmp1
        a = 1.0;
        tmp1.zeroValues();  // must always zero out the output vec for triangular solve
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             precond_vals, dp_rowp, dp_cols, block_dim, info_L,
                                             w_ptr, tmp1_ptr, policy_L, pBuffer))

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, tmp1_ptr, 1, &nrm_debug))
        if (j == 0 || j == (m - 1)) {
            printf("\t||L^-1 * A*v[%d]|| = %.4e\n", j, nrm_debug);
        }
        if (isnan(nrm_debug)) {
            printf("NaN detected in L^-1 solve!\n");
        }
        // printf("checkpt6\n");

        // U^-1 * tmp1 => w
        a = 1.0;
        w.zeroValues();  // must always zero out the output vec for triangular solve
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             precond_vals, dp_rowp, dp_cols, block_dim, info_U,
                                             tmp1_ptr, w_ptr, policy_U, pBuffer))

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, w_ptr, 1, &nrm_debug))
        if (j == 0 || j == (m - 1)) {
            printf("\t||U^-1 * L^-1 * A*v[%d]|| = %.4e\n", j, nrm_debug);
        }
        if (isnan(nrm_debug)) {
            printf("NaN detected in U^-1 solve!\n");
        }
        // printf("checkpt7\n");

        // auto h_precond_mat = precond_mat.getVec().createHostVec();
        // write_to_csv<int>(rowPtr,nnodes+1, "csv/precond_rowp.csv");
        // write_to_csv<int>(colPtr, rowPtr[nnodes], "csv/precond_cols.csv");
        // write_to_csv<double>(h_precond_mat.getPtr(), h_precond_mat.getSize(),
        // "csv/precond_vals.csv");

        // now we have found wj = M^-1 * A * vj with wj as tmp2
        // recall that v_j is still in temp1 or d_tmp1 from last iteration

        // modified Gram-Schmidt orthogonalization
        for (int i = 0; i < j + 1; i++) {
            // Hij = dot(wj, vi), wj is tmp2 and vi is V[i]
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
            CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, w_ptr, 1, V[i].getPtr(), 1, &H[i * m + j]));
            
            // compute wj = wj - H[i,j] * vi with vi as V[i] and wj as tmp2
            a = -H[i * m + j]; // a * tmp1 + b * tmp2 => into tmp2
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, V[i].getPtr(), 1, w_ptr, 1));

========
            CHECK_CUBLAS(cublasDdot(cublasHandle, m, w_ptr, 1, V[i].getPtr(), 1, &H[i * m + j]));

            // compute wj = wj - H[i,j] * vi with vi as V[i] and wj as tmp2
            a = -H[i * m + j];  // a * tmp1 + b * tmp2 => into tmp2
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, V[i].getPtr(), 1, w_ptr, 1));
            ;
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
        }

        // now check orthogonality of w with previous V[i]
        // is orthogonal
        // for (int i = 0; i < j+1; i++) {

        //     // Hij = dot(wj, vi), wj is tmp2 and vi is V[i]
        //     double dotprod;
        //     CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, w_ptr, 1, V[i].getPtr(), 1, &dotprod));
            
        //     printf("dotprod w with v[%d] = %.4e\n", i, dotprod);

        // }

        // reorthogonalization step
        // for (int i = 0; i < j+1; i++) {
        //     // correction = vi dot w
        //     CHECK_CUBLAS(cublasDdot(cublasHandle, m, tmp2_ptr, 1, tmp1_ptr, 1, &H[i * m + j]));
        // }

        // compute || wj || => into H[j+1,j] (with wj as tmp2)
        double norm_w;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, w_ptr, 1, &norm_w));
        H[(j + 1) * m + j] = norm_w;

        // now compute v_{j+1} still in tmp1 as v_{j+1} = wj / H[j+1,j] with wj as tmp2
        a = 1.0 / H[(j + 1) * m + j], b = 0.0;  // a * tmp2 => tmp2
        CHECK_CUBLAS(cublasDscal(cublasHandle, nvars, &a, w_ptr, 1));

        // then copy tmp1 into V[j+1]
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
        V[j+1].zeroValues();
        w.copyValuesTo(V[j+1]);
========
        w.copyValuesTo(V[j + 1]);
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

        // apply given's rotations
        for (int ii = 0; ii < j; ii++) {
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
            double _temp = cs[ii] * H[ii * m + j] + ss[ii] * H[(ii+1) * m + j];
            H[(ii+1) * m + j] = -ss[ii] * H[ii * m + j] + cs[ii] * H[(ii+1)*m + j];
            H[ii * m + j] = _temp;
========
            double _temp = cs[ii] * H[ii * m + k] + ss[ii] * H[(ii + 1) * m + k];
            H[(ii + 1) * m + k] = -ss[ii] * H[ii * m + k] + cs[ii] * H[(ii + 1) * m + k];
            H[ii * m + k] = _temp;
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
        }

        // compute new givens rotatios on H[j,j] and H[j+1,j] into cs[j], ss[j]
        double _a = H[j * m + j], _b = H[(j + 1) * m + j];

        // from p. 162 of Saad
        double hyp = sqrt(_a * _a + _b * _b);
        cs[j] = _a / hyp;
        ss[j] = _b / hyp;

        // update Hessenberg matrix with given's rotations
        H[j * m + j] = cs[j] * H[j * m + j] + ss[j] * H[(j + 1) * m + j];
        H[(j + 1) * m + j] = 0.0;

        g[j + 1] = -ss[j] * g[j];
        g[j] = cs[j] * g[j];

<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
        printf("j = %d, g[%d] = %.4e\n", j, j+1, g[j+1]);
        // printf("\tpre zero H[%d,%d] = %.4e\n", j+1, j, _b);
        err.push_back(abs(g[j+1]));

        if (abs(g[j+1]) < tolerance) {
            printf("g[%d+1] = %.4e < %.4e so break\n", j, abs(g[j+1]), tolerance);
            jj = j;
========
        printf("j = %d, g[%d] = %.4e\n", j, j + 1, g[j + 1]);
        printf("\tpre zero H[%d,%d] = %.4e\n", j + 1, j, _b);

        if (abs(g[j + 1]) < tolerance) {
            printf("g[%d+1] = %.4e so break\n", j, tolerance);
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
            break;
        }
    }

    // write g[j] Givens rotation history to csv
    // write_to_csv<double>(g.getPtr(), jj+1, "csv/givens_rot.csv");
    write_to_csv<double>(err.data(), err.size(), "csv/givens_rot.csv");

    // extract the (jj+1) x (jj+1) triangular part of H out of it on the host
    // to do this just use only the m^2 values since stored in row-major format
    // copy data onto the device now
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
========
    double *d_H;
    cudaMalloc(&d_H, m * m * sizeof(double));
    cudaMemcpy(d_H, H.data(), m * m * sizeof(double), cudaMemcpyHostToDevice);
    // double check if this part is right by looking at the python GMRES code (H becomes m x m)?
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    // make a smaller Hred matrix of size jj+1 x jj+1 on the host
    std::vector<double> Hred((jj+1)*(jj+1), 0.0);
    for (int row = 0; row < jj+1; row++) {
        for (int col = 0; col < jj+1; col++) {
            // in-place transpose to be compatible with column-major cublasDtrsv later on
            // Hred[row * (jj+1) + col] = H[row * m + col]; // non-transpose copy
            Hred[row * (jj+1) + col] = H[col * m + row];
        }
    }

    // TODO: cleanup later

    // now copy data from Hred to d_Hred
    double *d_Hred;
    cudaMalloc(&d_Hred, (jj+1) * (jj+1) * sizeof(double));
    cudaMemcpy(d_Hred, Hred.data(), (jj+1) * (jj+1) * sizeof(double), cudaMemcpyHostToDevice);
    
    // also create gred
    HostVec<T> gred(jj+1);
    memcpy(gred.getPtr(), g.getPtr(), (jj+1) * sizeof(T));

    auto d_y = gred.createDeviceVec(); // inout so soln temp becomes rhs
    auto d_gcopy = gred.createDeviceVec();

    // auto h_y_pre = d_y.createHostVec();
    // printf("g rhs prelim: ");
    // printVec<double>(jj+1, h_y_pre.getPtr());

    // solve the upper triangular system H * y = g
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    // since d_Hred is stored row-major but this is column major, I need CUBLAS_OP_N not CUBLAS_OP_T
    CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                jj+1, d_Hred, jj+1, d_y.getPtr(), 1));
========
    // d_H was stored row-major use CUBLAS_OP_T to transpose it to column-major (the assumed
    // storage)
    cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, d_H, m,
                d_y, 1);
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    auto h_y = d_y.createHostVec();
    // printf("H * y = g, answer y: ");
    // printVec<double>(jj+1, h_y.getPtr());

    // H matrix
    // printf("H matrix:");
    // printVec<double>((jj+1)*(jj+1), Hred.data());

    // check whether the solution to H * y = g is achieved
    int n = jj + 1;
    auto d_Hy = DeviceVec<T>(jj+1);
    double alpha = 1.0;
    beta = 0.0;
    // here CUBLAS_OP_T because Hred is row major
    cublasDgemv(cublasHandle, CUBLAS_OP_N, n, n, &alpha, d_Hred, n, d_y.getPtr(), 1, &beta, d_Hy.getPtr(), 1);

    // Compute residual: residual = Hy - rhs
    alpha = -1.0;
    // cublasDcopy(handle, n, d_Hy, 1, d_residual, 1);  // Copy Hy to residual
    cublasDaxpy(cublasHandle, n, &alpha, d_gcopy.getPtr(), 1, d_Hy.getPtr(), 1); // residual -= rhs
    
    // Compute the norm of the residual
    double resNorm;
    cublasDnrm2(cublasHandle, n, d_Hy.getPtr(), 1, &resNorm);

    printf("residual of H * y = g system is: %.4e\n", resNorm);

    // update the solution
    // soln = 1.0 * soln + V @ y (do V * y product manually since V stored in vectors)

<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    // so now I update the solution vector with V * y in order for the solution update
    
    for (int j = 0; j < jj+1; j++) {
========
    for (int j = 0; j < m; j++) {
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu
        // copy vj into tmp1
        // V[j].copyValuesTo(tmp1);

        a = h_y[j];  // y[j] * tmp1 + d_soln => into d_soln
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, V[j].getPtr(), 1, soln_ptr, 1));
        
        // auto h_soln = soln.createHostVec();
        // printf("h_soln at step j %d: ", j);
        // printVec<double>(30, h_soln.getPtr());

    }

    // compute residual again
<<<<<<<< HEAD:tests/archive/iterative_GMRES/gmres.cu
    rhs.copyValuesTo(resid); // copies b into resid
========
    loads.copyValuesTo(resid);  // copies b into resid
>>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615:tests/iterative_GMRES/gmres.cu

    // compute resid = - A * x0 + resid
    a = -1.0, b = 1.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  kmat_vals, dk_rowp, dk_cols, block_dim, soln_ptr, &b, resid_ptr));

    // L^-1 * resid => tmp2
    a = 1.0;
    tmp2.zeroValues();  // must always zero out the output vec for triangular solve
    cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, precond_vals,
                          dp_rowp, dp_cols, block_dim, info_L, resid_ptr, tmp2_ptr, policy_L,
                          pBuffer);

    // U^-1 * tmp2 => resid
    a = 1.0;
    resid.zeroValues();  // must always zero out the output vec for triangular solve
    cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, precond_vals,
                          dp_rowp, dp_cols, block_dim, info_U, tmp2_ptr, resid_ptr, policy_U,
                          pBuffer);

    // compute new resid norm
    double final_nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, resid_ptr, 1, &final_nrm_R));
    printf("  Final Residual: Norm %e' threshold %e\n", final_nrm_R, threshold);

    // now we can permute the solution back to get correct answer (so now soln is inv permuted
    // again)
    bsr_post_solve<DeviceVec<T>>(kmat, rhs, soln);

    return 0;
};

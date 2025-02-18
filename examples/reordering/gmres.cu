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

    // preconditioner ILU(0) factorization in cusparse (equiv to ILU(k) factor because of ILU(k) sparsity)
    // implement matrix-vec products (K un-filled in), preconditioner solve M^-1 x
    // write GMRES algorithm (running with data on GPU)

    return 0;
};

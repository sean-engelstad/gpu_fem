#include "../plate_demo/_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "mesh/vtk_writer.h"
#include "shell/shell.h"
#include <iostream>

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
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

    auto bsr_data = assembler.getBsrData();
    int *orig_rowPtr = bsr_data.rowPtr;
    int *orig_colPtr = bsr_data.colPtr;
    int nnodes = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;

    // RCM reordering // based on 1585 of TACSAssembler.cpp
    bool perform_rcm = true;
    int *rowPtr, *colPtr;
    int *perm = new int[nnodes];
    ;
    if (perform_rcm) {
        int *_new_perm = new int[nnodes];
        int root_node = 0;
        int num_rcm_iters = 1;
        TacsComputeRCMOrder(nnodes, orig_rowPtr, orig_colPtr, _new_perm,
                            root_node, num_rcm_iters);
        // also do we need to reverse the order here? (see line 1585 of
        // TACSAssembler.cpp)
        if (perm) {
            for (int k = 0; k < nnodes; k++) {
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
        // su_mat.perm = _new_perm;
        auto su_mat2 =
            SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, perm);

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
    }

    printf("perm1: ");
    printVec<int>(nnodes, perm);

    bool perform_qordering = true;
    if (perform_qordering) {
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
        for (int iprune = 0; iprune < num_prunes; iprune++) {
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
        auto su_mat2 = SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(
            su_mat, q_perm.data());

        nnzb = su_mat2->nnz;
        rowPtr = su_mat2->rowp;
        colPtr = su_mat2->cols;

        // now perform random reordering in q-ordering
    }

    // printf("rowPtr: ");
    // printVec<int>(nnodes + 1, rowPtr);
    // printf("colPtr: ");
    // printVec<int>(nnzb, colPtr);

    // write out matrix sparsity to debug
    write_to_csv<int>(rowPtr, nnodes + 1, "csv/plate_rowPtr.csv");
    write_to_csv<int>(colPtr, nnzb, "csv/plate_colPtr.csv");
    return 0;
};

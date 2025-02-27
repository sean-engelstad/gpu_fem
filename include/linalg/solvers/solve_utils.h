#pragma once
#include "../bsr_mat.h"

template <class Vec> Vec bsr_pre_solve(BsrMat<Vec> mat, Vec rhs, Vec soln) {
    int *perm = mat.getPerm();
    int block_dim = mat.getBlockDim();

    Vec rhs_perm = rhs.createPermuteVec(block_dim, perm);
    return rhs_perm;
}

template <class Vec> void bsr_post_solve(BsrMat<Vec> mat, Vec rhs, Vec soln) {
    int *iperm = mat.getIPerm();
    int block_dim = mat.getBlockDim();
    soln.permuteData(block_dim, iperm);
    return;
}
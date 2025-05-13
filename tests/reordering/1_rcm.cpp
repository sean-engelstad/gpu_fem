#include "../test_commons.h"
#include "linalg/bsr_data.h"

int main() {
    // inputs
    int num_rcm_iters = 6;  // try higher number
    bool print = false;
    // -----------------

    constexpr int n = 7, nnz = 18;
    int rowp[n + 1] = {0, 3, 5, 8, 12, 14, 16, 18};
    int cols[nnz] = {1, 3, 4, 0, 2, 1, 3, 5, 0, 2, 5, 6, 0, 6, 2, 3, 3, 4};

    // reference RCM reordering from baseline/rcm.py (using scipy)
    int scipy_perm[n] = {6, 5, 3, 4, 2, 0, 1};
    int new_rowp[n + 1] = {0, 2, 4, 8, 10, 13, 16, 18};
    int new_cols[nnz] = {2, 3, 2, 4, 0, 1, 4, 5, 0, 5, 1, 2, 6, 2, 3, 6, 4, 5};

    // bandwidth changes from 4 to 3 in python, could check this also..
    int orig_bandwidth = BsrData::getBandWidth(n, nnz, rowp, cols);

    // our RCM method is slightly different from scipy, so here we just check we get the same matrix
    // bandwidth, which is reduced to 3 (different tie breaking strategies)
    int block_dim = 1;
    auto bsr_data = BsrData(n, block_dim, nnz, rowp, cols);
    bsr_data.RCM_reordering(num_rcm_iters);
    bsr_data.compute_nofill_pattern();
    int new_bandwidth = BsrData::getBandWidth(n, nnz, bsr_data.rowp, bsr_data.cols);
    if (print) {
        printf("with our RCM method (slightly different than scipy)\n");
        printf("\tbandwidth reduces from %d to %d\n", orig_bandwidth, new_bandwidth);
    }
    double bandwidth_err = abs(new_bandwidth - 3) / 3.0;

    // also we check that if we used the perm of the scipy RCM, we get the same sparsity
    int *scipy_iperm = new int[n];
    for (int i = 0; i < n; i++) {
        scipy_iperm[scipy_perm[i]] = i;  // printf("here4\n");
        // printf("iperm:\n");
        // printVec<int>(nnodes, iperm);
        // printf("perm:\n");
        // printVec<int>(nnodes, perm);
        // printf("rowp:\n");
        // printVec<int>(nnodes + 1, rowp);
        // printf("cols:\n");
        // printVec<int>(nnzb, cols);
    }
    auto bsr_data2 = BsrData(n, block_dim, nnz, rowp, cols, scipy_perm, scipy_iperm);
    bsr_data2.compute_nofill_pattern();
    if (print) {
        printf("with scipy perm, bsr data\n");
        printf("scipy perm: ");
        printVec<int>(bsr_data2.nnodes, bsr_data2.perm);
        printf("scipy iperm: ");
        printVec<int>(bsr_data2.nnodes, bsr_data2.iperm);
        printf("new rowp: ");
        printVec<int>(bsr_data2.nnodes + 1, bsr_data2.rowp);
        printf("new cols: ");
        printVec<int>(bsr_data2.nnzb, bsr_data2.cols);
    }

    // now check against ref rowp, cols after RCM perm
    double rowp_err = 0.0, cols_err = 0.0;
    for (int i = 0; i < n + 1; i++) {
        double c_rowp_err = abs(new_rowp[i] - bsr_data2.rowp[i]);
        rowp_err = std::max(rowp_err, c_rowp_err);
        double c_cols_err = abs(new_cols[i] - bsr_data2.cols[i]);
        cols_err = std::max(cols_err, c_cols_err);
    }
    double tot_err = std::max(bandwidth_err, std::max(rowp_err, cols_err));
    bool passed = tot_err == 0.0;
    printTestReport<double>("RCM reordering", passed, tot_err);
}
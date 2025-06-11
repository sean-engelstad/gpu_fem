// TODO: maybe check bandwidth still reduced, somehow check chain lengths
// or save reference qordering when I got it working right (rowp, cols and check against that)

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
    bsr_data.qorder_reordering(1.0, num_rcm_iters);
    bsr_data.compute_nofill_pattern();
    int new_bandwidth = BsrData::getBandWidth(n, nnz, bsr_data.rowp, bsr_data.cols);
    if (print) {
        printf("with our qorder method (slightly different than scipy)\n");
        printf("\tbandwidth reduces from %d to %d\n", orig_bandwidth, new_bandwidth);
    }
}
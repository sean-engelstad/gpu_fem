#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

template <typename T>
void BSRtoCSR(int block_dim, int N, int nnzb, int *bsr_rowp, int *bsr_cols, T *bsr_vals,
              int **csr_rowp, int **csr_cols, T **csr_vals, int *nz) {
    assert(N % 2 == 0);

    std::vector<int> rowp, cols;
    std::vector<T> vals;
    rowp.push_back(0);
    int nvals = 0;
    int block_dim2 = block_dim * block_dim;

    for (int row = 0; row < N; row++) {
        int brow = row / block_dim;
        int inner_row = row % block_dim;

        for (int j = bsr_rowp[brow]; j < bsr_rowp[brow + 1]; j++) {
            int bcol = bsr_cols[j];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int col = block_dim * bcol + inner_col;
                cols.push_back(col);
                int val_ind = block_dim2 * j + block_dim * inner_row + inner_col;
                T val = bsr_vals[val_ind];
                vals.push_back(val);
                nvals += 1;
            }
        }

        rowp.push_back(nvals);
    }

    // now deep copy out of it
    *nz = nvals;
    *csr_rowp = new int[N + 1];
    *csr_cols = new int[nvals];
    *csr_vals = new T[nvals];

    for (int i = 0; i < N + 1; i++) {
        (*csr_rowp)[i] = rowp[i];
    }
    for (int i = 0; i < nvals; i++) {
        (*csr_cols)[i] = cols[i];
        (*csr_vals)[i] = vals[i];
    }
}
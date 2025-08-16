#include "include/pde.h"

int main() {
    // make poisson solver and try and run it..

    // int nxe = 2;
    int nxe = 4;
    auto solver = PoissonSolver<float>(nxe);

    // try printing out data structures to check..
    int N = solver.N;
    printf("h_rowp: ");
    int *h_rowp = solver.d_csr_rowp.createHostVec().getPtr();
    printVec<int>(N+1, h_rowp);

    printf("csr_nnz %d\n", (int)solver.csr_nnz);
    printf("h_cols: ");
    int *h_cols = solver.d_csr_cols.createHostVec().getPtr();
    printVec<int>(solver.csr_nnz, h_cols);

    // now printout lhs

    return 0;
};
// SUITE SPARSE
#ifdef SUITE_SPARSE
#include <cholmod.h>
#endif

#ifdef SUITE_SPARSE
__HOST__ void get_fill_in_ssparse(const int &nnodes, int &nnzb, int *&rowPtr,
                                  int *&colPtr, const bool print) {

    if (print) {
        printf("nnodes = %d\n", nnodes);
        printf("orig rowPtr\n");
        printVec<int32_t>(nnodes + 1, rowPtr);
        printf("orig colPtr\n");
        printVec<int32_t>(nnzb, colPtr);
    }

    // define input matrix in CSC format for cholmod
    // printf("do fillin\n");
    cholmod_common c;
    cholmod_start(&c);
    c.print = print; // no print from cholmod

    int n = nnodes;
    int nzmax = nnzb;

    // double Ax[nnzb];
    // memset(Ax, 0.0, nnzb * sizeof(double));
    // temporarily randomize
    HostVec<double> Ax_vec(nnzb);
    Ax_vec.randomize();
    double *Ax = Ax_vec.getPtr();

    int sorted = 1, packed = 1,
        stype = 1, // would prefer to not pack values but have to it seems
        xdtype = CHOLMOD_DOUBLE +
                 CHOLMOD_REAL; // CHOLMOD_PATTERN; // pattern only matrix
    cholmod_sparse *A =
        cholmod_allocate_sparse(n, n, nzmax, sorted, packed, stype, xdtype, &c);
    A->p = rowPtr;
    A->i = colPtr;
    A->x = Ax;
    A->stype = -1; // lower triangular? 1 for upper triangular

    cholmod_factor *L = cholmod_analyze(A, &c); // symbolic factorization
    // TODO : is this slow on CPU?
    // ! TODO : could run this part on the GPU.. at some point and copy it off
    // GPU
    cholmod_factorize(A, L, &c); // numerical factorization

    // now also transpose the L matrix so we can fillin with U sparsity pattern
    // cholmod_sparse *L_sparse = F->L; // L factor (sparse form)
    int *Lp = (int *)L->p;
    int *Li = (int *)L->i;

    cholmod_sparse *L_sparse = cholmod_allocate_sparse(
        n, n, L->nzmax, sorted, packed, stype, xdtype, &c);
    L_sparse->p = Lp;
    L_sparse->i = Li;
    L_sparse->x = L->x;
    L_sparse->stype = -1;

    // Transpose the sparsity pattern of L to get U (upper triangular part)
    int mode = 0; // 0 for pattern only (what I want), 1 for numerical real, 2
                  // for numerical complex
    cholmod_sparse *U_sparse = cholmod_transpose(L_sparse, mode, &c);
    int *Up = (int *)U_sparse->p;
    int *Ui = (int *)U_sparse->i;

    // DEBUG: (seems to work now though)
    if (print) {
        printf("fillin L rowPtr\n");
        printVec<int32_t>(nnodes + 1, Lp);
        printf("fillin L colPtr\n");
        printVec<int32_t>(L->nzmax, Li);

        printf("fillin U rowPtr\n");
        printVec<int32_t>(nnodes + 1, Up);
        printf("fillin U colPtr\n");
        printVec<int32_t>(L->nzmax, Ui);
    }

    // TODO : debug here and print out L->p, L->i? to see if fill-in worked?
    // is this code efficient?

    // now update rowPtr, colPtr for each row with fill-in values
    nnzb = 0; // reset nnzb
    std::vector<int> _rowPtr(nnodes + 1, 0);
    std::vector<int> _colPtr;

    for (int inode = 0; inode < nnodes; inode++) {
        std::vector<int32_t>
            temp; // put all colPtr vals from orig and fill-in into this

        // original sparsity
        for (int icol = rowPtr[inode]; icol < rowPtr[inode + 1]; icol++) {
            temp.push_back(colPtr[icol]);
        }

        // lower triangular fillin
        for (int icol = Lp[inode]; icol < Lp[inode + 1]; icol++) {
            temp.push_back(Li[icol]);
        }

        // upper triangular fillin
        for (int icol = Up[inode]; icol < Up[inode + 1]; icol++) {
            temp.push_back(Ui[icol]);
        }

        // now make unique list of columns for this row / node
        std::sort(temp.begin(), temp.end());
        auto last = std::unique(temp.begin(), temp.end());
        temp.erase(last, temp.end());

        // show temp on each row
        // printf("row %d: ", inode);
        // printVec<int32_t>(temp.size(), temp.data());

        // add into new colPtr, rowPtr, nnzb
        nnzb += temp.size();
        _colPtr.insert(_colPtr.end(), temp.begin(), temp.end());
        _rowPtr[inode + 1] = nnzb;
    }

    // copy data to output pointers (deep copy)
    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new int[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);

    if (print) {
        printf("final rowPtr\n");
        printVec<int32_t>(nnodes + 1, rowPtr);
        printf("final colPtr\n");
        printVec<int32_t>(nnzb, colPtr);
    }
}
#endif
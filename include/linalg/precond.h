#include "bsr_mat.h"

template <class Vec_>
class Preconditioner
{
public:
    using Vec = typename Vec_;
    using T = typename Vec::type;

    Preconditioner(BSRMat &mat, int lev_fill, double fill, bool print) : mat(mat), lev_fill(lev_fill), fill(fill), print(print)
    {
        // need to new sparsity for pmat the preconditioner matrix (based on ILU(k) and a reordering)
#ifdef USE_GPU
        // convert back to host pointer data then do ILU(k) and put back on device
        auto d_bsr_data = mat.getBsrData();
        auto h_bsr_data = d_bsr_data.createHostBsrData();
        h_bsr_data.symbolic_iluk(lev_fill, fill, print);
        auto bsr_data = h_bsr_data.createDeviceBsrData();
#else
        auto bsr_data = mat.getBsrData();
        bsr_data.symbolic_iluk(lev_fill, fill, print);
#endif
        // make a new matrix
        auto precond_bsr_data = bsr_data;
        precond_mat = new BSRMat(bsr_data);
    }

    void factor()
    {
        mat.copyValues(precond_mat);

        // call ILU(0) factorization in cusparse probably..
        // on ILU(k) nz's so that the numeric is ILU(k) factorization
        precond_mat.numeric_ilu0();
    }

private:
    Mat &mat, &precond_mat;
    int lev_fill;
    double fill;
    bool print;
};
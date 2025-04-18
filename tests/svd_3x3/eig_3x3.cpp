#include "_test_utils.h"
#include "a2dcore.h"
#include "linalg/svd_utils.h"
#include "utils.h"

int main() {
    double H[9] = {1.0813e-03,  1.9075e-04, 7.4367e-06,  1.9075e-04, 1.0813e-03,
                   -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07};

    double A[9];
    A2D::MatMatMultCore3x3<double, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);

    double sigsq[3], VT[9];
    eig3x3_givens<double, 15>(A, sigsq, VT);

    printf("H:");
    printVec<double>(3, H);
    printVec<double>(3, &H[3]);
    printVec<double>(3, &H[6]);
    printf("sigsq:");
    printVec<double>(3, sigsq);
    printf("VT:");
    printVec<double>(3, VT);
    printVec<double>(3, &VT[3]);
    printVec<double>(3, &VT[6]);

    // compute error in each one
    double sig_ref[3] = {1.61811120e-06, 7.93352971e-07, 2.94572739e-17};
    double VT_ref[9] = {-7.07106781e-01, -7.07106781e-01, 1.73472348e-18,
                        7.07057476e-01,  -7.07057476e-01, 1.18089119e-02,
                        -8.35016167e-03, 8.35016167e-03,  9.99930272e-01};

    double sig_err = rel_err<3>(sigsq, sig_ref);
    double VT_err = abs_err<9>(VT, VT_ref);

    printf("\n\nsig_rel_err: %.4e, VT_err: %.4e\n", sig_err, VT_err);

    return 0;
}
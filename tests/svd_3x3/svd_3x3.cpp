#include "_test_utils.h"
#include "a2dcore.h"
#include "linalg/svd_utils.h"
#include "utils.h"

int main() {
    double H[9] = {1.0813e-03,  1.9075e-04, 7.4367e-06,  1.9075e-04, 1.0813e-03,
                   -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07};

    double U[9], sig[3], VT[9];
    // svd3x3_cubic(H, sig, U, VT);
    // svd3x3_givens(H, sig, U, VT);
    svd3x3_QR(H, sig, U, VT);

    printf("H:");
    printVec<double>(3, H);
    printVec<double>(3, &H[3]);
    printVec<double>(3, &H[6]);
    printf("U:");
    printVec<double>(3, U);
    printVec<double>(3, &U[3]);
    printVec<double>(3, &U[6]);
    printf("sig:");
    printVec<double>(3, sig);
    printf("VT:");
    printVec<double>(3, VT);
    printVec<double>(3, &VT[3]);
    printVec<double>(3, &VT[6]);

    // also compute R = U * VT
    double R[9];
    A2D::MatMatMultCore3x3<double>(U, VT, R);

    printf("R:");
    printVec<double>(3, R);
    printVec<double>(3, &R[3]);
    printVec<double>(3, &R[6]);

    // compute error in each one
    double Uref[9] = {-7.07106781e-01, 7.07034111e-01,  -1.01373504e-02,
                      -7.07106781e-01, -7.07034111e-01, 1.01373504e-02,
                      6.07153217e-18,  1.43363785e-02,  9.99897229e-01};
    double sig_ref[3] = {1.27205000e-03, 8.90703638e-04, 5.42745557e-09};
    double VT_ref[9] = {-7.07106781e-01, -7.07106781e-01, 1.73472348e-18,
                        7.07057476e-01,  -7.07057476e-01, 1.18089119e-02,
                        -8.35016167e-03, 8.35016167e-03,  9.99930272e-01};
    double R_ref[9] = {9.99998403e-01, 1.59729481e-06,  -1.78734007e-03,
                       1.59729481e-06, 9.99998403e-01,  1.78734007e-03,
                       1.78734007e-03, -1.78734007e-03, 9.99996805e-01};

    double sig_err = rel_err<3>(sig, sig_ref);
    double R_err = abs_err<9>(R, R_ref);

    printf("\n\nsig_rel_err: %.4e, R_abs_err: %.4e\n", sig_err, R_err);

    return 0;
}
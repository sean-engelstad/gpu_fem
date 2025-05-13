#include "../test_commons.h"
#include "a2dcore.h"
#include "linalg/svd_utils.h"
#include "utils.h"

int main() {
    /* test H => R computation for an aero node that should give R = I */
    double H[9] = {3.163269487e-02,  1.265915030e-02, -1.074607003e-02,
                   1.265915030e-02,  3.525505498e-02, 1.248878091e-02,
                   -1.074607003e-02, 1.248878091e-02, 3.575272340e-02};

    bool print = true;

    double U[9], sig[3], VT[9];
    // svd3x3_cubic(H, sig, U, VT);
    // svd3x3_givens(H, sig, U, VT);
    svd3x3_QR(H, sig, U, VT);

    if (print) {
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
    }

    // also compute R = U * VT
    double R[9];
    A2D::MatMatMultCore3x3<double>(U, VT, R);

    if (print) {
        printf("R:");
        printVec<double>(3, R);
        printVec<double>(3, &R[3]);
        printVec<double>(3, &R[6]);
    }

    // compute error in each one
    double Uref[9] = {0.20104624, 0.76493609, -0.61192581, 0.7737372,  0.25910147,
                      0.57809789, 0.60075882, -0.58969417, -0.53976813};
    double sig_ref[3] = {0.04824122, 0.0442049, 0.01019438};
    double VT_ref[9] = {0.20104624,  0.7737372,   0.60075882, 0.76493609, 0.25910147,
                        -0.58969417, -0.61192581, 0.57809789, -0.53976813};
    double R_ref[9] = {1.00000000e+00,  -4.89266043e-17, -3.28841603e-17,
                       1.05180589e-16,  1.00000000e+00,  2.84699152e-16,
                       -1.43906463e-16, 6.70696966e-17,  1.00000000e+00};

    double sig_err = rel_err(3, sig, sig_ref);
    double R_err = abs_err(9, R, R_ref);
    double max_rel_err = std::max(sig_err, R_err);
    bool passed = max_rel_err < 1e-8;
    printTestReport("svd 3x3 test", passed, max_rel_err);
    printf("\tsig_rel_err %.4e, R_abs_err %.4e\n", sig_err, R_err);
    return 0;
}
#include "../test_commons.h"
#include "a2dcore.h"
#include "adscalar.h"
#include "linalg/svd_utils.h"
#include "utils.h"

int main() {
    using T = double;
    constexpr bool givens = true;
    bool print = true;

    T H[9] = {1.0813e-03,  1.9075e-04, 7.4367e-06,  1.9075e-04, 1.0813e-03,
              -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07};

    double A[9];
    A2D::MatMatMultCore3x3<double, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);

    // divide A by 1e-6 to scale for derivatives easier
    for (int i = 0; i < 9; i++) {
        A[i] /= 1e-6;
    }

    if (print) {
        printf("A: ");
        printVec<T>(9, A);
        printf("---------------\n");
    }

    double Atmp[9];
    for (int i = 0; i < 9; i++) {
        Atmp[i] = A[i];
    }

    T sig[3], VT[9];
    if constexpr (givens) {
        eig3x3_givens<T, 45>(Atmp, sig, VT);
    } else {
        eig3x3_cubic<T>(Atmp, sig, VT);
    }

    if (print) printf("---------------\n");

    // check R jacobian dR/dH
    // forward AD version
    using T2 = A2D::ADScalar<T, 1>;
    T2 A2[9];
    for (int i = 0; i < 9; i++) {
        A2[i].value = A[i];
        A2[i].deriv[0] = (1.0 + 2.0 * i + 3.134243 * sqrt(i * i));
    }

    T2 VT2[9], sig2[3];
    if constexpr (givens) {
        eig3x3_givens<T2, 45>(A2, sig2, VT2);
    } else {
        eig3x3_cubic<T2>(A2, sig2, VT2);
    }

    // debug return early
    // return 0;

    if (print) {
        printf("sig: ");
        printVec<T>(3, sig);

        printf("VT: ");
        printVec<T>(3, VT);
        printVec<T>(3, &VT[3]);
        printVec<T>(3, &VT[6]);

        printf("sig2: ");
        printVec<T2>(3, sig2);
    }

    T outv[9], fAD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        outv[i] = i * sqrt(i) * 1.2432;
        fAD_deriv += outv[i] * VT2[i].deriv[0];
    }

    // finite difference
    T Ap[9], VTp[9];
    T h = 1e-5;
    for (int i = 0; i < 9; i++) {
        Ap[i] = A[i] + h * A2[i].deriv[0];
    }

    if constexpr (givens) {
        eig3x3_givens<T, 45>(Ap, sig, VTp);
    } else {
        eig3x3_cubic<T>(Ap, sig, VTp);
    }
    T finD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        finD_deriv += outv[i] * (VTp[i] - VT[i]) / h;
    }

    if (print) {
        printf("VT-AD: ");
        printVec<T2>(3, VT2);
        printVec<T2>(3, &VT2[3]);
        printVec<T2>(3, &VT2[6]);
    }

    // compute finite diff version
    T VT_FD[9];
    for (int i = 0; i < 9; i++) {
        VT_FD[i] = (VTp[i] - VT[i]) / h;
    }
    if (print) {
        printf("VT-FD: ");
        printVec<T>(3, VT_FD);
        printVec<T>(3, &VT_FD[3]);
        printVec<T>(3, &VT_FD[6]);
    }

    double rel_err = abs(fAD_deriv - finD_deriv) / abs(finD_deriv);
    bool passed = rel_err < 0.01;
    printTestReport("eig3x3 jacobian", passed, rel_err);
    printf("\tfAD_deriv = %.4e, finD_deriv = %.4e\n", fAD_deriv, finD_deriv);

    return 0;
}
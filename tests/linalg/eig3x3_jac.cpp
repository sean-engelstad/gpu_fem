#include "../test_commons.h"
#include "a2dcore.h"
#include "adscalar.h"
#include "linalg/svd_utils.h"
#include "utils.h"

// check a rotation invariant metric
template <typename T>
void rot_inv_matrix(T VT[9], T M[9]) {
    // M = V^T * D * V where D is defined below
    T D[9];
    // for (int i = 0; i < 9; i++) {
    //     D[i] = 0.134 + 0.257 * i;
    //     if constexpr (std::is_same_v<T, A2D::ADScalar<double, 1>>) {
    //         D[i].deriv[0] = 0.0;
    //     }
    // }
    // make sigma here (so reconstructs A for temp test)
    for (int i = 0; i < 9; i++) D[i] = 0.0;
    D[0] = 1.61811e0;
    D[4] = 7.93353e-1;
    D[8] = 2.94573e-11;
    T tmp[9];

    // M = V^T * D * V
    A2D::MatMatMultCore3x3<T>(VT, D, tmp);
    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(tmp, VT, M);
}

int main() {
    // inputs

    using T = double;
    constexpr bool givens = true;
    bool print = true;
    constexpr bool can_swap = false;

    T h = 1e-6;

    // ----------------------------

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

    T sig[3], VT[9], M[9];
    if constexpr (givens) {
        eig3x3_exact_givens<T, 45, true, can_swap>(Atmp, sig, VT);
    } else {
        eig3x3_cubic<T>(Atmp, sig, VT);
    }
    rot_inv_matrix<T>(VT, M);

    if (print) printf("---------------\n");

    // check R jacobian dR/dH
    // forward AD version
    using T2 = A2D::ADScalar<T, 1>;
    T2 A2[9];
    for (int i = 0; i < 9; i++) {
        A2[i].value = A[i];
        A2[i].deriv[0] = (1.0 + 2.0 * i + 3.134243 * sqrt(i * i));
    }

    T2 VT2[9], sig2[3], M2[9];
    if constexpr (givens) {
        eig3x3_exact_givens<T2, 45, true, can_swap>(A2, sig2, VT2);
    } else {
        eig3x3_cubic<T2>(A2, sig2, VT2);
    }
    rot_inv_matrix<T2>(VT2, M2);

    // debug return early
    // return 0;

    if (print) {
        printf("sig: ");
        printVec<T>(3, sig);

        printf("VT: ");
        printVec<T>(3, VT);
        printVec<T>(3, &VT[3]);
        printVec<T>(3, &VT[6]);

        printf("M0: ");
        printVec<T>(3, M);
        printVec<T>(3, &M[3]);
        printVec<T>(3, &M[6]);

        printf("sig2: ");
        printVec<T2>(3, sig2);
    }

    T outv[9], fAD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        outv[i] = i * sqrt(i) * 1.2432;
        fAD_deriv += outv[i] * M2[i].deriv[0];
    }

    // finite difference
    T Ap[9], VTp[9], Mp[9];
    for (int i = 0; i < 9; i++) {
        Ap[i] = A[i] + h * A2[i].deriv[0];
    }

    if constexpr (givens) {
        eig3x3_exact_givens<T, 45, true, can_swap>(Ap, sig, VTp);
    } else {
        eig3x3_cubic<T>(Ap, sig, VTp);
    }
    rot_inv_matrix<T>(VTp, Mp);

    T finD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        finD_deriv += outv[i] * (Mp[i] - M[i]) / h;
    }

    if (print) {
        printf("M-AD: ");
        printVec<T2>(3, M2);
        printVec<T2>(3, &M2[3]);
        printVec<T2>(3, &M2[6]);
    }

    // compute finite diff version
    T M_FD[9];
    for (int i = 0; i < 9; i++) {
        M_FD[i] = (Mp[i] - M[i]) / h;
    }
    if (print) {
        printf("Mp: ");
        printVec<T>(3, Mp);
        printVec<T>(3, &Mp[3]);
        printVec<T>(3, &Mp[6]);

        printf("M-FD: ");
        printVec<T>(3, M_FD);
        printVec<T>(3, &M_FD[3]);
        printVec<T>(3, &M_FD[6]);
    }

    double rel_err = abs(fAD_deriv - finD_deriv) / abs(finD_deriv);
    bool passed = rel_err < 0.01;
    printTestReport("eig3x3 jacobian", passed, rel_err);
    printf("\tfAD_deriv = %.4e, finD_deriv = %.4e\n", fAD_deriv, finD_deriv);

    return 0;
}
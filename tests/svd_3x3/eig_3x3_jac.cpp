#include "_test_utils.h"
#include "a2dcore.h"
#include "linalg/svd_utils.h"
#include "utils.h"
#include "adscalar.h"

int main() {
    using T = double;
    T H[9] = {1.0813e-03,  1.9075e-04, 7.4367e-06,  1.9075e-04, 1.0813e-03,
                   -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07};

    
    double A[9];
    A2D::MatMatMultCore3x3<double, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);

    T sig[3], VT[9];
    eig3x3_givens<T>(A, VT);

    printf("VT:");
    printVec<T>(3, VT);
    printVec<T>(3, &VT[3]);
    printVec<T>(3, &VT[6]);

    // check R jacobian dR/dH
    // forward AD version
    using T2 = A2D::ADScalar<T,1>;
    T2 A2[9];
    for (int i = 0; i < 9; i++) {
        A2[i].value = A[i];
        A2[i].deriv[0] = 1.0 + 2.0 * i + 3.134243 * i * i;
    }

    T2 VT2[9];
    eig3x3_givens<T2>(A2, VT2);

    T outv[9], fAD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        outv[i] = i * sqrt(i) * 1.2432;
        fAD_deriv += outv[i] * VT2[i].deriv[0];
    }

    // finite difference
    T Ap[9], VTp[9];
    T h = 1e-6;
    for (int i = 0; i < 9 ; i++) {
        Ap[i] = A[i] + h * A2[i].deriv[0];
    }

    computeRotation<T>(Ap, VTp);
    T finD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        finD_deriv += outv[i] * (VTp[i] - VT[i]) / h;
    }

    printf("VT2 value:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            printf("%.4e,", VT2[3*row+col].value);
        }
        printf("\n");
    }

    printf("VT2.deriv[0]:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            printf("%.4e,", VT2[3*row+col].deriv[0]);
        }
        printf("\n");
    }

    printf("VTp:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            T fd = (VTp[3*row+col] - VT[3*row+col]) / h;
            printf("%.4e,", fd);
        }
        printf("\n");
    }

    printf("fAD_deriv = %.4e, finD_deriv = %.4e\n", fAD_deriv, finD_deriv);

    return 0;
}
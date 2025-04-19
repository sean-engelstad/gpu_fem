#include "_test_utils.h"
#include "a2dcore.h"
#include "adscalar.h"
#include "linalg/svd_utils.h"
#include "utils.h"

int main() {
    using T = double;
    T H[9] = {1.0813e-03,  1.9075e-04, 7.4367e-06,  1.9075e-04, 1.0813e-03,
              -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07};

    // rescale H so finite difference approx better
    for (int i = 0; i < 9; i++) {
        H[i] /= 1e-3;
    }

    T R[9];
    computeRotation<T>(H, R);

    printf("R:");
    printVec<T>(3, R);
    printVec<T>(3, &R[3]);
    printVec<T>(3, &R[6]);

    // check R jacobian dR/dH
    // forward AD version
    using T2 = A2D::ADScalar<T, 1>;
    T2 H2[9];
    for (int i = 0; i < 9; i++) {
        H2[i].value = H[i];
        H2[i].deriv[0] = 1.0 + 2.0 * i + 3.134243 * i * i;
    }

    T2 R2[9];
    computeRotation<T2>(H2, R2);

    T outv[9], fAD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        outv[i] = i * sqrt(i) * 1.2432;
        fAD_deriv += outv[i] * R2[i].deriv[0];
    }

    // finite difference
    T Hp[9], Rp[9];
    T h = 1e-6;
    for (int i = 0; i < 9; i++) {
        Hp[i] = H[i] + h * H2[i].deriv[0];
    }

    computeRotation<T>(Hp, Rp);
    T finD_deriv = 0.0;
    for (int i = 0; i < 9; i++) {
        finD_deriv += outv[i] * (Rp[i] - R[i]) / h;
    }

    printf("R2 value:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            printf("%.4e,", R2[3 * row + col].value);
        }
        printf("\n");
    }

    printf("R2.deriv[0]:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            printf("%.4e,", R2[3 * row + col].deriv[0]);
        }
        printf("\n");
    }

    printf("Rp:");
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            T fd = (Rp[3 * row + col] - R[3 * row + col]) / h;
            printf("%.4e,", fd);
        }
        printf("\n");
    }

    printf("fAD_deriv = %.4e, finD_deriv = %.4e\n", fAD_deriv, finD_deriv);

    return 0;
}
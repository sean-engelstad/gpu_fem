/**
goal of this test is to check numerical stability of Givens rotations (to improve Givens eigen
strategy) and their forward AD vs Finite diff

This is because the accuracy of the eig3x3_exact_givens is only as accurate as the Givens rotation
step I believe
 */

#include "a2dcore.h"
#include "adscalar.h"

template <typename T>
T givens_constraints(T a11, T a12, T a22, T &c, T &s) {
    /* checks the constraints for Givens rotation and sine vs cosine */
    // diagonalization constraint for 2x2 matrix off-diagonal = 0
    T off_diag = (a22 - a11) * c * s + a12 * (c * c - s * s);
    // rot matrix / trig constraint
    T rot = c * c + s * s - 1;
    // printf("off_diag %.4e, rot %.4e\n", off_diag, rot);
    // total resid
    T resid = sqrt(off_diag * off_diag + rot * rot);
    return resid;
}

template <typename T, int givens = 1>
void givens_rot(T a11, T a12, T a22, T &c, T &s) {
    if constexpr (givens == 1) {
        // original / method 1, exactly solves the Givens rotation angles
        // but sine, cosine, tangent a bit expensive in device code (not too bad though), but could
        // be faster
        T th = 0.5 * atan(2.0 * a12 / (a11 - a22 + 1e-14));
        s = sin(th);
        c = cos(th);
    } else if (givens == 2) {
        // derived from Wilkinson method for computing Givens rotations
        if (fabs(a12) < 1e-20) {
            c = 1.0;
            s = 0.0;
            return;
        }

        T tau = (a22 - a11) / (2.0 * a12);
        T t = ((tau >= 0.0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        c = 1.0 / sqrt(1.0 + t * t);
        s = t * c;
    } else if (givens == 3) {
        // try using atan2 on device
        T th = 0.5 * atan2(T(2.0) * a12 + 1e-14, a11 - a22 + 1e-14);
        s = sin(th);
        c = cos(th);
    } else if (givens == 4) {
        T d = sqrt((a11 - a22) * (a11 - a22) + T(4.0) * a12 * a12);
        T th_good = T(0.5) * atan2(T(2.0) * a12, a11 - a22);
        T th_safe = T(0.25) * M_PI;  // arbitrary, e.g. 45 degrees
        T th = (d < 1e-4) ? th_safe : th_good;

        c = cos(th);
        s = sin(th);
    }
}

template <typename T, int givens = 1>
void test_resid(T a11, T a12, T a22) {
    T c, s;
    givens_rot<T, givens>(a11, a12, a22, c, s);
    T resid = givens_constraints(a11, a12, a22, c, s);
    printf("resid for A=[%.4e,%.4e;%.4e,%.4e] => %.4e\n", a11, a12, a12, a22, resid);
}

template <typename T, int givens = 1>
void test_forward_AD(T a11, T a12, T a22) {
    // eval with forward AD
    using T2 = A2D::ADScalar<T, 1>;
    T2 c2, s2;
    T2 a11_ = a11, a12_ = a12, a22_ = a22;
    a11_.deriv[0] = 1.0;
    a12_.deriv[0] = 0.5;
    a22_.deriv[0] = 0.78;
    givens_rot<T2, givens>(a11_, a12_, a22_, c2, s2);

    // compare to finite difference
    T c0, s0, cp, sp;
    givens_rot<T, givens>(a11, a12, a22, c0, s0);
    T h = 1e-6;
    givens_rot<T, givens>(a11 + a11_.deriv[0] * h, a12 + a12_.deriv[0] * h, a22 + a22_.deriv[0] * h,
                          cp, sp);
    T cd = (cp - c0) / h;
    T sd = (sp - s0) / h;
    printf("\nforward AD test with A = [%.4e,%.4e;%.4e,%.4e]\n", a11, a12, a12, a22);
    printf("\tforward AD c %.4e, s %.4e\n", c2.deriv[0], s2.deriv[0]);
    printf("\tfinite diff c %.4e, s %.4e\n", cd, sd);
}

int main() {
    // which type of givens computation
    // constexpr int givens = 1; // original, unstable derivatives near diagonal matrix
    // constexpr int givens = 2;
    // constexpr int givens = 3;
    constexpr int givens = 4;

    printf("using givens method %d\n", givens);

    // check residuals
    test_resid<double, givens>(1.0, 0.1, 0.5);
    test_resid<double, givens>(1.0, 0.1, 1.0);
    test_resid<double, givens>(1.0, 1e-8, 1.0);

    // check forward AD
    test_forward_AD<double, givens>(1.0, 0.1, 0.9);
    test_forward_AD<double, givens>(1.0, 1e-8, 1.0 + 1e-7);
}
#pragma once
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
#include "a2dcore.h"
#include "cuda_utils.h"
#include "math.h"

// perform SVD on 3x3 covariance matrix H
// need to implement myself since can't call from inside the __global__ other cusparse, cublas
// methods, etc. was going to try this one: jacobi SVD
// https://netlib.org/lapack/lawnspdf/lawn170.pdf which is fast for small matrices this is a nice
// one too, but a lot of steps: https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf decided to
// use this exact solution for eigenvalues of a 3x3 matrix,
// https://dl.acm.org/doi/pdf/10.1145/355578.366316

template <typename T>
__HOST_DEVICE__ void eig3x3(const T A[9], T sigma[3], T VT[9]) {
    // 3x3 analytic eigenvalue solve using cubic formula style approach

    double pi = 3.141592653589723846;

    // exact eigenvalues of a 3x3 matrix from this source,
    // https://dl.acm.org/doi/pdf/10.1145/355578.366316 using cubic equation formula (Cardano's),
    // uses invariants => trace(A), det(A) and the squared one

    // m = trace(A)/3
    T m = (A[0] + A[4] + A[8]) / 3.0;

    // p = 1/6 * sum square elements of A-mI
    // q = 0.5 * det(A - mI)
    T p, q;
    {
        T AmI[9];
        for (int i = 0; i < 9; i++) {
            AmI[i] = A[i];
        }
        AmI[0] -= m;
        AmI[4] -= m;
        AmI[8] -= m;

        q = 0.5 * A2D::MatDetCore<T, 3>(AmI);

        p = 0.0;
        for (int i = 0; i < 9; i++) {
            p += AmI[i] * AmI[i];
        }
        p /= 6.0;
    }

    // printf("m %.4e p %.4e q %.4e\n", m, p, q);

    // we now have our three invariants m, p, q
    // we also need this angle from trigonomery phi
    T phi = atan(sqrt(p * p * p - q * q) / q) / 3.0;
    // ensures phi is in [0,pi] and ? symbol prevents warp divergence
    phi += phi < 0.0 ? pi : 0.0;

    // then the three eigenvalues of A are sigma^2 eigvals
    sigma[0] = m + 2 * sqrt(p) * cos(phi);
    sigma[1] = m - sqrt(p) * (cos(phi) + sqrt(3.0) * sin(phi));
    sigma[2] = m - sqrt(p) * (cos(phi) - sqrt(3.0) * sin(phi));

    // now that we have S diag matrix, how do we get V and U?
    // for V we can solve the system (A - sigma[i] * I) * vi = 0 for each S[i]
    // one cool way to find each vi is to take the cross product of the first two rows of Bi = A -
    // S[i] * I
    {  // scope block here
        for (int ieig = 0; ieig < 3; ieig++) {
            // compute B_i = A - lam_i *I
            T B[9];
            for (int i = 0; i < 9; i++) {
                B[i] = A[i];
            }
            B[0] -= sigma[ieig];
            B[4] -= sigma[ieig];
            B[8] -= sigma[ieig];

            // now take cross product of each pair of rows and add them to get null space vector
            T c[3], tmp[3];
            for (int i = 0; i < 3; i++) {
                c[i] = 0.0;
            }
            A2D::VecCrossCore<T>(&B[0], &B[3], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            A2D::VecCrossCore<T>(&B[0], &B[6], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            A2D::VecCrossCore<T>(&B[3], &B[6], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            // normalize c
            T cnorm = sqrt(A2D::VecDotCore<T, 3>(c, c));  //  + 1e-12
            A2D::VecScaleCore<T, 3>(1.0 / cnorm, c, &VT[3 * ieig]);
        }

        // now need to apply Graham-Schmidt process on the V matrix
        // first v2 -= <v1,v2> v1
        T v12 = A2D::VecDotCore<T, 3>(&VT[0], &VT[3]);
        A2D::VecAddCore<T, 3>(-v12, &VT[0], &VT[3]);

        // normalize v2
        T norm2 = sqrt(A2D::VecDotCore<T, 3>(&VT[3], &VT[3])) + 1e-12;
        A2D::VecScaleCore<T, 3>(1.0 / norm2, &VT[3], &VT[3]);

        // then v3 -= <v1,v3> v1
        T v13 = A2D::VecDotCore<T, 3>(&VT[0], &VT[6]);
        A2D::VecAddCore<T, 3>(-v13, &VT[0], &VT[6]);

        // v3 -= <v2,v3> v2
        T v23 = A2D::VecDotCore<T, 3>(&VT[3], &VT[6]);
        A2D::VecAddCore<T, 3>(-v23, &VT[3], &VT[6]);

        // normalize v3
        T norm3 = sqrt(A2D::VecDotCore<T, 3>(&VT[6], &VT[6])) + 1e-12;
        A2D::VecScaleCore<T, 3>(1.0 / norm3, &VT[6], &VT[6]);
    }  // end of scope block for getting VT
}

template <typename T>
__HOST_DEVICE__ void svd3x3(const T H[9], T sigma[3], T U[9], T VT[9], const bool print = false) {
    // so are given H a 3x3 matrix and we wish to find H = U * Sigma * V^T

    // now call the 3x3 eigenvalue problem on A = H^T H = V * Sigma^2 * V^T
    T A[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);
    eig3x3<T>(A, sigma, VT);

    // now call the 3x3 eigenvalue problem on A = H H^T = U * Sigma^2 * U^T
    // technically this writes into sigma again, that's fine
    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(H, H, A);
    T UT[9];
    eig3x3<T>(A, sigma, UT);
    // un-transpose UT => U
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[3 * i + j] = UT[3 * j + i];
        }
    }

    // change sigma^0.5 => sigma because we had sigma^2 before
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrt(sigma[i]);
    }

    if (print) {
        printf("rt-sigmas: [%.4e, %.4e, %.4e]\n", sigma[0], sigma[1], sigma[2]);
    }

    if (print) {
        printf("VT-GS: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", VT[0], VT[1],
               VT[2], VT[3], VT[4], VT[5], VT[6], VT[7], VT[8]);
    }

    if (print) {
        printf("U: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", U[0], U[1], U[2],
               U[3], U[4], U[5], U[6], U[7], U[8]);
    }

    // now we have completed the SVD
}

template <typename T>
__HOST_DEVICE__ void computeRotation(const T H[9], T R[9], const bool print = false) {
    T sigma[3], U[9], VT[9];
    svd3x3<T>(H, sigma, U, VT, print);

    // compute rotation matrix R = U * VT
    A2D::MatMatMultCore3x3<T>(U, VT, R);
}

template <typename T>
__HOST_DEVICE__ void computeRotation(const T H[9], T R[9], T S[9]) {
    T sigma[3], U[9], VT[9];
    svd3x3<T>(H, sigma, U, VT);

    // compute rotation matrix R = U * VT
    A2D::MatMatMultCore3x3<T>(U, VT, R);

    // compute symmetric matrix S = V * Sigma * VT
    T tmp[9];
    for (int i = 0; i < 3; i++) {
        A2D::VecScaleCore<T, 3>(sigma[i], &VT[3 * i], &tmp[3 * i]);
    }

    // S = V * tmp
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(VT, tmp, S);
}

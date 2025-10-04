#pragma once
#include "../../../cuda_utils.h"

template <typename T>
__HOST_DEVICE__ void assembleFrame(const T a[], const T b[], const T c[], T frame[]) {
    for (int i = 0; i < 3; i++) {
        frame[3 * i] = a[i];
        frame[3 * i + 1] = b[i];
        frame[3 * i + 2] = c[i];
    }
}

// __HOST_DEVICE__ void extractFrame(const T frame[], T a[], T b[], T c[]) {
//     for (int i = 0; i < 3; i++) {
//         a[i] = frame[3 * i];
//         b[i] = frame[3 * i + 1];
//         c[i] = frame[3 * i + 2];
//     }
// }

template <typename T, class Data>
__HOST_DEVICE__ void ShellComputeTransform(const T refAxis[], const T dXdxi[], const T dXdeta[],
                                           const T n0[], T Tmat[]) {
    A2D::Vec<T, 3> n(n0), nhat;
    A2D::VecNormalize(n, nhat);

    A2D::Vec<T, 3> t1;
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        t1 = A2D::Vec<T, 3>(refAxis);
    } else {  // doesn't have ref axis
        // shell natural transform
        t1 = A2D::Vec<T, 3>(dXdxi);
    }

    // remove normal component from t1
    A2D::Vec<T, 3> temp, t1hat;
    T d = A2D::VecDotCore<T, 3>(nhat.get_data(), t1.get_data());
    A2D::VecSum(T(1.0), t1, -d, nhat, temp);
    A2D::VecNormalize(temp, t1hat);

    // compute t2 by cross product of normalized unit vectors
    A2D::Vec<T, 3> t2hat;
    A2D::VecCross(nhat, t1hat, t2hat);

    // save values in Tmat 3x3 matrix
    for (int i = 0; i < 3; i++) {
        Tmat[3 * i] = t1hat[i];
        Tmat[3 * i + 1] = t2hat[i];
        Tmat[3 * i + 2] = nhat[i];
    }
}

template <typename T, class Basis>
__HOST_DEVICE__ T getDetXd(const T pt[], const T xpts[], const T fn[]) {
    T n0[3], Xxi[3], Xeta[3];
    Basis::template interpFields<3, 3>(pt, fn, n0);
    Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);

    // assemble frames dX/dxi in comp coord
    T Xd[9];
    assembleFrame<T>(Xxi, Xeta, n0, Xd);
    return A2D::MatDetCore<T, 3>(Xd);
}

// Basis related utils
template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeNodeNormals(const T Xpts[], T fn[]) {
    // the nodal normal vectors are used for director methods
    // fn is 3*num_nodes each node normals
    // Xdn is list of shell frames
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // compute the computational coord gradients of Xpts for xi, eta
        T dXdxi[3], dXdeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, dXdxi, dXdeta);

        // compute the normal vector fn at each node
        T tmp[3];
        A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
        T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
        A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, &fn[3 * inode]);
    }
}
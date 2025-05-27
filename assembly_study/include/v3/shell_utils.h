#pragma once
#include "cuda_utils.h"

template <typename T, class Basis, int vars_per_node>
__HOST_DEVICE__ static void assembleFrameLightTripleProduct(const T pt[], const T values[],
                                                            const T normal[], const T x[],
                                                            const T y[], T &xFy) {
    /* light implementation of assembleFrame for xpts */
    T Fxi, Feta;
    xFy = 0.0;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        Fxi = Basis::template interpFieldsGradLight<XI, vars_per_node>(i, pt, values);
        Feta = Basis::template interpFieldsGradLight<ETA, vars_per_node>(i, pt, values);
        xFy += x[i] * (Fxi * y[0] + Feta * y[1] + normal[i] * y[2]);
    }
}

template <typename T, class Basis, int vars_per_node>
__HOST_DEVICE__ static void assembleFrameLightTripleProductSens(const T pt[], const T x[],
                                                                const T y[], const T &xFy_bar,
                                                                T values_bar[]) {
    /* light implementation of assembleFrame for xpts */
    // version with no sens in normal vec..

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int inode = 0; inode < Basis::num_nodes; inode++) {
            T Fxi_bar = x[i] * y[0] * xFy_bar;
            T Feta_bar = x[i] * y[1] * xFy_bar;
            values_bar[vars_per_node * inode + i] +=
                Fxi_bar * Basis::template lagrangeLobatto2DGradLight<XI>(inode, pt[0], pt[1]);
            values_bar[vars_per_node * inode + i] +=
                Feta_bar * Basis::template lagrangeLobatto2DGradLight<ETA>(inode, pt[0], pt[1]);
        }
    }
}

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const int inode, const T vars[], const T Tmat[],
                                                 const T XdinvT[], T &etn) {
    /* computes the nodal contributions to the drill strain separately on each thread for greater
     * parallelism */

    // prelim block, hopefully this doesn't use too many registers, if not I can hard code it
    // compiler should just compile away my copy statements here
    T node_pt[2];
    Basis::getNodePoint(inode, node_pt);
    T zero[3];
#pragma unroll
    for (int i = 0; i < 3; i++) zero[i] = 0.0;

    // TODO : compute the inner products [0,1] and [1,0] entries of the u0xn, C with Tmat and XdinvT
    // (see v2), e.g. need inner product C'_{0,1} = t1hat^t * C * t2hat scalar triple product
    // try to do this with fast mult add __fma intrinsic? fma is on by default, but off if you
    // use_fast_math compiler flag

    T u01, u10;
    assembleFrameLightTripleProduct<T, Basis, vars_per_node>(node_pt, vars, zero, &Tmat[0],
                                                             &XdinvT[3], u01);
    assembleFrameLightTripleProduct<T, Basis, vars_per_node>(node_pt, vars, zero, &Tmat[3],
                                                             &XdinvT[0], u10);

    // compute rotation matrix at this node
    T C01, C10;
    Director::template computeRotationMatScalarProduct<vars_per_node>(inode, vars, &Tmat[0],
                                                                      &Tmat[3], C01);
    Director::template computeRotationMatScalarProduct<vars_per_node>(inode, vars, &Tmat[3],
                                                                      &Tmat[0], C10);

    // TODO : try turning off use_fast_math compiler flag on double type (only keeps this intrinsic
    // for float type, see doc)
    etn = Director::evalDrillStrain(u01, u10, C01, C10);  // this one only works for linear rotation

}  // end of method ShellComputeDrillStrainFast

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensFast(const int inode, const T vars[],
                                                     const T Tmat[], const T XdinvT[],
                                                     const T &et_bar, T res[]) {
    // prelim block
    T node_pt[2];
    Basis::getNodePoint(inode, node_pt);

    // reverse from teh
    T u01b, u10b, C01b, C10b;
    Director::evalDrillStrainSens(et_bar, u01b, u10b, C01b, C10b);

    assembleFrameLightTripleProductSens<T, Basis, vars_per_node>(node_pt, &Tmat[0], &Tmat[3], u01b,
                                                                 res);
    assembleFrameLightTripleProductSens<T, Basis, vars_per_node>(node_pt, &Tmat[3], &Tmat[0], u10b,
                                                                 res);

    Director::template computeRotationMatScalarProductSens<vars_per_node>(inode, &Tmat[0],
                                                                          &XdinvT[3], C01b, res);
    Director::template computeRotationMatScalarProductSens<vars_per_node>(inode, &Tmat[3],
                                                                          &XdinvT[0], C10b, res);

}  // end of method ShellComputeDrillStrainSensFast

template <typename T, class Basis>
__HOST_DEVICE__ static void ShellComputeNodeNormalLight(const T pt[], const T xpts[], T n0[]) {
    // compute the shell node normal at a single node given already the pre-computed spatial
    // gradients
    T Xxi[3], Xeta[3];
    for (int i = 0; i < 3; i++) {
        Xxi[i] = Basis::template interpFieldsGradLight<XI, 3>(i, pt, xpts);
        Xeta[i] = Basis::template interpFieldsGradLight<ETA, 3>(i, pt, xpts);
    }

    // compute and normalize X,xi cross X,eta
    T tmp[3];
    A2D::VecCrossCore<T>(Xxi, Xeta, tmp);
    T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
    norm = 1.0 / norm;
    A2D::VecScaleCore<T, 3>(norm, tmp, n0);
}

template <typename T, class Basis, int vars_per_node>
__HOST_DEVICE__ static void assembleFrameLight(const T pt[], const T values[], const T normal[],
                                               T frame[]) {
    /* light implementation of assembleFrame for xpts */

#pragma unroll
    for (int i = 0; i < 3; i++) {
        frame[3 * i] = Basis::template interpFieldsGradLight<XI, vars_per_node>(i, pt, values);
        frame[3 * i + 1] = Basis::template interpFieldsGradLight<ETA, vars_per_node>(i, pt, values);
        frame[3 * i + 2] = normal[i];
    }
}

template <typename T, class Basis, int vars_per_node>
__HOST_DEVICE__ static void assembleFrameLightSens(const T pt[], const T frame_bar[],
                                                   T normal_bar[3], T values_bar[]) {
    /* light implementation of assembleFrame for xpts */

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int inode = 0; inode < Basis::num_nodes; inode++) {
            values_bar[vars_per_node * inode + i] +=
                Basis::template lagrangeLobatto2DGradLight<XI>(inode, pt[0], pt[1]) *
                frame_bar[3 * i];
            values_bar[vars_per_node * inode + i] +=
                Basis::template lagrangeLobatto2DGradLight<ETA>(inode, pt[0], pt[1]) *
                frame_bar[3 * i + 1];
        }
        normal_bar[i] = frame_bar[3 * i + 2];
    }
}

// ShellComputeTransformLight defined in previous shell_utils

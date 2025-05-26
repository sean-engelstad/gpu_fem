#pragma once
#include "cuda_utils.h"

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const int inode, const T quad_pt[], const T xpts[],
                                                 const T vars[], const T Tmat[], const T XdinvT[],
                                                 T &et) {
    /* computes the nodal contributions to the drill strain separately on each thread for greater
     * parallelism */

    // instead of storing etn[4], we add to interpolated et on the fly..
    T node_pt[2];
    Basis::getNodePoint(inode, node_pt);

    // TODO : compute the inner products [0,1] and [1,0] entries of the u0xn, C with Tmat and XdinvT
    // (see v2), e.g. need inner product C'_{0,1} = t1hat^t * C * t2hat scalar triple product
    // try to do this with fast mult add __fma intrinsic? fma is on by default, but off if you
    // use_fast_math compiler flag
    T etn;
    et = etn * Basis::lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]);

    // TODO : try turning off use_fast_math compiler flag on double type (only keeps this intrinsic
    // for float type, see doc)

    // warp reduction to add across each node (for increase parallelism)
    unsigned mask = 0xFFFFFFFF;

    // Reduce across each of the 4 nodes at single quadpt and element (assumes diff nodes are adj
    // threads); Threads: inode 0,1,2,3
    et += __shfl_down_sync(mask, et, 1, 4);
    et += __shfl_down_sync(mask, et, 2, 4);

    // Broadcast from lane 0 of the group (inode == 0)
    int lane_in_group = threadIdx.x % 4;
    double reduced = __shfl_sync(mask, et, threadIdx.x - lane_in_group, 4);

}  // end of method ShellComputeDrillStrainFast

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensFast(const int inode, const T quad_pt[],
                                                     const T xpts[], const T vars[], const T Tmat[],
                                                     const T XdinvT[], const T et_bar[], T res[]) {
    // TODO : interp back et_bar => etn_bar nodal for single node

    // TODO : just backprop for single node et_bar to res (compared to before it looped over the
    // nodes)

    // also TODO : try to compute as much on the fly as possible.. or least # computations
    return;
}  // end of method ShellComputeDrillStrainSensFast
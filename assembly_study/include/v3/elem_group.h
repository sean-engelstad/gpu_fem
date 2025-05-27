#pragma once

#include "a2dcore.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_, int kernel_option_ = 1>
class ShellElementGroupV3 {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroupV3<T, Director_, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
    static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int kernel_option = kernel_option_;

    template <class Data>
    __HOST_DEVICE__ static void add_drill_strain_quadpt_residual_fast(
        const bool active_thread, const int iquad, const T vars[dof_per_elem], const Data &physData,
        const T Tmat[9], const T XdinvT[9], const T &detXdq, T res[dof_per_elem]) {
        /*
        new fast drill strain residual
        goal: all methods use shared memory or registers only, no local memory
        I will then go back and see if I can make A2D more GPU friendly at some point..
        Trying to us epre-computed Tmat, XdinvT to use less registers here..
        */

        if (!active_thread) return;
        // inode and iquad parallelism used at different points in the method, with warp ops
        int inode = iquad;

        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        T scale = detXdq * weight;

        // compute nodal drill strain with forward
        T etn_f, etq_f = 0.0;
        ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(inode, vars, Tmat, XdinvT,
                                                                       etn_f);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        //     printf("etn_f %.4e\n", etn_f);
        // }

        // do warp sync of etn_f to etn[4] and then interp to etq with warp operations
        // T etn[4] previously now just add directly into etq_f
        // get which start lane for [0-3], [4-7] groups of 4 threads in warp
        // TODO : need to double check this works correctly..
        // TODO : potential issue in speed or undefined behavior if we exceed num elements here?
        int thread_ind = blockDim.x * threadIdx.y + threadIdx.x;
        int warp_ind = thread_ind % 32;
        // maybe should use this? See one of the kernels warp reductions,
        // int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
        int group_start = (warp_ind / 4) * 4;
#pragma unroll  // warp broadcast aka sync
        for (int i = 0; i < 4; i++) {
            // TODO : should this be 0xffffffff ?

            T etni = __shfl_sync(0xFFFFFFFF, etn_f, group_start + i);  // extra arg default width=32
            etq_f += etni * Basis::lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]);
        }

        // backprop from strain energy to et, this is the total energy for single quadpt, added
        // across all nodes
        T etq_b;
        Phys::template compute_drill_strain_grad<T>(physData, scale, etq_f, etq_b);

        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
            printf("etq_f %.4e, etq_b %.4e\n", etq_f, etq_b);
        }

        // backprop from etq_b to etn_b, won't work for nonlinear rotation, TBD on that
        T etn_b = 0.0;
#pragma unroll  // warp broadcast aka sync
        for (int i = 0; i < 4; i++) {
            T etqi = __shfl_sync(0xFFFFFFFF, etq_f, group_start + i);  // extra arg default width=32
            etn_b += etqi * Basis::lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]);
        }

        // // backprop from drill strain to residual
        ShellComputeDrillStrainSensFast<T, vars_per_node, Basis, Director>(inode, vars, Tmat,
                                                                           XdinvT, etn_b, res);

        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("\tinside drill strain resid\n");
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_nodal_transforms(bool &active_thread, const int inode,
                                                         const T xpts[], const Data &data,
                                                         T Tmatn[], T XdinvTn[]) {
        if (!active_thread) return;

        T node_pt[2];
        Basis::getNodePoint(inode, node_pt);

        // get shell transform and Xdn frame scope
        T Xdinv[9];
        {
            T n0[3];
            ShellComputeNodeNormalLight<T, Basis>(node_pt, xpts, n0);

            // assemble Xd frame (Tmat treated here as Xd)
            T *Xd = &Tmatn[0];
            assembleFrameLight<T, Basis, 3>(node_pt, xpts, n0, Xd);
            A2D::MatInvCore<T, 3>(Xd, Xdinv);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransformLight<T, Basis, Data>(data.refAxis, node_pt, xpts, n0, Tmatn);
        }  // end of Xd and shell transform scope

        // get full transform product
        A2D::MatMatMultCore3x3<T>(Xdinv, Tmatn, XdinvTn);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_quadpt_transforms(bool &active_thread, const int iquad,
                                                          const T xpts[], T *detXd) {
        T quad_pt[2];
        Quadrature::getQuadraturePoint(iquad, quad_pt);

        T n0[3];
        Basis::interpNodeNormalLight(quad_pt, xpts, n0);

        T Xd[9];
        assembleFrameLight<T, Basis, 3>(quad_pt, xpts, n0, Xd);
        *detXd = A2D::MatDetCore<T, 3>(Xd);
    }
};
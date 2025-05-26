#pragma once

#include "a2dcore.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_, bool full_strain_ = true>
class ShellElementGroupV3 {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroupV2<T, Director_, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
    static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr bool full_strain = full_strain_;

#ifdef USE_GPU
    static constexpr dim3 res_block = dim3(num_nodes * num_quad_pts, 32, 1);
    // static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    template <class Data>
    __HOST_DEVICE__ static void _add_drill_strain_quadpt_residual_fast(
        const bool active_thread, const int inode, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[9], const T XdinvT[9],
        T res[dof_per_elem]) {
        /*
        new fast drill strain residual
        goal: all methods use shared memory or registers only, no local memory
        I will then go back and see if I can make A2D more GPU friendly at some point..
        Trying to us epre-computed Tmat, XdinvT to use less registers here..
        */

        if (!active_thread) return;

        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        T et_f = 0.0, et_b = 0.0;

        // compute the interpolated drill strain
        {
            ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(
                inode, quad_pt, xpts, vars, Tmat, XdinvT, et_f);
        }

        // need to get scale = detXd * weight somehow
        T detXd = Basis::getDetXd(quad_pt, xpts);
        T scale = detXd * weight;

        // backprop from strain energy to et, this is the total energy for single quadpt, added
        // across all nodes
        Phys::template compute_drill_strain_grad<T>(physData, scale, et_f, et_b);

        // // backprop from drill strain to residual
        ShellComputeDrillStrainSensFast<T, vars_per_node, Basis, Director>(
            inode, quad_pt, xpts, vars, Tmat, XdinvT, et.bvalue().get_data(), res);

        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("\tinside drill strain resid\n");
    }
};
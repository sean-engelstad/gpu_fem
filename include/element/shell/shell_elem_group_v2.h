#pragma once

#include "../base_elem_group.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "basis.h"
#include "director.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroupV2 : public BaseElementGroup<ShellElementGroup<T, Director_, Basis_, Phys_>, T,
                                                  typename Basis_::Geo, Basis_, Phys_> {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroup<T, Director_, Basis_, Phys_>;
    using Base = BaseElementGroup<ElemGroup, T, Geo, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Base::xpts_per_elem;
    static constexpr int32_t dof_per_elem = Base::dof_per_elem;
    static constexpr int32_t num_quad_pts = Base::num_quad_pts;
    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

// TODO : way to make this more general if num_quad_pts is not a multiple of 3?
// some if constexpr stuff on type of Basis?
#ifdef USE_GPU
    static constexpr dim3 energy_block = dim3(32, num_quad_pts, 1);
    // static constexpr dim3 res_block = dim3(32, num_quad_pts, 1);
    static constexpr dim3 res_block = dim3(128, num_quad_pts, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
    // static constexpr dim3 jac_block = dim3(dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T &Uelem) {
        
        
        if (!active_thread) return;
        //                                                     // TODO : do nodal parallelization? I'm not sure you can .. you need to add into et the drill strian across all nodes the interp
        // // maybe you can warp reduce across nodes from each thread? multi-step kernel maybe..
        // add_drill_strain_energy(iquad, xpts, vars, physData, Uelem);
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem])   {
        // this will just be for debug on CPU
        // TODO : setup our own kernel functions for each element type and tunable launch params for each
        // for this case, I want to launch a separate kernel for the drill strain, tying strain, bending strains, etc.

        if (!active_thread) return;
        // split up the quadpt residual into three pieces that contribute
        // separate terms to the strain energy for a metal / symmetric composite laminate
        // if not not sym laminate, could add extra term with k^T B eps_0 strain energy

        int VERSION = 2; // 3
        constexpr int CONTRIBUTION = 1; // for prelim testing, turn on only one term here

        if constexpr (CONTRIBUTION == 0) {
            _add_drill_strain_quadpt_residual<Data, VERSION>(iquad, xpts, vars, physData, res);
        } else if (CONTRIBUTION == 1) {
            _add_tying_strain_quadpt_residual<Data>(iquad, xpts, vars, physData, res);
        } else {
            _add_bending_strain_quadpt_residual<Data>(iquad, xpts, vars, physData, res);
        }
    }

    template <class Data, int version>
    __HOST_DEVICE__ static void _add_drill_strain_quadpt_residual(const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T res[dof_per_elem]) {
        // this one sped up by calling 128 elements per block..
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        A2D::ADObj<A2D::Vec<T, 1>> et;

        // compute the interpolated drill strain
        if constexpr (version == 2) {
            ShellComputeDrillStrainV2<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, et.value().get_data());
        } else if (version == 3) {
            ShellComputeDrillStrainV3<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, et.value().get_data());
        }

        // need to get scale = detXd * weight somehow
        T detXd = Basis::getDetXd(pt, xpts);
        T scale = detXd * weight;

        // backprop from strain energy to et
        Phys::template compute_drill_strain_grad<T>(physData, scale, et);
            
        // backprop from drill strain to residual
        if constexpr (version == 2) {
            ShellComputeDrillStrainSensV2<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, et.bvalue().get_data(), res);
        } else if (version == 3) {
            ShellComputeDrillStrainSensV3<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, et.bvalue().get_data(), res);
        }
        
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_tying_strain_quadpt_residual(const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T res[dof_per_elem]) {
        // TODO
        T pt[2], detXd;
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

        // forward scope block
        T XdinvT[9];
        { 
            // compute tying strain at the tying points
            T ety[Basis::num_all_tying_points];
            computeTyingStrainLight<T, Phys, Basis, Director>(xpts, vars, ety);

            // interp and rotate the tying strain
            A2D::SymMat<T, 3> gty;
            interpTyingStrainLight<T, Basis>(pt, ety, gty.get_data());
            detXd = getFrameRotation<T, Data, Basis>(physData.refAxis, pt, xpts, XdinvT);
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty.get_data(), e0ty.value().get_data());
        }

        // backprop from strain energy to tying strain gradient in physics
        T scale = detXd * weight;
        Phys::template compute_tying_strain_midplane_grad<T>(physData, scale, e0ty);
        Phys::template compute_tying_strain_transverse_grad<T>(physData, scale, e0ty);

        // reverse scope block
        { 
            // interp tying strain sens
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
            {
                A2D::SymMat<T, 3> gty_bar;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gty_bar.get_data());
                addinterpTyingStrainTransposeLight<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
            }

            // compute tying strain sens
            computeTyingStrainSensLight<T, Phys, Basis, Director>(xpts, vars, ety_bar.get_data(), res);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_bending_strain_quadpt_residual(const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T &Uelem) {
        // TODO
        
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // TODO: but do residual first
    }


};
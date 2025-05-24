#pragma once

#include "../base_elem_group.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "basis.h"
#include "director.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroupV2 : public BaseElementGroup<ShellElementGroup<T, Director_, Basis_, Phys_>,
                                                    T, typename Basis_::Geo, Basis_, Phys_> {
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
    static constexpr dim3 res_block = dim3(32, num_quad_pts, 1);
    // static constexpr dim3 res_block = dim3(64, num_quad_pts, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
    // static constexpr dim3 jac_block = dim3(dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T &Uelem) {
        if (!active_thread) return;
        //                                                     // TODO : do nodal parallelization?
        //                                                     I'm not sure you can .. you need to
        //                                                     add into et the drill strian across
        //                                                     all nodes the interp
        // // maybe you can warp reduce across nodes from each thread? multi-step kernel maybe..
        // add_drill_strain_energy(iquad, xpts, vars, physData, Uelem);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_nodal_shell_transforms(const bool active_thread,
                                                               const int inode,
                                                               const T xpts[xpts_per_elem],
                                                               const Data physData, T Tmat[9],
                                                               T XdinvT[9]) {
        if (!active_thread) return;

        ShellComputeNodalTransforms<T, Data, Basis>(inode, xpts, physData, Tmat, XdinvT);
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem]) {
        // this will just be for debug on CPU
        // TODO : setup our own kernel functions for each element type and tunable launch params for
        // each for this case, I want to launch a separate kernel for the drill strain, tying
        // strain, bending strains, etc.

        if (!active_thread) return;
        // split up the quadpt residual into three pieces that contribute
        // separate terms to the strain energy for a metal / symmetric composite laminate
        // if not not sym laminate, could add extra term with k^T B eps_0 strain energy

        constexpr int CONTRIBUTION = 0;  // for prelim testing, turn on only one term here

        if constexpr (CONTRIBUTION == 0) {
            constexpr int VERSION = 2;  // 3
            _add_drill_strain_quadpt_residual<Data, VERSION>(iquad, xpts, vars, physData, res);
            // _add_drill_strain_quadpt_residual_fast<Data>(iquad, xpts, vars, physData, res);
        } else if (CONTRIBUTION == 1) {
            _add_tying_strain_quadpt_residual<Data>(iquad, xpts, vars, physData, res);
        } else {
            _add_bending_strain_quadpt_residual<Data>(iquad, xpts, vars, physData, res);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual_fast(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[36], const T XdinvT[36],
        T res[dof_per_elem]) {
        // in this version I'm trying to use only xpt

        if (!active_thread) return;
        // split up the quadpt residual into three pieces that contribute
        // separate terms to the strain energy for a metal / symmetric composite laminate
        // if not not sym laminate, could add extra term with k^T B eps_0 strain energy

        constexpr int CONTRIBUTION = 0;  // for prelim testing, turn on only one term here

        if constexpr (CONTRIBUTION == 0) {
            _add_drill_strain_quadpt_residual_fast<Data>(iquad, xpts, vars, physData, Tmat, XdinvT,
                                                         res);
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
                                                                  const Data physData,
                                                                  T res[dof_per_elem]) {
        // printf("in drill strain\n");

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
    __HOST_DEVICE__ static void _add_drill_strain_quadpt_residual_fast(
        const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
        const Data physData, const T Tmat[36], const T XdinvT[36], T res[dof_per_elem]) {
        /*
        new fast drill strain residual
        goal: all methods use shared memory or registers only, no local memory
        I will then go back and see if I can make A2D more GPU friendly at some point..
        Trying to us epre-computed Tmat, XdinvT to use less registers here..
        */

        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        A2D::ADObj<A2D::Vec<T, 1>> et;

        // compute the interpolated drill strain
        {
            ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(
                quad_pt, xpts, vars, Tmat, XdinvT, et.value().get_data());
        }

        // very fast up to here..

        // need to get scale = detXd * weight somehow
        T detXd = Basis::getDetXd(quad_pt, xpts);
        T scale = detXd * weight;

        // backprop from strain energy to et
        Phys::template compute_drill_strain_grad<T>(physData, scale, et);

        // // backprop from drill strain to residual
        ShellComputeDrillStrainSensFast<T, vars_per_node, Basis, Director>(
            quad_pt, xpts, vars, Tmat, XdinvT, et.bvalue().get_data(), res);
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_tying_strain_quadpt_residual(const int iquad,
                                                                  const T xpts[xpts_per_elem],
                                                                  const T vars[dof_per_elem],
                                                                  const Data physData,
                                                                  T res[dof_per_elem]) {
        // TODO
        // printf("in tying strain\n");
        T pt[2], detXd;
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

        // forward scope block
        T XdinvT[9];  // 16 registers here

        {
            // compute tying strain at the tying points
            T ety[Basis::num_all_tying_points];
            computeTyingStrainLight<T, Phys, Basis, Director>(xpts, vars, ety);

            // return; // 27 registers here

            // interp and rotate the tying strain
            A2D::SymMat<T, 3> gty;
            interpTyingStrainLight<T, Basis>(pt, ety, gty.get_data());
            detXd = getFrameRotation<T, Data, Basis>(physData.refAxis, pt, xpts, XdinvT);
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
        }

        // return; // 29 registers per thread

        // backprop from strain energy to tying strain gradient in physics
        T scale = detXd * weight;
        Phys::template compute_tying_strain_midplane_grad<T>(physData, scale, e0ty);
        Phys::template compute_tying_strain_transverse_grad<T>(physData, scale, e0ty);

        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //	A2D::SymMat<T,3>& e0ty_f = e0ty.value();
        //	printf("e0ty_f: %.4e %.4e %.4e %.4e %.4e %.4e\n", e0ty_f[0], e0ty_f[1], e0ty_f[2],
        // e0ty_f[3], e0ty_f[4], e0ty_f[5]); 	A2D::SymMat<T,3>& e0ty_b = e0ty.bvalue();
        //	printf("e0ty_b: %.4e %.4e %.4e %.4e %.4e %.4e\n", e0ty_b[0], e0ty_b[1], e0ty_b[2],
        // e0ty_b[3], e0ty_b[4], e0ty_b[5]);
        // }

        // return; // 32 registers per thread

        // reverse scope block
        {
            // interp tying strain sens
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
            {
                A2D::SymMat<T, 3> gty_bar;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                    gty_bar.get_data());

                // return; // still 32 registers per thread

                // TODO : the dot sens methods here are bugged out, massively grows the registers
                // and slow down runtime by 10x lol this is the bottleneck function..
                addInterpTyingStrainTransposeLight<T, Basis>(pt, gty_bar.get_data(),
                                                             ety_bar.get_data());
            }

            // return; // still 32 registers per thread

            // compute tying strain sens
            computeTyingStrainSensLight<T, Phys, Basis, Director>(xpts, vars, ety_bar.get_data(),
                                                                  res);
        }

        return;  //
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_bending_strain_quadpt_residual(const int iquad,
                                                                    const T xpts[xpts_per_elem],
                                                                    const T vars[dof_per_elem],
                                                                    const Data physData, T res[]) {
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
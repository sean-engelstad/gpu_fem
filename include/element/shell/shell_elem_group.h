#pragma once

#include "../base_elem_group.h"
#include "_shell_types.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "basis.h"
#include "director.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroup : public BaseElementGroup<ShellElementGroup<T, Director_, Basis_, Phys_>, T,
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
    static constexpr dim3 energy_block = dim3(num_quad_pts, 32, 1);
    static constexpr dim3 res_block = dim3(num_quad_pts, 32, 1);
    static constexpr dim3 jac_block = dim3(num_quad_pts, dof_per_elem, 1);
#endif  // USE_GPU

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T &Uelem) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        A2D::ADObj<T> _Uelem;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            T d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

            // get the scale for disp grad sens of the energy
            T scale = detXd * weight;

            // compute energy + energy-dispGrad sensitivites with physics
            Phys::template computeStrainEnergy<T>(physData, scale, u0x, u1x, e0ty, et, _Uelem);

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Uelem += _Uelem.value();
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem])

    {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T d[3 * num_nodes];   // needed for reverse mode, nonlinear case
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T detXd;

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);
            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // TODO : have to compoute detXd here later

            // TODO : split up calls here more by bending and tying strains (disp grad does some
            // tying strain computations right now)

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

        }  // end of forward scope block for strain energy
           // ------------------------------------------------

        // get the scale for disp grad sens of the energy
        T scale = detXd * weight;

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeWeakRes<T>(physData, scale, u0x, u1x, e0ty, et);

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

        // T ety_bar[Basis::num_all_tying_points];
        ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            e0ty.bvalue(), res, d_bar.get_data(), ety_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), res,
                                               d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(), res);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col_no_resid(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T matCol[dof_per_elem]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T scale;              // scale for energy derivatives
        T d[3 * num_nodes];   // need directors in reverse for nonlinear strains
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T detXd;

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;
        A2D::A2DObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward section
        {
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope

        // hforward section (pvalue's)
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        T p_d[3 * num_nodes];
        {
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, et.pvalue().get_data());

            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                             p_d);

            T p_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                   p_ety);

            ShellComputeDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, p_ety,
                u0x.pvalue().get_data(), u1x.pvalue().get_data(), e0ty.pvalue());

        }  // end of hforward scope

        // derivatives over disp grad to strain energy portion
        // ---------------------
        Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, e0ty, et);
        // ---------------------
        // begin reverse blocks from strain energy => physical disp grad sens

        // breverse (1st order derivs)
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
        if constexpr (is_nonlinear) {
            A2D::Vec<T, 3 * num_nodes> d_bar;     // zeroes out on init
            constexpr bool back_to_dbar = false;  // change to
            ShellComputeDispGradSens_NoResid<T, vars_per_node, Basis, Data, back_to_dbar>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), d_bar.get_data(), ety_bar.get_data());

            // TODO : if also computing nonlinear rotation will need an extra compute director sens
            // call here
        }  // end of breverse scope (1st order derivs)

        // hreverse (2nd order derivs)
        {
            A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
            A2D::Vec<T, 3 * num_nodes> d_hat;                  // zeroes out on init

            ShellComputeDispGradHrev<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), e0ty.hvalue(), matCol, d_hat.get_data(),
                ety_hat.get_data());

            computeTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                   ety_bar.get_data(), ety_hat.get_data(), matCol,
                                                   d_hat.get_data());

            Director::template computeDirectorHrev<vars_per_node, num_nodes>(fn, d_hat.get_data(),
                                                                             matCol);

            ShellComputeDrillStrainHrev<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);
        }  // end of hreverse scope (2nd order derivs)
    }      // add_element_quadpt_jacobian_col

    template <class Data>
    __HOST_DEVICE__ static void compute_shell_transform_data(const bool active_thread,
                                                             const int iquad,
                                                             const T xpts[xpts_per_elem],
                                                             const Data physData, T Tmat[],
                                                             T XdinvT[], T &detXd) {
        /* compute shell transform matrices at each quadpt, not fully optimized since I just need */
        if (!active_thread) return;
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // get the shell node normal at the quadpt
        T dXdxi[3], dXdeta[3], n0[3];
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);
        T tmp[3];
        A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
        T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
        A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, n0);

        _compute_shell_transforms<T, Data, Basis>(pt, physData.refAxis, xpts, n0, Tmat, XdinvT,
                                                  detXd);
    }

    template <class Data>
    __DEVICE__ static void compute_shell_transforms2(const bool active_thread, const int iquad,
                                                     const T xpts[], Data physData, T Tmat[],
                                                     T XdinvT[], T &detXd) {
        if (!active_thread) return;
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        // get Tmat (if not available)
        {
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, xpts, dXdxi, dXdeta);

            // compute shell normal
            T n0[3], tmp[3];
            A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
            T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
            A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, n0);

            // assemble Xd frame
            T Xd[9];
            Basis::assembleFrame(dXdxi, dXdeta, n0, Xd);

            // compute the shell transform based on the ref axis in Data object
            ShellComputeTransform<T, Data>(physData.refAxis, dXdxi, dXdeta, n0, Tmat);

            // get XdinvT too
            T Xdinv[9];
            A2D::MatInvCore<T, 3>(Xd, Xdinv);
            detXd = A2D::MatDetCore<T, 3>(Xd);
            A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_drill_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[], const T XdinvT[],
        const T &detXd, T res[dof_per_elem])

    {
        /* low register method to get drill strain residual only */
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Vec<T, 1>> et;

        // forward scope block
        {
            // assemble u0xn frame
            T u0x_1, u0x_3;
            {
                T u0x[9];
                // compute midplane disp field gradients
                T u0xi[3], u0eta[3];
                Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

                A2D::Vec<T, 3> zero;
                Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0x);

                // u0x' = T^T * u0x * XdinvT (but only [1], [3] entries of that)
                u0x_1 = A2D::Mat3x3MatTripleProduct<T>(0, 1, Tmat, u0x, XdinvT);
                u0x_3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, u0x, XdinvT);
            }

            T C1, C3;
            {
                // interpolate rotation vars to this quadpt (linear rotation only)
                T rots[6];
                Basis::template interpFieldsOffset<vars_per_node, 3, 3>(pt, vars, &rots[3]);

                // compute the rotation matrix
                T C[9];
                Director::template computeRotationMat<vars_per_node, 1>(rots, C);

                C3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, C, Tmat);
                C1 = -C3;
            }

            // compute drill strains
            et.value()[0] = 0.5 * (C3 + u0x_3 - u0x_1 - C1);
        }

        // get the scale for disp grad sens of the energy
        T scale = detXd * weight;

        // compute energy strain residual
        // Phys::template computeWeakRes<T>(physData, scale, u0x, u1x, e0ty, et);
        T drill;
        {  // TODO : could just compute G here separately.., less data
            T C[6], E = physData.E, nu = physData.nu, thick = physData.thick;
            Data::evalTangentStiffness2D(E, nu, C);
            T As = Data::getTransShearCorrFactor() * thick * C[5];
            drill = Data::getDrillingRegularization() * As;
            et.bvalue()[0] = scale * drill * et.value()[0];  // backprop from strain energy
        }

        // beginning of backprop section to final residual derivatives
        // ----------------------------------------------------
        T etb = et.bvalue()[0];
        // backprop through rotation mat
        {
            T C3b = 0.5 * etb, C1b = -0.5 * etb;
            T Cb[9];
            memset(Cb, 0.0, 9 * sizeof(T));
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, Tmat, C3b, Cb);
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, Tmat, C1b, Cb);

            T rot_sens[6];
            Director::template computeRotationMatSens<vars_per_node, 1>(Cb, rot_sens);

            Basis::template interpFieldsOffsetSens<vars_per_node, 3, 3>(pt, &rot_sens[3], res);
        }

        // backprop through u0x disp grad
        {
            T u0xb3 = 0.5 * etb;
            T u0xb1 = -u0xb3;

            T u0xb[9];
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, XdinvT, u0xb3, u0xb);
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, XdinvT, u0xb1, u0xb);

            // transpose u0xb
#pragma unroll
            for (int i = 0; i < 3; i++) {
#pragma unroll
                for (int j = 0; j < i; j++) {
                    T tmp = u0xb[3 * i + j];
                    u0xb[3 * i + j] = u0xb[3 * j + i];
                    u0xb[3 * j + i] = tmp;
                }
            }

            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0xb[0], &u0xb[3],
                                                                        res);
        }

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_drill_jacobian_col(
        const bool active_thread, const int iquad, const int ideriv, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[], const T XdinvT[],
        const T &detXd, T mat_col[dof_per_elem]) {
        /* low register method to get drill strain residual only */
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        constexpr bool nonlinear_rotation =
            false;  // set this in physics later, not settable yet (so assumes linear rot)

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Vec<T, 1>> et;

        // forward scope block
        {
            // assemble u0xn frame
            T u0x_1, u0x_3;
            {
                T u0x[9];
                // compute midplane disp field gradients
                T u0xi[3], u0eta[3];
                Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

                A2D::Vec<T, 3> zero;
                Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0x);

                // u0x' = T^T * u0x * XdinvT (but only [1], [3] entries of that)
                u0x_1 = A2D::Mat3x3MatTripleProduct<T>(0, 1, Tmat, u0x, XdinvT);
                u0x_3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, u0x, XdinvT);
            }

            T C1, C3;
            {
                // interpolate rotation vars to this quadpt (linear rotation only)
                T rots[6];
                Basis::template interpFieldsOffset<vars_per_node, 3, 3>(pt, vars, &rots[3]);

                // compute the rotation matrix
                T C[9];
                Director::template computeRotationMat<vars_per_node, 1>(rots, C);

                C3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, C, Tmat);
                C1 = -C3;
            }

            // compute drill strains
            et.value()[0] = 0.5 * (C3 + u0x_3 - u0x_1 - C1);
        }

        // hforward scope block (TODO : would store pvalue of C for hrev part)
        if constexpr (nonlinear_rotation) {
            A2D::Vec<T, dof_per_elem> p_vars;
            p_vars[ideriv] = 1.0;  // p_vars is unit vector for current column to compute
            // assemble u0xn frame
            T u0x_1, u0x_3;
            {
                T u0x[9];
                // compute midplane disp field gradients
                T u0xi[3], u0eta[3];
                Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, u0xi, u0eta);

                A2D::Vec<T, 3> zero;
                Basis::assembleFrame(u0xi, u0eta, zero.get_data(), u0x);

                // u0x' = T^T * u0x * XdinvT (but only [1], [3] entries of that)
                u0x_1 = A2D::Mat3x3MatTripleProduct<T>(0, 1, Tmat, u0x, XdinvT);
                u0x_3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, u0x, XdinvT);
            }

            T C1, C3;
            {
                // interpolate rotation vars to this quadpt (linear rotation only)
                T rots[6];
                Basis::template interpFieldsOffset<vars_per_node, 3, 3>(pt, p_vars, &rots[3]);

                // compute the rotation matrix
                T C[9];
                Director::template computeRotationMat<vars_per_node, 1>(rots, C);

                C3 = A2D::Mat3x3MatTripleProduct<T>(1, 0, Tmat, C, Tmat);
                C1 = -C3;
            }

            // compute drill strains
            et.pvalue()[0] = 0.5 * (C3 + u0x_3 - u0x_1 - C1);
        }  // end of hforward block

        // get the scale for disp grad sens of the energy
        T scale = detXd * weight;

        // compute energy strain residual
        // Phys::template computeWeakRes<T>(physData, scale, u0x, u1x, e0ty, et);
        T drill;
        {  // TODO : could just compute G here separately.., less data
            T C[6], E = physData.E, nu = physData.nu, thick = physData.thick;
            Data::evalTangentStiffness2D(E, nu, C);
            T As = Data::getTransShearCorrFactor() * thick * C[5];
            drill = Data::getDrillingRegularization() * As;
            // this is basically computing the output level proj Hessian (that we can now backprop)
            et.hvalue()[0] = scale * drill * et.pvalue()[0];  // backprop from strain energy
        }

        // TODO : would need first order backprop if nonlinear rot later

        // hrev section (projected hessians)
        // ----------------------------------------------------
        T eth = et.hvalue()[0];
        // backprop through rotation mat (TODO : missing nonlinear rot part)
        {
            T C3b = 0.5 * eth, C1b = -0.5 * eth;
            T Cb[9];
            memset(Cb, 0.0, 9 * sizeof(T));
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, Tmat, C3b, Cb);
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, Tmat, C1b, Cb);

            T rot_sens[6];
            Director::template computeRotationMatSens<vars_per_node, 1>(Cb, rot_sens);

            Basis::template interpFieldsOffsetSens<vars_per_node, 3, 3>(pt, &rot_sens[3], mat_col);
        }

        // backprop through u0x disp grad
        {
            T u0xb3 = 0.5 * eth;
            T u0xb1 = -u0xb3;

            T u0xb[9];
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, XdinvT, u0xb3, u0xb);
            A2D::Mat3x3MatTripleProductSens<T>(1, 0, Tmat, XdinvT, u0xb1, u0xb);

            // transpose u0xb
#pragma unroll
            for (int i = 0; i < 3; i++) {
#pragma unroll
                for (int j = 0; j < i; j++) {
                    T tmp = u0xb[3 * i + j];
                    u0xb[3 * i + j] = u0xb[3 * j + i];
                    u0xb[3 * j + i] = tmp;
                }
            }

            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0xb[0], &u0xb[3],
                                                                        mat_col);
        }

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_bending_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[], const T XdinvT[],
        const T &detXd, T res[dof_per_elem]) {
        // add bending strain contribution to residual
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // compute un-rotated disp grads forward
        T XdinvzT[9], u0xb[9], u1xb[9];  // have to put u0xb, u1xb here because if defined inside
        // u0x, u1x scope, lost for backprop
        {
            T u0x[9], u1x[9];
            {
                T d[12];  // director section
                {
                    T fn[12];
                    ShellComputeNodeNormals<T, Basis>(xpts, fn);

                    Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                    T nxi[3], neta[3];
                    Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

                    T Xdz[9];
                    A2D::Vec<T, 3> zero;
                    Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

                    // recompute Xdinv = XdinvT * T^t
                    T Xdinv[9];
                    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(
                        XdinvT, Tmat, Xdinv);

                    // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
                    T tmp[9];
                    A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
                    A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);
                }

                // compute unrotated disp grads
                {
                    // then get u0x
                    {
                        // interp midplane disp grads
                        T u0xi[3], u0eta[3];
                        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

                        T d0[3];
                        Basis::template interpFields<3, 3>(pt, d, d0);
                        Basis::assembleFrame(u0xi, u0eta, d0, u0x);
                    }

                    // then get u1x
                    {
                        T d0xi[3], d0eta[3];
                        A2D::Vec<T, 3> zero;
                        Basis::template interpFieldsGrad<3, 3>(pt, d, d0xi, d0eta);
                        Basis::assembleFrame(d0xi, d0eta, zero.get_data(), u1x);
                    }
                }
            }  // directors unloaded

            // rotate disp grads forward
            {
                T tmp[9];

                // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
                A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
                A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
                A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1x);

                // compute u0x = T^{T}*u0d*Xdinv*T
                A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
                A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0x);
            }

            // disp grads fw to reverse (weak disp grad residual here)
            {
                // compute bending strains
                T ek[3];
                if constexpr (is_nonlinear) {
                    ek[0] = u1x[0] + (u0x[0] * u1x[0] + u0x[3] * u1x[3] + u0x[6] * u1x[6]);  // k11
                    ek[1] = u1x[4] + (u0x[1] * u1x[1] + u0x[4] * u1x[4] + u0x[7] * u1x[7]);  // k22
                    ek[2] = u1x[1] + u1x[3] +
                            (u0x[0] * u1x[1] + u0x[3] * u1x[4] + u0x[6] * u1x[7] + u1x[0] * u0x[1] +
                             u1x[3] * u0x[4] + u1x[6] * u0x[7]);  // k12
                } else {
                    ek[0] = u1x[0];           // k11
                    ek[1] = u1x[4];           // k22
                    ek[2] = u1x[1] + u1x[3];  // k12
                }

                // compute bending stresses in eb (for backprop from energy, equiv)
                T eb[3], C[6];
                Data::evalTangentStiffness2D(physData.E, physData.nu, C);
                T thick = physData.thick;
                T I = thick * thick * thick / 12.0;  // assuming thickOffset = 0 here
                A2D::SymMatVecCoreScale3x3<T, true>(I, C, ek, eb);

                // now backprop to u0xb, u1xb
                if constexpr (is_nonlinear) {
                    // k11 computation
                    u1xb[0] = eb[0];
                    u0xb[0] = u1x[0] * eb[0];
                    u1xb[0] += u0x[0] * eb[0];
                    u0xb[3] = u1x[3] * eb[0];
                    u1xb[3] = u0x[3] * eb[0];
                    u0xb[6] = u1x[6] * eb[0];
                    u1xb[6] = u0x[6] * eb[0];
                    // k22 computation
                    u1xb[4] = eb[1];
                    u0xb[1] = u1x[1] * eb[1];
                    u1xb[1] = u0x[1] * eb[1];
                    u0xb[4] = u1x[4] * eb[1];
                    u1xb[4] += u0x[4] * eb[1];
                    u0xb[7] = u1x[7] * eb[1];
                    u1xb[7] = u0x[7] * eb[1];
                    // k12 computation
                    u1xb[1] += eb[2];
                    u1xb[3] += eb[2];
                    u0xb[0] += u1x[1] * eb[2];
                    u1xb[0] += u0x[1] * eb[2];
                    u0xb[1] += u1x[0] * eb[2];
                    u1xb[1] += u0x[0] * eb[2];
                    u0xb[3] += u1x[4] * eb[2];
                    u1xb[3] += u0x[4] * eb[2];
                    u0xb[4] += u1x[3] * eb[2];
                    u1xb[4] += u0x[3] * eb[2];
                    u0xb[6] += u1x[7] * eb[2];
                    u1xb[6] += u0x[7] * eb[2];
                    u0xb[7] += u1x[6] * eb[2];
                    u1xb[7] += u0x[6] * eb[2];
                } else {
                    u1xb[0] = eb[0];  // k11
                    u1xb[4] = eb[1];  // k22
                    u1xb[1] = eb[2];  // k12
                    u1xb[3] = eb[2];  // k12
                }
            }  // end of compute u0xb, u1xb block, strains and stresses unloaded
        }      // u0x, u1x forward disp grads unloaded

        constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
        constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

        T d_bar[3];
        // scope for u0d_barT
        {
            T u0d_barT[9], tmp[9];

            // u0d_bar^t = XdinvT * u0x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xb, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

            // u0d_bar^t += XdinvzT * u1x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvzT, u1xb, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

            Basis::template interpFieldsTranspose<3, 3>(pt, &u0d_barT[6], d_bar);
            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0d_barT[0],
                                                                        &u0d_barT[3], res);
        }  // end of u0d_barT scope

        // scope for u1d_barT
        {
            T u1d_barT[9], tmp[9];

            // u1d_barT^t = XdinvT * u1x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u1xb, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u1d_barT);

            // prev : cols of u1d are {d0xi, d0eta, zero} => extract from rows of u1d_bar^T => d_bar
            Basis::template interpFieldsGradTranspose<3, 3>(pt, &u1d_barT[0], &u1d_barT[3], d_bar);
        }  // end of u0d_barT scope

        // transfer back through directors
        {
            T fn[12];

            ShellComputeNodeNormals<T, Basis>(xpts, fn);  // recompute fn here
            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar, res);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_bending_jacobian_col(
        const bool active_thread, const int iquad, const int ideriv, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[], const T XdinvT[],
        const T &detXd, T mat_col[dof_per_elem]) {
        // add bending strain contribution to residual
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // compute un-rotated disp grads forward
        T XdinvzT[9], u0xh[9], u1xh[9];  // have to put u0xb, u1xb here because if defined inside
        // u0x, u1x scope, lost for backprop
        {
            T u0x[9], u1x[9];
            {
                T d[12];  // director section
                {
                    T fn[12];
                    ShellComputeNodeNormals<T, Basis>(xpts, fn);

                    Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                    T nxi[3], neta[3];
                    Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

                    T Xdz[9];
                    A2D::Vec<T, 3> zero;
                    Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

                    // recompute Xdinv = XdinvT * T^t
                    T Xdinv[9];
                    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(
                        XdinvT, Tmat, Xdinv);

                    // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
                    T tmp[9];
                    A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
                    A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);
                }

                // compute unrotated disp grads
                {
                    // then get u0x
                    {
                        // interp midplane disp grads
                        T u0xi[3], u0eta[3];
                        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

                        T d0[3];
                        Basis::template interpFields<3, 3>(pt, d, d0);
                        Basis::assembleFrame(u0xi, u0eta, d0, u0x);
                    }

                    // then get u1x
                    {
                        T d0xi[3], d0eta[3];
                        A2D::Vec<T, 3> zero;
                        Basis::template interpFieldsGrad<3, 3>(pt, d, d0xi, d0eta);
                        Basis::assembleFrame(d0xi, d0eta, zero.get_data(), u1x);
                    }
                }
            }  // directors unloaded

            // rotate disp grads forward
            {
                T tmp[9];

                // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
                A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
                A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
                A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1x);

                // compute u0x = T^{T}*u0d*Xdinv*T
                A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
                A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0x);
            }

            // pforward to ekp and start hreverse to strains (linear part 2nd order)
            T u0xp[9], u1xp[9], eh[3];
            {
                A2D::Vec<T, dof_per_elem> p_vars;
                p_vars[ideriv] = 1.0;  // p_vars is unit vector for current column to compute

                {
                    T pd[12];  // director section
                    {
                        T fn[12];
                        ShellComputeNodeNormals<T, Basis>(xpts, fn);

                        Director::template computeDirector<vars_per_node, num_nodes>(
                            p_vars.get_data(), fn, pd);
                    }

                    // compute unrotated disp grads
                    {
                        // then get u0x
                        {
                            // interp midplane disp grads
                            T u0xi[3], u0eta[3];
                            Basis::template interpFieldsGrad<vars_per_node, 3>(
                                pt, p_vars.get_data(), u0xi, u0eta);

                            T d0[3];
                            Basis::template interpFields<3, 3>(pt, pd, d0);
                            Basis::assembleFrame(u0xi, u0eta, d0, u0xp);
                        }

                        // then get u1x
                        {
                            T d0xi[3], d0eta[3];
                            A2D::Vec<T, 3> zero;
                            Basis::template interpFieldsGrad<3, 3>(pt, pd, d0xi, d0eta);
                            Basis::assembleFrame(d0xi, d0eta, zero.get_data(), u1xp);
                        }
                    }

                }  // directors unloaded

                // rotate disp grads forward
                {
                    T tmp[9];

                    // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
                    A2D::MatMatMultCore3x3<T>(u1xp, XdinvT, tmp);
                    A2D::MatMatMultCore3x3Add<T>(u0xp, XdinvzT, tmp);
                    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1xp);

                    // compute u0x = T^{T}*u0d*Xdinv*T
                    A2D::MatMatMultCore3x3<T>(u0xp, XdinvT, tmp);
                    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0xp);
                }

                T ekp[3];
                if constexpr (is_nonlinear) {
                    ekp[0] = u1xp[0] +
                             (u0xp[0] * u1xp[0] + u0xp[3] * u1xp[3] + u0xp[6] * u1xp[6]);  // k11
                    ekp[1] = u1xp[4] +
                             (u0xp[1] * u1xp[1] + u0xp[4] * u1xp[4] + u0xp[7] * u1xp[7]);  // k22
                    ekp[2] = u1xp[1] + u1xp[3] +
                             (u0xp[0] * u1xp[1] + u0xp[3] * u1xp[4] + u0xp[6] * u1xp[7] +
                              u1xp[0] * u0xp[1] + u1xp[3] * u0xp[4] + u1xp[6] * u0xp[7]);  // k12
                } else {
                    ekp[0] = u1xp[0];            // k11
                    ekp[1] = u1xp[4];            // k22
                    ekp[2] = u1xp[1] + u1xp[3];  // k12
                }

                {  // D * ep => eh (linear part of proj hessian)
                    T C[6];
                    Data::evalTangentStiffness2D(physData.E, physData.nu, C);
                    T thick = physData.thick;
                    T I = thick * thick * thick / 12.0;  // assuming thickOffset = 0 here
                    A2D::SymMatVecCoreScale3x3<T, true>(I, C, ekp, eh);
                }
            }  // end of pforward disp grads

            // disp grads fw to hreverse (weak disp grad residual here)
            {
                // compute first order bending strain sens
                T eb[3];
                if constexpr (is_nonlinear) {
                    // compute forward bending strains
                    T ek[3];
                    if constexpr (is_nonlinear) {
                        ek[0] =
                            u1x[0] + (u0x[0] * u1x[0] + u0x[3] * u1x[3] + u0x[6] * u1x[6]);  // k11
                        ek[1] =
                            u1x[4] + (u0x[1] * u1x[1] + u0x[4] * u1x[4] + u0x[7] * u1x[7]);  // k22
                        ek[2] = u1x[1] + u1x[3] +
                                (u0x[0] * u1x[1] + u0x[3] * u1x[4] + u0x[6] * u1x[7] +
                                 u1x[0] * u0x[1] + u1x[3] * u0x[4] + u1x[6] * u0x[7]);  // k12
                    } else {
                        ek[0] = u1x[0];           // k11
                        ek[1] = u1x[4];           // k22
                        ek[2] = u1x[1] + u1x[3];  // k12
                    }

                    // compute bending stresses (1st order reverse)

                    {
                        T C[6];
                        Data::evalTangentStiffness2D(physData.E, physData.nu, C);
                        T thick = physData.thick;
                        T I = thick * thick * thick / 12.0;  // assuming thickOffset = 0 here
                        A2D::SymMatVecCoreScale3x3<T, true>(I, C, ek, eb);
                    }
                }  // end of first order bending strains sens

                // hrev backprop of bending strains to disp grad
                if constexpr (is_nonlinear) {
                    // k11 computation
                    u1xh[0] = eh[3];
                    //   nonlinear input * h terms
                    u0xh[0] = u1x[0] * eh[3];
                    u1xh[0] += u0x[0] * eh[3];
                    u0xh[3] = u1x[3] * eh[3];
                    u1xh[3] = u0x[3] * eh[3];
                    u0xh[6] = u1x[6] * eh[3];
                    u1xh[6] = u0x[6] * eh[3];
                    //   nonlinear bar * ptest terms
                    u0xh[0] += u1xp[0] * eb[3];
                    u1xh[0] += u0xp[0] * eb[3];
                    u0xh[3] += u1xp[3] * eb[3];
                    u1xh[3] += u0xp[3] * eb[3];
                    u0xh[6] += u1xp[6] * eb[3];
                    u1xh[6] += u0xp[6] * eb[3];

                    // k22 computation
                    u1xh[4] = eh[4];
                    //   nonlinear input * h terms
                    u0xh[1] = u1x[1] * eh[4];
                    u1xh[1] = u0x[1] * eh[4];
                    u0xh[4] = u1x[4] * eh[4];
                    u1xh[4] += u0x[4] * eh[4];
                    u0xh[7] = u1x[7] * eh[4];
                    u1xh[7] = u0x[7] * eh[4];
                    //   nonlinear bar * ptest terms
                    u0xh[1] += u1xp[1] * eb[4];
                    u1xh[1] += u0xp[1] * eb[4];
                    u0xh[4] += u1xp[4] * eb[4];
                    u1xh[4] += u0xp[4] * eb[4];
                    u0xh[7] += u1xp[7] * eb[4];
                    u1xh[7] += u0xp[7] * eb[4];

                    // k12 computation
                    u1xh[1] += eh[5];
                    u1xh[3] += eh[5];
                    //   nonlinear input * h terms
                    u0xh[0] += u1x[1] * eh[5];
                    u1xh[0] += u0x[1] * eh[5];
                    u0xh[1] += u1x[0] * eh[5];
                    u1xh[1] += u0x[0] * eh[5];
                    u0xh[3] += u1x[4] * eh[5];
                    u1xh[3] += u0x[4] * eh[5];
                    u0xh[4] += u1x[3] * eh[5];
                    u1xh[4] += u0x[3] * eh[5];
                    u0xh[6] += u1x[7] * eh[5];
                    u1xh[6] += u0x[7] * eh[5];
                    u0xh[7] += u1x[6] * eh[5];
                    u1xh[7] += u0x[6] * eh[5];
                    //   nonlinear bar * ptest terms
                    u0xh[0] += u1xp[1] * eb[5];
                    u1xh[0] += u0xp[1] * eb[5];
                    u0xh[1] += u1xp[0] * eb[5];
                    u1xh[1] += u0xp[0] * eb[5];
                    u0xh[3] += u1xp[4] * eb[5];
                    u1xh[3] += u0xp[4] * eb[5];
                    u0xh[4] += u1xp[3] * eb[5];
                    u1xh[4] += u0xp[3] * eb[5];
                    u0xh[6] += u1xp[7] * eb[5];
                    u1xh[6] += u0xp[7] * eb[5];
                    u0xh[7] += u1xp[6] * eb[5];
                    u1xh[7] += u0xp[6] * eb[5];
                } else {
                    u1xh[0] = eh[3];  // k11
                    u1xh[4] = eh[4];  // k22
                    u1xh[1] = eh[5];  // k12
                    u1xh[3] = eh[5];  // k12
                }
            }  // end of compute u0xb, u1xb block, strains and stresses unloaded
        }      // u0x, u1x forward disp grads unloaded

        constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
        constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

        T d_bar[3];
        // scope for u0d_barT
        {
            T u0d_barT[9], tmp[9];

            // u0d_bar^t = XdinvT * u0x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xh, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

            // u0d_bar^t += XdinvzT * u1x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvzT, u1xh, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

            Basis::template interpFieldsTranspose<3, 3>(pt, &u0d_barT[6], d_bar);
            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0d_barT[0],
                                                                        &u0d_barT[3], mat_col);
        }  // end of u0d_barT scope

        // scope for u1d_barT
        {
            T u1d_barT[9], tmp[9];

            // u1d_barT^t = XdinvT * u1x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u1xh, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u1d_barT);

            // prev : cols of u1d are {d0xi, d0eta, zero} => extract from rows of u1d_bar^T => d_bar
            Basis::template interpFieldsGradTranspose<3, 3>(pt, &u1d_barT[0], &u1d_barT[3], d_bar);
        }  // end of u0d_barT scope

        // transfer back through directors
        {
            T fn[12];

            ShellComputeNodeNormals<T, Basis>(xpts, fn);  // recompute fn here
            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar, mat_col);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_gmat_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars0[dof_per_elem], const T path[dof_per_elem], const Data physData,
        T matCol[dof_per_elem]) {
        if (!active_thread) return;

        // compute the stability matrix G
        using T2 = A2D::ADScalar<T, 1>;

        // copy variables over and set
        // directional deriv in for path deriv of vars
        T2 vars[dof_per_elem];
        for (int i = 0; i < dof_per_elem; i++) {
            vars[i].value = vars0[i];
            vars[i].deriv = path[i];
        }

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T scale;              // scale for energy derivatives
        T d[3 * num_nodes];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Mat<T2, 3, 3>> u0x, u1x;
        A2D::A2DObj<A2D::SymMat<T2, 3>> e0ty;
        A2D::A2DObj<A2D::Vec<T2, 1>> et;

        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // TODO : need to add flexible type for all vars-related like et
            // here needs to be T2 type

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // TODO : need to add flexible type for all vars-related like et
            // here needs to be T2 type

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // compute forward projection vectors (like hforward in A2D)
        // ---------------------------------------------------------
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        {
            // goal of this section is to compute pvalue()'s of u0x, u1x, e0ty,
            // et in order to do projected Hessian reversal
            // TODO : if nonlinear, may need to recompute / store some projected
            // hessians to reverse also TODO : may need to write dot version of
            // these guys if some formulations are nonlinear (if linear same
            // method call)

            // compute the projected drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, et.pvalue().get_data());

            // compute projected directors
            T p_d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                         p_d);

            // compute tying strain projection
            T p_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                p_vars.get_data(), p_d, p_ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, p_ety,
                u0x.pvalue().get_data(), u1x.pvalue().get_data(), e0ty.pvalue());
        }

        // now we have pvalue()'s set into each in/out var => get projected
        // hessian hvalue()'s now with reverse mode AD below
        Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, e0ty, et);

        // residual backprop section (1st order derivs)
        {
            A2D::Vec<T, 3 * num_nodes> d_bar;
            T ety_bar[Basis::num_all_tying_points];
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), matCol, d_bar.get_data(), ety_bar);

            // drill strain sens
            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), matCol);

            // backprop tying strain sens ety_bar to d_bar and res
            computeTyingStrainSens<Phys, Basis>(xpts, fn, vars, d, ety_bar, matCol,
                                                d_bar.get_data());

            // directors back to residuals
            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                             matCol);

            // TODO : rotation constraint sens for some director classes (zero
            // for linear rotation)
        }  // end of 1st order deriv section

        // proj hessian backprop section (2nd order derivs)
        // -----------------------------------------------------
        {
            // TODO : make method versions of these guys for proj hessians
            // specifically (if nonlinear terms)
            A2D::Vec<T, 3 * num_nodes> d_hbar;
            T ety_hbar[Basis::num_all_tying_points];
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), e0ty.hvalue(), matCol, d_hbar.get_data(), ety_hbar);

            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);

            computeTyingStrainHrev<Phys, Basis>(xpts, fn, vars, d, ety_hbar, matCol,
                                                d_hbar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_hbar.get_data(),
                                                                             matCol);

            // TODO : rotation constraint sens for some director classes (zero
            // for linear rotation)
        }  // end of 2nd order deriv section

    }  // add_element_quadpt_gmat_col

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_adj_res_product(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T psi[dof_per_elem], T loc_dv_sens[])

    {
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // intermediate strain states
        A2D::Mat<T, 3, 3> u0x, u1x, psi_u0x, psi_u1x;
        A2D::SymMat<T, 3> e0ty, psi_e0ty;
        A2D::Vec<T, 1> et, psi_et;
        T scale = 0.0, detXd = 0.0;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        T d[3 * num_nodes];
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.get_data(), u1x.get_data(), e0ty);

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // hforward scope block for psi[u] => psi[E] the strain level equiv adjoint
        // ------------------------------------
        {
            // compute the interpolated drill strain
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, psi, fn, psi_et.get_data());

            // compute directors (linearized Hfwd)
            T psi_d[3 * num_nodes];
            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(psi, fn, psi_d);

            // compute tying strain (hfwd version, so linearized)
            T psi_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, psi, psi_d, psi_ety);

            // compute all shell displacement gradients (linearized Hfwd version)
            ShellComputeDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, psi, fn, psi_d, psi_ety, psi_u0x.get_data(),
                psi_u1x.get_data(), psi_e0ty);

        }  // end of hforward scope block for psi
        // ------------------------------------

        // want psi[u]^T d^2Pi/du/dx = psi[E]^T d^2Pi/dE/dx
        // instead of backprop sensitivities, hfwd and compute product on the strains
        Phys::template compute_strain_adjoint_res_product<T>(
            physData, scale, u0x, u1x, e0ty, et, psi_u0x, psi_u1x, psi_e0ty, psi_et, loc_dv_sens);

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
        const Data &physData, A2D::Mat<T, 3, 3> &u0x, A2D::Mat<T, 3, 3> &u1x,
        A2D::SymMat<T, 3> &e0ty, A2D::Vec<T, 1> &et) {
        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strains
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            T d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.get_data(), u1x.get_data(), e0ty);

        }  // end of forward scope block for strains
        // ------------------------------------------------
    }

    template <class Data>
    __HOST_DEVICE__ static void get_element_quadpt_failure_index(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        T &fail_index) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndex(physData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                           fail_index);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_dv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T loc_dv_sens[]) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndexDVSens(physData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                                 fail_sens, loc_dv_sens);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_sv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T dfdu_local[]) {
        if (!active_thread) return;

        // if (threadIdx.x == 0 && threadIdx.y == 3 && blockIdx.x == 0) {
        //     printf("xpts:");
        //     printVec<T>(12, xpts);
        //     printf("vars:");
        //     printVec<T>(24, vars);
        //     printf("rhoKS %.4e, SF %.4e\n", rhoKS, safetyFactor);
        // }

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        T d[3 * num_nodes];
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Phys::template computeFailureIndexSVSens<T>(physData, rhoKS, safetyFactor, fail_sens, u0x,
                                                    u1x, e0ty, et);

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
        // T ety_bar[Basis::num_all_tying_points];
        ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            e0ty.bvalue(), dfdu_local, d_bar.get_data(), ety_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), dfdu_local,
                                               d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                         dfdu_local);

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), dfdu_local);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T strains[vars_per_node])

    {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Vec<T, 9>> E, S;
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        T scale = 1.0;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x.value(), u1x.value(),
                                             e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(physData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Ef = E.value();
        for (int i = 0; i < 6; i++) {
            strains[i] = Ef[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_stresses(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T stresses[vars_per_node]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Vec<T, 9>> E, S;
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        T scale = 1.0;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x.value(), u1x.value(),
                                             e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(physData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Sf = S.value();
        for (int i = 0; i < 6; i++) {
            stresses[i] = Sf[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_mass(bool active_thread, int iquad,
                                                            const T xpts[], const Data physData,
                                                            T *output) {
        // compute int[rho * thick] dA
        if (!active_thread) return;

        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // compute shell node normals
        T fn[3 * Basis::num_nodes];  // node normals
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute detXd to transform dxideta to dA
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        T Xd[9];
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // compute area density quadpt contribution (then int across area with element sums)
        *output = weight * detXd * physData.rho * physData.thick;
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_dmass_dx(bool active_thread, int iquad,
                                                                const T xpts[], const Data physData,
                                                                T *dm_dxlocal) {
        // mass = int[rho * thick] dA summed over all elements
        // then return dmass/dx for x this element thickness
        // since one DV here just put in first
        if (!active_thread) return;

        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // compute shell node normals
        T fn[3 * Basis::num_nodes];  // node normals
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute detXd to transform dxideta to dA
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        T Xd[9];
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // only one local DV in isotropic shell (panel thickness)
        dm_dxlocal[0] = weight * detXd * physData.rho;
    }
};
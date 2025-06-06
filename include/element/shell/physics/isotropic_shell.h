#pragma once

#include "../a2dshell.h"
#include "../data/isotropic.h"
#include "a2dcore.h"

template <typename T, class Data_, bool isNonlinear = false>
class IsotropicShell {
   public:
    using Data = Data_;

    // ensure the data is only allowed to be ShellIsotropicData
    // static_assert(std::is_same<Data_, ShellIsotropicData>::value,
    //               "Error IsotropicShell physics class must use Data class 'ShellIsotropicData'");

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = 6;
    // whether strain is linear or nonlinear (in this case linear)
    static constexpr A2D::ShellStrainType STRAIN_TYPE =
        isNonlinear ? A2D::ShellStrainType::NONLINEAR : A2D::ShellStrainType::LINEAR;
    static constexpr bool is_nonlinear = isNonlinear;
    static constexpr int num_dvs = 1;

    // function declarations
    // -------------------------------------------------------

    /**
    template <typename T2>
    __HOST_DEVICE__ static void computeStrainEnergy(const Data physData, const T scale,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u0x,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u1x,
                                                    A2D::ADObj<A2D::SymMat<T2, 3>> e0ty,
                                                    A2D::ADObj<A2D::Vec<T2, 1>> et,
                                                    A2D::ADObj<T2> &Uelem);

    template <typename T2>
    __HOST_DEVICE__ static void computeWeakRes(const Data &physData, const T &scale,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                               A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                               A2D::ADObj<A2D::Vec<T2, 1>> &et);

    template <typename T2>
    __HOST_DEVICE__ static void computeWeakJacobianCol(const Data &physData, const T &scale,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                       A2D::A2DObj<A2D::SymMat<T2, 3>> &e0ty,
                                                       A2D::A2DObj<A2D::Vec<T2, 1>> &et);

                                                       template <typename T2>
    __HOST_DEVICE__ static void computeQuadptSectionalLoads(const Data &physData, const T &scale,
                                                            A2D::Mat<T2, 3, 3> &u0x,
                                                            A2D::Mat<T2, 3, 3> &u1x,
                                                            A2D::SymMat<T2, 3> &e0ty,
                                                            A2D::Vec<T2, 1> &et, A2D::Vec<T, 9> S);

                                                            template <typename T2>
    __HOST_DEVICE__ static void computeQuadptStrains(const Data &physData, const T &scale,
                                                     A2D::Mat<T2, 3, 3> &u0x,
                                                     A2D::Mat<T2, 3, 3> &u1x,
                                                     A2D::SymMat<T2, 3> &e0ty, A2D::Vec<T2, 1> &et,
                                                     A2D::Vec<T, 9> E);

    __HOST_DEVICE__
    static void computeKSFailure(const Data &data, T rho_KS, T strains[vars_per_node],
                                 T *fail_index);
     */

    // -------------------------------------------------------
    // end of function declarations

    template <typename T2>
    __HOST_DEVICE__ static void computeStrainEnergy(const Data physData, const T scale,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u0x,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u1x,
                                                    A2D::ADObj<A2D::SymMat<T2, 3>> e0ty,
                                                    A2D::ADObj<A2D::Vec<T2, 1>> et,
                                                    A2D::ADObj<T2> &Uelem) {
        A2D::ADObj<A2D::Vec<T2, 9>> E, S;
        A2D::ADObj<T2> ES_dot;

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // A2D::VecScale(1.0, E, S), // debugging statement
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(0.5 * scale) * ES_dot, Uelem));
        // printf("Uelem = %.8e\n", Uelem.value());

    }  // end of computeStrainEnergy

    // could template by ADType = ADObj or A2DObj later to allow different
    // derivative levels maybe
    template <typename T2>
    __HOST_DEVICE__ static void computeWeakRes(const Data &physData, const T &scale,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                               A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                               A2D::ADObj<A2D::Vec<T2, 1>> &et) {
        // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
        A2D::ADObj<A2D::Vec<T2, 9>> E, S;
        A2D::ADObj<T2> ES_dot, Uelem;
        // isotropicShellStress expression uses many fewer floats than storing ABD
        // matrix

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                           // the 0.5 correct?]
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
        // printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void computeWeakJacobianCol(const Data &physData, const T &scale,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                       A2D::A2DObj<A2D::SymMat<T2, 3>> &e0ty,
                                                       A2D::A2DObj<A2D::Vec<T2, 1>> &et) {
        // computes a projected Hessian (or jacobian column)

        // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
        A2D::A2DObj<A2D::Vec<T2, 9>> E, S;
        A2D::A2DObj<T2> ES_dot, Uelem;

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                           // the 0.5 correct?]
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
        // note TACS differentiates based on 2 * Uelem here.. hmm
        // printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.hproduct();  // computes projected hessians
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    }  // end of JacobianCol

    template <typename T2>
    __HOST_DEVICE__ static void compute_strain_adjoint_res_product(
        const Data &physData, const T &scale, const A2D::Mat<T2, 3, 3> &u0x,
        const A2D::Mat<T2, 3, 3> &u1x, const A2D::SymMat<T2, 3> &e0ty, const A2D::Vec<T2, 1> &et,
        const A2D::Mat<T2, 3, 3> &psi_u0x, const A2D::Mat<T2, 3, 3> &psi_u1x,
        const A2D::SymMat<T2, 3> &psi_e0ty, const A2D::Vec<T2, 1> &psi_et, T loc_dv_sens[]) {
        /* compute psi[E]^T * d^2Pi/dE/dx product at strain level (equiv to back at disp level) */
        A2D::Vec<T2, 9> E;
        A2D::Vec<T2, 9> psi_E;

        // first compute strains on the regular disps
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        // then compute adjoint strains using (linearized hfwd)
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainForwardCore<T>(psi_u0x.get_data(), psi_u1x.get_data(),
                                                 psi_e0ty.get_data(), psi_et.get_data(),
                                                 psi_E.get_data());
        } else {
            A2D::NonlinearShellStrainForwardCore<T>(psi_u0x.get_data(), psi_u1x.get_data(),
                                                    psi_e0ty.get_data(), psi_et.get_data(),
                                                    psi_E.get_data());
        }

        // then compute psi_E^T dEbar/dx the strain sensitivity design var sens for each design var
        physData.evalStrainDVSensProduct(scale, E.get_data(), psi_E.get_data(), loc_dv_sens);

    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndex(const Data physData, A2D::Mat<T2, 3, 3> u0x,
                                                    A2D::Mat<T2, 3, 3> u1x, A2D::SymMat<T2, 3> e0ty,
                                                    A2D::Vec<T2, 1> et, const T &rhoKS,
                                                    T &fail_index) {
        A2D::Vec<T2, 9> E;

        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        fail_index = physData.evalFailure(rhoKS, E.get_data());

    }  // end of computeFailureIndex

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndexDVSens(
        const Data physData, A2D::Mat<T2, 3, 3> u0x, A2D::Mat<T2, 3, 3> u1x,
        A2D::SymMat<T2, 3> e0ty, A2D::Vec<T2, 1> et, const T &rhoKS, const T &scale, T dv_sens[]) {
        /* compute df/dx constribution to ks failure */
        A2D::Vec<T2, 9> E;
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        physData.evalFailureDVSens(rhoKS, E.get_data(), scale, dv_sens);

    }  // end of computeFailureIndexDVSens

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndexSVSens(
        const Data physData, A2D::ADObj<A2D::Mat<T2, 3, 3>> u0x, A2D::ADObj<A2D::Mat<T2, 3, 3>> u1x,
        A2D::ADObj<A2D::SymMat<T2, 3>> e0ty, A2D::ADObj<A2D::Vec<T2, 1>> et, const T &rhoKS,
        const T &scale, T dv_sens[]) {
        /* compute df/du RHS constribution to ks failure */
        A2D::ADObj<A2D::Vec<T2, 9>> E;
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.value().get_data(), u1x.value().get_data(),
                                          e0ty.value().get_data(), et.value().get_data(),
                                          E.value().get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.value().get_data(), u1x.value().get_data(),
                                             e0ty.value().get_data(), et.value().get_data(),
                                             E.value().get_data());
        }

        // go from E forward to E backwards (the strain)
        physData.evalFailureStrainSens(scale, rhoKS, E.value().get_data(), E.bvalue().get_data());

        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainReverseCore<T>(u0x.bvalue().get_data(), u1x.bvalue().get_data(),
                                                 e0ty.bvalue().get_data(), et.bvalue().get_data(),
                                                 E.bvalue().get_data());
        } else {
            A2D::NonlinearShellStrainReverseCore<T>(
                u0x.bvalue().get_data(), u1x.bvalue().get_data(), e0ty.bvalue().get_data(),
                et.bvalue().get_data(), E.bvalue().get_data());
        }

    }  // end of computeFailureIndexDVSens
};
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
    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void computeQuadptSectionalLoads(const Data &physData, const T &scale,
                                                            A2D::Mat<T2, 3, 3> &u0x,
                                                            A2D::Mat<T2, 3, 3> &u1x,
                                                            A2D::SymMat<T2, 3> &e0ty,
                                                            A2D::Vec<T2, 1> &et, A2D::Vec<T, 9> S) {
        A2D::ADObj<A2D::Vec<T, 9>> E;

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S));

        // technically the IsotropicShellStress computes sectional stresses aka sectional loads
    }  // end of computeQuadptSectionalLoads

    template <typename T2>
    __HOST_DEVICE__ static void computeQuadptStrains(const Data &physData, const T &scale,
                                                     A2D::Mat<T2, 3, 3> &u0x,
                                                     A2D::Mat<T2, 3, 3> &u1x,
                                                     A2D::SymMat<T2, 3> &e0ty, A2D::Vec<T2, 1> &et,
                                                     A2D::Vec<T, 9> E) {
        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E));
    }  // end of computeQuadptStrains

    template <typename T2>
    __HOST_DEVICE__ static void computeQuadptStrainsSens(const Data &physData, const T &scale,
                                                         A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                         A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                         A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                         A2D::ADObj<A2D::Vec<T2, 1>> &et,
                                                         T strain_bar[9]) {
        A2D::Vec<T, 9> E;
        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E));
        T *Eb = E.bvalue();
        for (int i = 0; i < 6; i++) {
            Eb[i] = strain_bar[i];
        }
        strain_energy_stack.reverse();
    }  // end of computeQuadptStrains

    __HOST_DEVICE__ static void vonMisesFailure2D(const Data &data, const T stress[3], T *failure) {
        // stress[3] refers to s11, s22, s12; ys refers to yield stress
        return sqrt(stress[0] * stress[0] + stress[1] * stress[1] - stress[0] * stress[1] +
                    3.0 * stress[2] * stress[2]) /
               data.ys;
    }

    __HOST_DEVICE__ static void vonMisesFailure2DSens(const Data &data, const T stress[3],
                                                      const T &fail_bar, T *stress_bar) {
        // stress[3] refers to s11, s22, s12; ys refers to yield stress
        T my_sqrt = sqrt(stress[0] * stress[0] + stress[1] * stress[1] - stress[0] * stress[1] +
                         3.0 * stress[2] * stress[2]);
        T factor = 0.5 * fail_bar / my_sqrt / data.ys;

        stress_bar[0] = factor * (2 * stress[0] - stress[1]);
        stress_bar[1] = factor * (2 * stress[1] - stress[0]);
        stress_bar[2] = factor * 6 * stress[2];
    }

    __HOST_DEVICE__
    static void computeKSFailure(const Data &data, T rho_KS, T strains[vars_per_node],
                                 T *fail_index) {
        // compute the von mises ksfailure index in the shell
        // strains are the 'nodal_strains' 6 of them

        // upper and lower surface thicknesses in z the normal coordinate
        T zU = (0.5 - data.tOffset) * data.thick;
        T zL = -(0.5 + data.tOffset) * data.thick;

        // compute the upper and lower surface strains
        T strainU[3], strainL[3];
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zU, &strains[3], strainU);
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zL, &strains[3], strainL);

        // compute the 2D tangent stiffness matrix C or Q
        T C[6];
        Data::evalTangentStiffness2D(data.E, data.nu, C);

        // compute upper and lower surface stresses
        T stressU[3], stressL[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainL, stressL);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainU, stressU);

        // note stress[3] refers to s11, s22, s12 stresses
        // compute von mises failure at upper and lower surfaces
        T failU, failL;
        vonMisesFailure2D(data, stressU, &failU);
        vonMisesFailure2D(data, stressL, &failL);

        // now do KS max among upper and lower surface stresses
        T ksMax;
        if (failU > failL) {
            ksMax = failU;
        } else {
            ksMax = failL;
        }

        T ksSum = exp(rho_KS * (failU - ksMax)) + exp(rho_KS * (failL - ksMax));
        T ksVal = ksMax + log(ksSum) / rho_KS;
        return ksVal;
    }

    __HOST_DEVICE__
    static void computeKSFailureDVSens(const Data &data, T rho_KS, const T strains[vars_per_node],
                                       const T fail_index_bar, T *x_bar) {
        // compute the von mises ksfailure index in the shell
        // strains are the 'nodal_strains' 6 of them
        // x_bar is a scalar

        // upper and lower surface thicknesses in z the normal coordinate
        T zU = (0.5 - data.tOffset) * data.thick;
        T zL = -(0.5 + data.tOffset) * data.thick;

        // compute the upper and lower surface strains
        T strainU[3], strainL[3];
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zU, &strains[3], strainU);
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zL, &strains[3], strainL);

        // compute the 2D tangent stiffness matrix C or Q
        T C[6];
        Data::evalTangentStiffness2D(data.E, data.nu, C);

        // compute upper and lower surface stresses
        T stressU[3], stressL[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainL, stressL);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainU, stressU);

        // note stress[3] refers to s11, s22, s12 stresses
        // compute von mises failure at upper and lower surfaces
        T failU, failL;
        vonMisesFailure2D(data, stressU, &failU);
        vonMisesFailure2D(data, stressL, &failL);

        // now do KS max among upper and lower surface stresses
        T ksMax;
        if (failU > failL) {
            ksMax = failU;
        } else {
            ksMax = failL;
        }

        T ksSum = exp(rho_KS * (failU - ksMax)) + exp(rho_KS * (failL - ksMax));
        T ksVal = ksMax + log(ksSum) / rho_KS;

        // end of forward analysis
        // backprop now first to dsigma_KS/dthick through zL and zU
        x_bar = 0.0;  // init the scalar to zero from this quadpt

        // first for lower stresses
        T fail_bar = exp(rho_KS * (failL - ksMax)) / ksSum;
        T stress_bar[3];
        vonMisesFailure2DSens(data, stressL, fail_bar, stress_bar);
        T surf_strain_bar[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);
        T zL_bar = A2D::VecDotCore<T, 3>(surf_strain_bar, &strains[3]);
        x_bar += zL_bar * -0.5;

        // then for upper stresses
        fail_bar = exp(rho_KS * (failU - ksMax)) / ksSum;
        vonMisesFailure2DSens(data, stressU, fail_bar, stress_bar);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);
        T zU_bar = A2D::VecDotCore<T, 3>(surf_strain_bar, &strains[3]);
        x_bar += zU_bar * 0.5;

        return ksVal;
    }

    __HOST_DEVICE__
    static void computeKSFailureSVSens(const Data &data, T rho_KS, const T strains[vars_per_node],
                                       const T fail_index_bar, T strain_bar[vars_per_node]) {
        // compute the von mises ksfailure index in the shell
        // strains are the 'nodal_strains' 6 of them
        // x_bar is a scalar

        // upper and lower surface thicknesses in z the normal coordinate
        T zU = (0.5 - data.tOffset) * data.thick;
        T zL = -(0.5 + data.tOffset) * data.thick;

        // compute the upper and lower surface strains
        T strainU[3], strainL[3];
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zU, &strains[3], strainU);
        A2D::VecSumCore<T, 3>(1.0, &strains[0], zL, &strains[3], strainL);

        // compute the 2D tangent stiffness matrix C or Q
        T C[6];
        Data::evalTangentStiffness2D(data.E, data.nu, C);

        // compute upper and lower surface stresses
        T stressU[3], stressL[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainL, stressL);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strainU, stressU);

        // note stress[3] refers to s11, s22, s12 stresses
        // compute von mises failure at upper and lower surfaces
        T failU, failL;
        vonMisesFailure2D(data, stressU, &failU);
        vonMisesFailure2D(data, stressL, &failL);

        // now do KS max among upper and lower surface stresses
        T ksMax;
        if (failU > failL) {
            ksMax = failU;
        } else {
            ksMax = failL;
        }

        T ksSum = exp(rho_KS * (failU - ksMax)) + exp(rho_KS * (failL - ksMax));
        T ksVal = ksMax + log(ksSum) / rho_KS;

        // end of forward analysis
        // backprop now first to dsigma_KS/dthick through zL and zU

        // first for lower stresses
        T fail_bar = exp(rho_KS * (failL - ksMax)) / ksSum;
        T stress_bar[3];
        vonMisesFailure2DSens(data, stressL, fail_bar, stress_bar);
        T surf_strain_bar[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);

        // backprop from lower surf strains to strain_bar
        A2D::VecSumCore<T, 3>(surf_strain_bar, &strain_bar[0]);
        A2D::VecSumCore<T, 3>(zL, surf_strain_bar, &strain_bar[3]);

        // then for upper stresses
        fail_bar = exp(rho_KS * (failU - ksMax)) / ksSum;
        vonMisesFailure2DSens(data, stressU, fail_bar, stress_bar);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, stress_bar, surf_strain_bar);

        // backprop from upper surf strains to strain_bar
        A2D::VecSumCore<T, 3>(surf_strain_bar, &strain_bar[0]);
        A2D::VecSumCore<T, 3>(zU, surf_strain_bar, &strain_bar[3]);

        return ksVal;
    }

    __HOST_DEVICE__ static void compute_dRdx(const int local_dv, const Data &physData,
                                             const T &scale, A2D::ADObj<A2D::Mat<T, 3, 3>> &u0x,
                                             A2D::ADObj<A2D::Mat<T, 3, 3>> &u1x,
                                             A2D::ADObj<A2D::SymMat<T, 3>> &e0ty,
                                             A2D::ADObj<A2D::Vec<T, 1>> &et) {
        // for isotropic material only one local_dv aka thickness
        // so ignore this input

        using T2 = A2D::ADScalar<T, 1>;

        // copy into local
        A2D::ADObj<A2D::Mat<T2, 3, 3>> _u0x, _u1x;
        A2D::ADObj<A2D::SymMat<T2, 3>> _e0ty;
        A2D::ADObj<A2D::Vec<T2, 1>> _et;

        for (int i = 0; i < 9; i++) {
            _u0x.value()[i].value = u0x.value()[i];
            _u1x.value()[i].value = u1x.value()[i];
        }
        for (int i = 0; i < 6; i++) {
            _e0ty.value()[i].value = e0ty.value()[i];
        }
        _et.value()[0].value = et.value()[0];

        // use forward AD to get dstrain/dthick?
        A2D::ADObj<A2D::Vec<T2, 9>> E, S;
        A2D::ADObj<T2> ES_dot, Uelem;

        // some way to check that T2 is indeed ADScalar<> type?
        T2 thick;
        if (local_dv == 0) {
            thick = physData.thick;
            thick.deriv = 1.0;
        }

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T2, Data>(T2(physData.E), T2(physData.nu),
                                                               thick, physData.tOffset, E, S),
                           // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                           // the 0.5 correct?]
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
        // printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated

        // now copy back data from ADScalar to regular types
        for (int i = 0; i < 9; i++) {
            u0x.value()[i] = _u0x.value()[i].deriv;
            u1x.value()[i] = _u1x.value()[i].deriv;
        }
        for (int i = 0; i < 6; i++) {
            e0ty.value()[i] = _e0ty.value()[i].deriv;
        }
        et.value()[0] = _et.value()[0].deriv;
    }  // end of computeWeakRes
};
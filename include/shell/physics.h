#pragma once

#include "a2dcore.h"

template <typename T, class Data_>
class LinearShell {
  public:
    using Data = Data_;

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = 6;
    // whether strain is linear or nonlinear (in this case linear) 
    static constexpr A2D::ShellStrainType STRAIN_TYPE = A2D::ShellStrainType::LINEAR;

    // could template by ADType = ADObj or A2DObj later to allow different derivative levels maybe
    template <typename T2>
    __HOST_DEVICE__ static void computeWeakRes(
        const Data physData, const T scale, 
        A2D::ADObj<A2D::Mat<T2,3,3>> u0x, A2D::ADObj<A2D::Mat<T2,3,3>> u1x,
        A2D::ADObj<A2D::SymMat<T2,3>> e0ty, A2D::ADObj<A2D::Vec<T2,1>> et
    ) {

        // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
        A2D::ADObj<A2D::Vec<T2,9>> E, S;
        A2D::ADObj<T2> ES_dot, Uelem;
        // isotropicShellStress expression uses many fewer floats than storing ABD matrix

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack = A2D::MakeStack(
            A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
            A2D::IsotropicShellStress<T, Data>(physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
            // A2D::VecScale(1.0, E, S), // debugging statement
            A2D::VecDot(E, S, ES_dot),
            A2D::Eval(T2(0.5 * scale) * ES_dot, Uelem)
        ); 
        printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    } // end of computeWeakRes

    template <class Basis>
    __HOST_DEVICE__ static void computeTyingStrain(
        const T Xpts[], const T fn[], const T vars[], 
        const T d[], T ety[]) {

        // using unrolled loop here for efficiency (if statements and for loops not great for device)
        int32_t offset, num_tying;

        // get g11 tying strain
        // ------------------------------------
        offset = Basis::tying_point_offsets(0);
        num_tying = Basis::num_tying_points(0);
        #pragma unroll // for low num_tying can speed up?
        for (int itying = 0; itying < num_tying; itying++) {
            T pt[2];
            Basis::template getTyingPoint<0>(itying, pt);

            // Interpolate the field value
            T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

            // store g11 strain
            ety[offset + itying] = A2D::VecDotCore<T,3>(Uxi, Xxi);
        } // end of itying for loop for g11

        // get g12 strain
        // ------------------------------------
        offset = Basis::tying_point_offsets(1);
        num_tying = Basis::num_tying_points(1);
        #pragma unroll // for low num_tying can speed up?
        for (int itying = 0; itying < num_tying; itying++) {
            T pt[2];
            Basis::template getTyingPoint<1>(itying, pt);

            // Interpolate the field value
            T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

            // store g12 strain
            ety[offset + itying] = 0.5 * (A2D::VecDotCore<T,3>(Uxi, Xeta) + A2D::VecDotCore<T,3>(Ueta, Xxi));
        } // end of itying for loop for g12

        // get g13 strain
        // ------------------------------------
        offset = Basis::tying_point_offsets(2);
        num_tying = Basis::num_tying_points(2);
        #pragma unroll // for low num_tying can speed up?
        for (int itying = 0; itying < num_tying; itying++) {
            T pt[2];
            Basis::template getTyingPoint<2>(itying, pt);

            // Interpolate the field value
            T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

            T d0[3], n0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            Basis::template interpFields<3, 3>(pt, fn, n0);

            // store g13 strain
            ety[offset + itying] = 0.5 * (A2D::VecDotCore<T,3>(Xxi, d0) + A2D::VecDotCore<T,3>(n0, Uxi));
        } // end of itying for loop for g13

        // get g22 strain
        // ------------------------------------
        offset = Basis::tying_point_offsets(3);
        num_tying = Basis::num_tying_points(3);
        #pragma unroll // for low num_tying can speed up?
        for (int itying = 0; itying < num_tying; itying++) {
            T pt[2];
            Basis::template getTyingPoint<3>(itying, pt);

            // Interpolate the field value
            T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

            // store g22 strain
            ety[offset + itying] = A2D::VecDotCore<T,3>(Ueta, Xeta);
        } // end of itying for loop for g22

        // get g23 strain
        // ------------------------------------
        offset = Basis::tying_point_offsets(4);
        num_tying = Basis::num_tying_points(4);
        #pragma unroll // for low num_tying can speed up?
        for (int itying = 0; itying < num_tying; itying++) {
            T pt[2];
            Basis::template getTyingPoint<4>(itying, pt);

            // Interpolate the field value
            T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
            Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

            T d0[3], n0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            Basis::template interpFields<3, 3>(pt, fn, n0);

            // store g23 strain
            ety[offset + itying] = 0.5 * (A2D::VecDotCore<T,3>(Xeta, d0) + A2D::VecDotCore<T,3>(n0, Ueta));
        } // end of itying for loop for g13

    } // end of computeTyingStrain

}; 
#pragma once

#include "a2dcore.h"
#include "a2dshell.h"

template <typename T, class Data_, bool isNonlinear = false>
class IsotropicShell
{
public:
  using Data = Data_;

  // u, v, w, thx, thy, thz
  static constexpr int32_t vars_per_node = 6;
  // whether strain is linear or nonlinear (in this case linear)
  static constexpr A2D::ShellStrainType STRAIN_TYPE =
      isNonlinear ? A2D::ShellStrainType::NONLINEAR
                  : A2D::ShellStrainType::LINEAR;

  template <typename T2>
  __HOST_DEVICE__ static void computeStrainEnergy(
      const Data physData, const T scale, A2D::ADObj<A2D::Mat<T2, 3, 3>> u0x,
      A2D::ADObj<A2D::Mat<T2, 3, 3>> u1x, A2D::ADObj<A2D::SymMat<T2, 3>> e0ty,
      A2D::ADObj<A2D::Vec<T2, 1>> et, A2D::ADObj<T2> &Uelem)
  {
    A2D::ADObj<A2D::Vec<T2, 9>> E, S;
    A2D::ADObj<T2> ES_dot;

    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack = A2D::MakeStack(
        A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
        A2D::IsotropicShellStress<T, Data>(
            physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
        // A2D::VecScale(1.0, E, S), // debugging statement
        A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(0.5 * scale) * ES_dot, Uelem));
    // printf("Uelem = %.8e\n", Uelem.value());

  } // end of computeStrainEnergy

  // could template by ADType = ADObj or A2DObj later to allow different
  // derivative levels maybe
  template <typename T2>
  __HOST_DEVICE__ static void computeWeakRes(
      const Data &physData, const T &scale, A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
      A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x, A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
      A2D::ADObj<A2D::Vec<T2, 1>> &et)
  {
    // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
    A2D::ADObj<A2D::Vec<T2, 9>> E, S;
    A2D::ADObj<T2> ES_dot, Uelem;
    // isotropicShellStress expression uses many fewer floats than storing ABD
    // matrix

    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack = A2D::MakeStack(
        A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
        A2D::IsotropicShellStress<T, Data>(
            physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
        // no 1/2 here to match TACS formulation (just scales eqns) [is removing the 0.5 correct?]
        A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
    // printf("Uelem = %.8e\n", Uelem.value());

    Uelem.bvalue() = 1.0;
    strain_energy_stack.reverse();
    // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
  } // end of computeWeakRes

  template <typename T2>
  __HOST_DEVICE__ static void computeWeakJacobianCol(
      const Data &physData, const T &scale, A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u0x,
      A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u1x, A2D::A2DObj<A2D::SymMat<T2, 3>> &e0ty,
      A2D::A2DObj<A2D::Vec<T2, 1>> &et)
  {
    // computes a projected Hessian (or jacobian column)

    // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
    A2D::A2DObj<A2D::Vec<T2, 9>> E, S;
    A2D::A2DObj<T2> ES_dot, Uelem;

    // use stack to compute shell strains, stresses and then to strain energy
    auto strain_energy_stack = A2D::MakeStack(
        A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
        A2D::IsotropicShellStress<T, Data>(
            physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
        // no 1/2 here to match TACS formulation (just scales eqns) [is removing the 0.5 correct?]
        A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(0.5 * scale) * ES_dot, Uelem));
    // printf("Uelem = %.8e\n", Uelem.value());

    Uelem.bvalue() = 1.0;
    strain_energy_stack.hproduct(); // computes projected hessians
    // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
  } // end of computeWeakRes

  template <class Basis>
  __HOST_DEVICE__ static void computeTyingStrain(const T Xpts[], const T fn[],
                                                 const T vars[], const T d[],
                                                 T ety[])
  {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<0>(itying, pt);

      // Interpolate the field value
      T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

      // store g11 strain
      ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);
    } // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<1>(itying, pt);

      // Interpolate the field value
      T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

      // store g22 strain
      ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);
    } // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<2>(itying, pt);

      // Interpolate the field value
      T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

      // store g12 strain
      ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) +
                                    A2D::VecDotCore<T, 3>(Ueta, Xxi));
    } // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<3>(itying, pt);

      // Interpolate the field value
      T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

      T d0[3], n0[3];
      Basis::template interpFields<3, 3>(pt, d, d0);
      Basis::template interpFields<3, 3>(pt, fn, n0);

      // store g23 strain
      ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) +
                                    A2D::VecDotCore<T, 3>(n0, Ueta));
    } // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<4>(itying, pt);

      // Interpolate the field value
      T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

      T d0[3], n0[3];
      Basis::template interpFields<3, 3>(pt, d, d0);
      Basis::template interpFields<3, 3>(pt, fn, n0);

      // store g13 strain
      ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) +
                                    A2D::VecDotCore<T, 3>(n0, Uxi));
    } // end of itying for loop for g13

  } // end of computeTyingStrain

  template <class Basis>
  __HOST_DEVICE__ static void computeTyingStrainSens(const T Xpts[],
                                                     const T fn[],
                                                     const T ety_bar[], T res[],
                                                     T d_bar[])
  {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<0>(itying, pt);
      //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

      T Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

      A2D::Vec<T, 3> Uxi_bar, zero;
      A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xxi, Uxi_bar.get_data());
      Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(), zero.get_data(),
                                                                  res);

    } // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<1>(itying, pt);
      //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

      T Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

      A2D::Vec<T, 3> Ueta_bar, zero;
      A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xeta, Ueta_bar.get_data());
      Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(), Ueta_bar.get_data(), res);

    } // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<2>(itying, pt);
      //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) +
      //                                 A2D::VecDotCore<T, 3>(Ueta, Xxi));

      T Xxi[3], Xeta[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

      A2D::Vec<T, 3> Uxi_bar, Ueta_bar;
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, Ueta_bar.get_data());
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, Uxi_bar.get_data());
      Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(), Ueta_bar.get_data(), res);
    } // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<3>(itying, pt);
      //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) +
      //                                 A2D::VecDotCore<T, 3>(n0, Ueta));

      T Xxi[3], Xeta[3], n0[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFields<3, 3>(pt, fn, n0);

      A2D::Vec<T, 3> zero, d0_bar, Ueta_bar;
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, d0_bar.get_data());
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Ueta_bar.get_data());
      Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(), Ueta_bar.get_data(), res);
      Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);
    } // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++)
    {
      T pt[2];
      Basis::template getTyingPoint<4>(itying, pt);
      //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) +
      //                                 A2D::VecDotCore<T, 3>(n0, Uxi));

      T Xxi[3], Xeta[3], n0[3];
      Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
      Basis::template interpFields<3, 3>(pt, fn, n0);

      A2D::Vec<T, 3> zero, Uxi_bar, d0_bar;
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, d0_bar.get_data());
      A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Uxi_bar.get_data());
      Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(), zero.get_data(), res);
      Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);

    } // end of itying for loop for g13

  } // end of computeTyingStrainSens
};
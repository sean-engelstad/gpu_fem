#pragma once

#include "quadrature.h"

#include "a2dcore.h"

template <typename T, class Quadrature_, A2D::GreenStrainType strainType>
class PlaneStressPhysics {
 public:
  using Quadrature = Quadrature_;

  // Variables at each node (u, v)
  static constexpr int32_t vars_per_node = 2;

  class IsotropicData {
    public:
    IsotropicData() : E(0.0), nu(0.0), t(0.0) {};
    IsotropicData(double E_, double nu_, double t_) : E(E_), nu(nu_), t(t_) {};
    double E, nu, t;
  };

  using Data = IsotropicData;
  using PlaneMat = typename A2D::Mat<T, 2, 2>;

  // template <
  __HOST_DEVICE__ static void computeWeakRes(Data physData, T scale, PlaneMat& dUdx_, PlaneMat& weak_res) {
    A2D::ADObj<PlaneMat> dUdx(dUdx_);
    A2D::ADObj<A2D::SymMat<T,2>> strain, stress;
    A2D::ADObj<T> energy;

    const double mu = 0.5 * physData.E / (1.0 + physData.nu);
    const double lambda = 2 * mu * physData.nu / (1.0 - physData.nu);

    auto strain_energy_stack = A2D::MakeStack(
      A2D::MatGreenStrain<strainType>(dUdx, strain),
      A2D::SymIsotropic(mu, lambda, strain, stress),
      A2D::SymMatMultTrace(strain, stress, energy)
    );

    // printf("energy = %.8e\n", energy.value());

    energy.bvalue() = 0.5 * scale * physData.t;
    strain_energy_stack.reverse();

    weak_res.copy(dUdx.bvalue());
  }

};
#pragma once
#include "a2dcore.h"
#include "quadrature.h"

template <typename T, class Quadrature_, int order = 2>
class ShellQuadBasis {
 public:
  using Quadrature = Quadrature_;

  // Required for loading solution data
  static constexpr int32_t num_nodes = order * order;
  static constexpr int32_t param_dim = 2;

  // MITC method => number of tying points for each strain component
  static constexpr int32_t mitcLevel2 = order * (order - 1);
  static constexpr int32_t mitcLevel3 = (order - 1) * (order - 1);
  static constexpr int32_t num_tying_components =
      5;  // five gij components for tying strains
  static constexpr int32_t num_all_tying_points = 4 * mitcLevel2 + mitcLevel3;

  __HOST_DEVICE__ static constexpr int32_t num_tying_points(int icomp) {
    // g11, g12, g13, g22, g23, g33=0 (not included g33)
    int32_t _num_tying_points[] = {mitcLevel2, mitcLevel3, mitcLevel2,
                                   mitcLevel2, mitcLevel2};
    return _num_tying_points[icomp];
  }

  __HOST_DEVICE__ static constexpr int32_t tying_point_offsets(int icomp) {
    // g11, g12, g13, g22, g23, g33=0 (not included g33)
    int32_t _tying_point_offsets[] = {0, mitcLevel2, mitcLevel2 + mitcLevel3,
                                      2 * mitcLevel2 + mitcLevel3,
                                      3 * mitcLevel2 + mitcLevel3};
    return _tying_point_offsets[icomp];
  }

  // isoperimetric has same #nodes geometry class inside it
  class LinearQuadGeo {
   public:
    // Required for loading nodal coordinates
    static constexpr int32_t spatial_dim = 2;

    // Required for knowning number of spatial coordinates per node
    static constexpr int32_t num_nodes = 4;

    // Number of quadrature points
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    // Data-size = spatial_dim * number of nodes
    // static constexpr int32_t geo_data_size = 5 * num_quad_pts;
  };  // end of class LinearQuadGeo
  using Geo = LinearQuadGeo;

  // shape functions here
  __HOST_DEVICE__ static void lagrangeLobatto1D(const T u, T *N) {
    if constexpr (order == 2) {
      N[0] = 0.5 * (1.0 - u);
      N[1] = 0.5 * (1.0 + u);
    }
  }

  template <int tyingOrder>
  __HOST_DEVICE__ static void lagrangeLobatto1D_tying(const T u, T *N) {
    if constexpr (tyingOrder == 1) {
      N[0] = 1.0;
    } else if constexpr (tyingOrder == 2) {
      N[0] = 0.5 * (1.0 - u);
      N[1] = 0.5 * (1.0 + u);
    }
  }

  __HOST_DEVICE__ static void lagrangeLobatto1DGrad(const T u, T *N, T *Nd) {
    if constexpr (order == 2) {
      N[0] = 0.5 * (1.0 - u);
      N[1] = 0.5 * (1.0 + u);

      // dN/du
      Nd[0] = -0.5;
      Nd[1] = 0.5;
    }
  }

  __HOST_DEVICE__ static void lagrangeLobatto2D(const T xi, const T eta, T *N) {
    // compute N_i(u) shape function values at each node
    if constexpr (order == 2) {
      T na[order], nb[order];
      lagrangeLobatto1D(xi, na);
      lagrangeLobatto1D(eta, nb);

      // now compute 2D N_a(xi) * N_b(eta) for each node
      for (int ieta = 0; ieta < order; ieta++) {
        for (int ixi = 0; ixi < order; ixi++) {
          N[order * ieta + ixi] = na[ixi] * nb[ieta];
        }
      }
    }
  }  // end of lagrangeLobatto2D

  __HOST_DEVICE__ static void lagrangeLobatto2DGrad(const T xi, const T eta,
                                                    T *N, T *dNdxi, T *dNdeta) {
    // compute N_i(u) shape function values at each node
    if constexpr (order == 2) {
      T na[order], nb[order];
      T dna[order], dnb[order];
      lagrangeLobatto1DGrad(xi, na, dna);
      lagrangeLobatto1DGrad(eta, nb, dnb);

      // now compute 2D N_a(xi) * N_b(eta) for each node
      for (int ieta = 0; ieta < order; ieta++) {
        for (int ixi = 0; ixi < order; ixi++) {
          N[order * ieta + ixi] = na[ixi] * nb[ieta];
          dNdxi[order * ieta + ixi] = dna[ixi] * nb[ieta];
          dNdeta[order * ieta + ixi] = na[ixi] * dnb[ieta];
        }
      }
    }
  }  // end of lagrangeLobatto2D_grad

  __HOST_DEVICE__ static void assembleFrame(const T a[], const T b[],
                                            const T c[], T frame[]) {
    for (int i = 0; i < 3; i++) {
      frame[3 * i] = a[i];
      frame[3 * i + 1] = b[i];
      frame[3 * i + 2] = c[i];
    }
  }

  __HOST_DEVICE__ static void extractFrame(const T frame[], T a[], T b[],
                                           T c[]) {
    for (int i = 0; i < 3; i++) {
      a[i] = frame[3 * i];
      b[i] = frame[3 * i + 1];
      c[i] = frame[3 * i + 2];
    }
  }

  __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
    // get xi, eta coordinates of each node point
    pt[0] = -1.0 + (2.0 / (order - 1)) * (n % order);
    pt[1] = -1.0 + (2.0 / (order - 1)) * (n / order);
  }

  template <int vars_per_node, int num_fields>
  __HOST_DEVICE__ static void interpFields(const T pt[], const T values[],
                                           T field[]) {
    T N[num_nodes];
    lagrangeLobatto2D(pt[0], pt[1], N);

    for (int ifield = 0; ifield < num_fields; ifield++) {
      field[ifield] = 0.0;
      for (int inode = 0; inode < num_nodes; inode++) {
        field[ifield] += N[inode] * values[inode * vars_per_node + ifield];
      }
    }
  }  // end of interpFields method

  template <int vars_per_node, int num_fields>
  __HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[],
                                               T dxi[], T deta[]) {
    T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
    lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

    for (int ifield = 0; ifield < num_fields; ifield++) {
      dxi[ifield] = 0.0;
      deta[ifield] = 0.0;
      for (int inode = 0; inode < num_nodes; inode++) {
        dxi[ifield] += dNdxi[inode] * values[inode * vars_per_node + ifield];
        deta[ifield] += dNdeta[inode] * values[inode * vars_per_node + ifield];
      }
    }
  }  // end of interpFieldsGrad method

  template <int vars_per_node, int num_fields>
  __HOST_DEVICE__ static void interpFieldsTranspose(const T pt[],
                                                    const T field_bar[],
                                                    T values_bar[]) {
    T N[num_nodes];  // TODO : double check this method (can we store less
                     // floats here also?)
    lagrangeLobatto2D(pt[0], pt[1], N);

    for (int ifield = 0; ifield < num_fields; ifield++) {
      for (int inode = 0; inode < num_nodes; inode++) {
        values_bar[inode * vars_per_node + ifield] += field[ifield] * N[inode];
      }
    }
  }  // end of interpFieldsTranspose method

  template <int vars_per_node, int num_fields>
  __HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[],
                                                        const T dxi_bar[],
                                                        const T deta_bar[],
                                                        T values_bar[]) {
    T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
    lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

    for (int ifield = 0; ifield < num_fields; ifield++) {
      for (int inode = 0; inode < num_nodes; inode++) {
        values_bar[inode * vars_per_node + ifield] +=
            dxi[ifield] * dNdxi[inode] + deta[ifield] * dNdeta[inode];
      }
    }
  }  // end of interpFieldsGrad method

  __HOST_DEVICE__ static void getTyingKnots(T red_knots[], T full_knots[]) {
    if constexpr (order == 2) {
      red_knots[0] = 0.0;
      full_knots[0] = -1.0;
      full_knots[1] = 1.0;
    }
  }

  // section for tying point interpolations
  template <int icomp>
  __HOST_DEVICE__ static void getTyingPoint(const int n, T pt[]) {
    T red_knots[order * (order - 1)], full_knots[order * order];
    getTyingKnots(red_knots, full_knots);

    // use constexpr here to prevent warp divergence
    if constexpr (icomp == 0 || icomp == 2) {  // g11 or g13
      // 1-dir uses reduced knots for 1j strains
      // also order*(order-1) matrix but order-1 columns
      pt[0] = red_knots[n % (order - 1)];
      pt[1] = full_knots[n / (order - 1)];
    } else if constexpr (icomp == 3 || icomp == 4) {  // g22 or g23
      // 2-dir uses reduced knots for 2j strains
      // also order*(order-1) matrix but order columns
      pt[0] = full_knots[n % order];
      pt[1] = red_knots[n / order];
    } else if constexpr (icomp == 1) {  // g12
      // 1-dir and 2-dir use reduced knots here
      pt[0] = red_knots[n % (order - 1)];
      pt[1] = red_knots[n / (order - 1)];
    }
  }

  template <int icomp>
  __HOST_DEVICE__ static void getTyingInterp(const T pt[], T N[]) {
    // get 1d knot vectors
    // T red_knots[(order-1)], full_knots[order];
    // getTyingKnots(red_knots, full_knots);

    T na[order], nb[order];
    lagrangeLobatto1D_tying<order>(pt[0], na);
    lagrangeLobatto1D_tying<order>(pt[1], nb);

    T nar[order], nbr[order];
    lagrangeLobatto1D_tying<order - 1>(pt[0], nar);
    lagrangeLobatto1D_tying<order - 1>(pt[1], nbr);

    if constexpr (icomp == 0) {
      // g11
      for (int j = 0; j < order; j++) {
        for (int i = 0; i < order - 1; i++, N++) {
          N[0] = nar[i] * nb[j];
        }
      }
    } else if constexpr (icomp == 1) {
      // g12
      for (int j = 0; j < order - 1; j++) {
        for (int i = 0; i < order - 1; i++, N++) {
          N[0] = nar[i] * nbr[j];
        }
      }
    } else if constexpr (icomp == 2) {
      // g13
      for (int j = 0; j < order; j++) {
        for (int i = 0; i < order - 1; i++, N++) {
          N[0] = nar[i] * nb[j];
        }
      }
    } else if constexpr (icomp == 3) {
      // g22
      for (int j = 0; j < order - 1; j++) {
        for (int i = 0; i < order; i++, N++) {
          N[0] = na[i] * nbr[j];
        }
      }
    } else if constexpr (icomp == 4) {
      // g23
      for (int j = 0; j < order - 1; j++) {
        for (int i = 0; i < order; i++, N++) {
          N[0] = na[i] * nbr[j];
        }
      }
    }
  }  // end of getTyingInterp

};  // end of class ShellQuadBasis

// Basis related utils
template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeNodeNormals(const T Xpts[], T fn[], T Xdn[]) {
  // the nodal normal vectors are used for director methods
  // fn is 3*num_nodes each node normals
  // Xdn is list of shell frames
  for (int inode = 0; inode < Basis::num_nodes; inode++) {
    T pt[2];
    Basis::getNodePoint(inode, pt);

    // compute the computational coord gradients of Xpts for xi, eta
    T dXdxi[3], dXdeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, Xpts, dXdxi, dXdeta);

    // compute the normal vector fn at each node
    A2D::VecCrossCore<T>(dXdxi, dXdeta, &fn[3 * inode]);
    T norm = sqrt(A2D::VecDotCore<T, 3>(&fn[3 * inode], &fn[3 * inode]));

    // get Xdn frames
    Basis::assembleFrame(dXdxi, dXdeta, &fn[3 * inode], &Xdn[9 * inode]);

    // removed these parts to have less temp storage on GPU
    // ----------------------------------------------------
    // // be careful this doesn't cause warp divergence on GPU (prob not but
    // just be careful) if (fnorm) {
    //     fnorm[inode] = norm;
    // }
    // if (Xdn) {
    //     Basis::template assembleFrame(dXdxi, dXdeta, &fn[3*inode],
    //     &Xdn[9*inode]);
    // }
  }
}

template <typename T, int vars_per_node, class Basis>
__HOST_DEVICE__ T ShellComputeDispGrad(const T pt[], const T xpts[],
                                       const T vars[], const T fn[],
                                       const T d[], const T Xxi[],
                                       const T Xeta[], const T n0[],
                                       const T Tmat[], T XdinvT[], T u0x[],
                                       T u1x[]) {
  // 75 floats used in this method
  // up to 217 floats here (but then goes back down to 169 floats after out of
  // scope)

  // shell normal comp gradients
  T nxi[3], neta[3];
  Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

  // assemble frames
  A2D::Vec<T, 3> zero;
  T Xd[9], Xdz[9];
  Basis::assembleFrame(Xxi, Xeta, n0, Xd);
  Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

  // invert the Xd transformation
  T Xdinv[9], tmp[9];
  A2D::MatInvCore<T, 3>(Xd, Xdinv);
  T detXd = A2D::MatDetCore<T, 3>(Xd);

  // compute XdinvT = Xdinv*T
  // T XdinvT[9]; // stored as output
  A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);

  // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
  T XdinvzT[9];
  A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
  A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);

  // get the disp gradients
  T d0[3], d0xi[3], d0eta[3];
  Basis::template interpFields<3, 3>(pt, d, d0);
  Basis::template interpFieldsGrad<3, 3>(pt, d, d0xi, d0eta);

  T u0xi[3], u0eta[3];
  Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

  // compute untransformed u0_(xi,eta) derivative
  Basis::assembleFrame(u0xi, u0eta, d0, u0x);

  // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
  Basis::assembleFrame(d0xi, d0eta, zero.get_data(), u1x);
  A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
  A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
  A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1x);

  // compute u0x = T^{T}*u0d*Xdinv*T
  A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
  A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0x);

  return detXd;
}

template <typename T, class Basis>
__HOST_DEVICE__ static void interpTyingStrain(const T pt[], const T ety[],
                                              T gty[]) {
  // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
  // in order {g11-n1, g11-n2, ..., g11-nN, g12-n1, g12-n2,...}
  // interp the final tying strain {g11, g12, g13, g22, g23} at this point with
  // g33 = 0 also
  int32_t offset;

  // get g11 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(0);
  constexpr int num_tying_g11 = Basis::num_tying_points(0);
  T N_g11[num_tying_g11];  // TODO : can we store less floats here?
  Basis::template getTyingInterp<0>(pt, N_g11);
  gty[0] = A2D::VecDotCore<T, num_tying_g11>(N_g11, &ety[offset]);

  // get g12 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(1);
  constexpr int num_tying_g12 = Basis::num_tying_points(1);
  T N_g12[num_tying_g12];
  Basis::template getTyingInterp<1>(pt, N_g12);
  gty[1] = A2D::VecDotCore<T, num_tying_g12>(N_g12, &ety[offset]);

  // get g13 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(2);
  constexpr int num_tying_g13 = Basis::num_tying_points(2);
  T N_g13[num_tying_g13];
  Basis::template getTyingInterp<2>(pt, N_g13);
  gty[2] = A2D::VecDotCore<T, num_tying_g13>(N_g13, &ety[offset]);

  // get g22 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(3);
  constexpr int num_tying_g22 = Basis::num_tying_points(3);
  T N_g22[num_tying_g22];
  Basis::template getTyingInterp<3>(pt, N_g22);
  gty[3] = A2D::VecDotCore<T, num_tying_g22>(N_g22, &ety[offset]);

  // get g23 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(4);
  constexpr int num_tying_g23 = Basis::num_tying_points(4);
  T N_g23[num_tying_g23];
  Basis::template getTyingInterp<4>(pt, N_g23);
  gty[4] = A2D::VecDotCore<T, num_tying_g23>(N_g23, &ety[offset]);

  // get g33 tying strain
  // --------------------
  gty[5] = 0.0;
}

template <typename T, class Basis>
__HOST_DEVICE__ static void interpTyingStrainTranspose(const T pt[],
                                                       const T gty_bar[],
                                                       T ety_bar[]) {
  // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
  // in order {g11-n1, g11-n2, ..., g11-nN, g12-n1, g12-n2,...}
  // interp the final tying strain {g11, g12, g13, g22, g23} at this point with
  // g33 = 0 also
  int32_t offset;

  // TODO : is this really the most efficient way to do this? Profile on GPU

  // get g11 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(0);
  constexpr int num_tying_g11 = Basis::num_tying_points(0);
  T N_g11[num_tying_g11];  // TODO : can we store less floats here?
  Basis::template getTyingInterp<0>(pt, N_g11);
  A2D::VecAddCore<T, num_tying_g11>(gty_bar[0], N_g11, &ety_bar[offset]);

  // get g12 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(1);
  constexpr int num_tying_g12 = Basis::num_tying_points(1);
  T N_g12[num_tying_g12];
  Basis::template getTyingInterp<1>(pt, N_g12);
  A2D::VecAddCore<T, num_tying_g11>(gty_bar[1], N_g12, &ety_bar[offset]);

  // get g13 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(2);
  constexpr int num_tying_g13 = Basis::num_tying_points(2);
  T N_g13[num_tying_g13];
  Basis::template getTyingInterp<2>(pt, N_g13);
  A2D::VecAddCore<T, num_tying_g11>(gty_bar[2], N_g13, &ety_bar[offset]);

  // get g22 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(3);
  constexpr int num_tying_g22 = Basis::num_tying_points(3);
  T N_g22[num_tying_g22];
  Basis::template getTyingInterp<3>(pt, N_g22);
  A2D::VecAddCore<T, num_tying_g11>(gty_bar[3], N_g22, &ety_bar[offset]);

  // get g23 tying strain
  // ------------------------------------
  offset = Basis::tying_point_offsets(4);
  constexpr int num_tying_g23 = Basis::num_tying_points(4);
  T N_g23[num_tying_g23];
  Basis::template getTyingInterp<4>(pt, N_g23);
  A2D::VecAddCore<T, num_tying_g11>(gty_bar[4], N_g23, &ety_bar[offset]);

  // get g33 tying strain
  // --------------------
  // zero so do nothing
}

template <typename T, int vars_per_node, class Basis>
__HOST_DEVICE__ T ShellComputeDispGradSens(
    const T pt[], const T xpts[], const T vars[], const T fn[], const T d[],
    const T Xxi[], const T Xeta[], const T n0[], const T Tmat[], T XdinvT[],
    const T u0x_bar[], const T u1x_bar[], T res[], T d_bar[]) {
  // 75 floats used in this method
  // up to 217 floats here (but then goes back down to 169 floats after out of
  // scope)

  // shell normal comp gradients
  T nxi[3], neta[3];
  Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

  // assemble frames
  A2D::Vec<T, 3> zero;
  T Xd[9], Xdz[9];
  Basis::assembleFrame(Xxi, Xeta, n0, Xd);
  Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

  // invert the Xd transformation
  T Xdinv[9], tmp[9];
  A2D::MatInvCore<T, 3>(Xd, Xdinv);
  T detXd = A2D::MatDetCore<T, 3>(Xd);

  // compute XdinvT = Xdinv*T
  // T XdinvT[9]; // stored as output
  A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);

  // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
  T XdinvzT[9];
  A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
  A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);

  // scope for u0d_barT
  {
    T u0d_barT[9];

    // reverse u0x_bar => u0d_bar
    // u0d_bar = T^t * Xdinv^-t * u0x_bar * T
    // compute the transpose version of u0d_bar to make it easier to extract
    // values though u0d_bar^t = T^t * u0x_bar^t * XdinvT
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::TRANSPOSE>(
        Tmat, u0x_bar, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, XdinvT, u0d_barT);

    // also u1x_bar => u0d_bar contribution
    // u0d_bar^t += T^t * u1x_bar^t * XdinvzT
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::TRANSPOSE>(
        Tmat, u1x_bar, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, XdinvzT, u0d_barT);

    // transfer back to u0xi, u0eta, d0 bar
    // each vec is columns of u0d_bar or rows of u0d_barT (more efficient since
    // don't need to call extractFrame now)
    Basis::template interpFieldsTranpspose<3, 3>(pt, &u0d_barT[6], d_bar);
    Basis::template interpFieldsGradTranspose<vars_per_node, 3>(
        pt, &u0d_barT[0], &u0d_barT[3], res);
  }  // end of u0d_barT scope

  // scope for u1d_barT
  {
    T u1d_barT[9];

    // reverse u1x_bar => u1d_barT
    // u1d_barT = T^t * u1x_bar^t * XdinvT
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::TRANSPOSE>(
        Tmat, u1x_bar, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, XdinvT, u1d_barT);

    // recall u1d = {d0xi, d0eta, zero} with assemble frame
    // rows of u1d_bar^t equiv to columns of u1d_bar aka d0xi, d0eta
    // transfer back to basis d_bar
    Basis::template interpFieldsGradTranspose<3, 3>(pt, &u1d_barT[0],
                                                    &u1d_barT[3], d_bar);
  }

  return detXd;
}
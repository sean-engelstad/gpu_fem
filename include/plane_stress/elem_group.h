#pragma once

#include "a2dcore.h"
#include "basis.h"
#include "geometry.h"
#include "physics.h"
#include "../base/elem_group.h"

template <typename T, class Geo_, class Basis_, class Phys_>
class PlaneStressElementGroup : public BaseElementGroup<PlaneStressElementGroup<T, Geo_, Basis_, Phys_>,
   T, Geo_, Basis_, Phys_> {
 public:
  using ElemGroup = PlaneStressElementGroup<T, Geo_, Basis_, Phys_>;
  using Base = BaseElementGroup<ElemGroup, T, Geo_, Basis_, Phys_>
  using Geo = Geo_;
  using Basis = Basis_;
  using Phys = Phys_;
  using Quadrature = typename Geo::Quadrature;
  using FADType = typename A2D::ADScalar<T, 1>;
  static constexpr int32_t xpts_per_elem = Base::xpts_per_elem;
  static constexpr int32_t dof_per_elem = Base::dof_per_elem;
  static constexpr int32_t num_quad_pts = Base::num_quad_pts;

  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_residual(
      const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
      const Data physData, T res[dof_per_elem]) {
    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    // for (int i = 0; i < 2; i++) {
    //   printf("pt[%d] = %8.e\n", i, pt[i]);
    // }

    A2D::Mat<T, Geo::spatial_dim, 2> J, Jinv;
    Geo::interpParamGradient(pt, xpts, J.get_data());
    T detJ;
    A2D::MatInv(J, Jinv);
    A2D::MatDet(J, detJ);

    // Compute state gradient in parametric space
    A2D::Mat<T, Phys::vars_per_node, 2> dUdxi, dUdxi_bar;
    Basis::template interpParamGradient<Phys::vars_per_node>(pt, vars,
                                                             dUdxi.get_data());

    // Transform to gradient in real space dudx = dudxi * J^-1
    A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim> dUdx, dUdx_bar;
    A2D::MatMatMult(dUdxi, Jinv, dUdx);

    T scale = detJ * weight;
    // printf("scale = %.8e\n", scale);

    // Compute weak residual (derivative of energy w.r.t state gradient)
    Phys::template computeWeakRes<T>(physData, scale, dUdx, dUdx_bar);

    // for (int i = 0; i < 4; i++) {
    //   printf("dUdx_bar[%d] = %.8e\n", i, dUdx_bar[i]);
    // }

    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(dUdx_bar, Jinv,
                                                               dUdxi_bar);

    // for (int i = 0; i < 4; i++) {
    //   printf("dUdxi_bar[%d] = %.8e\n", i, dUdxi_bar[i]);
    // }

    Basis::template addInterpParamGradientSens<Phys::vars_per_node>(
        pt, dUdxi_bar.get_data(), res);

    // for (int i = 0; i < 12; i++) {
    //   printf("res[%d] = %.8e\n", i, res[i]);
    // }
  }

  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
      const int iquad, const int ideriv, const T xpts[xpts_per_elem],
      const T vars[dof_per_elem], const Data physData, T res[dof_per_elem],
      T matCol[dof_per_elem]) {
    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    // for (int i = 0; i < 2; i++) {
    //   printf("pt[%d] = %8.e\n", i, pt[i]);
    // }

    A2D::Mat<T, Geo::spatial_dim, 2> J, Jinv;
    Geo::interpParamGradient(pt, xpts, J.get_data());
    T detJ;
    A2D::MatInv(J, Jinv);
    A2D::MatDet(J, detJ);

    // compute disp grad from q
    A2D::Mat<T, Phys::vars_per_node, 2> dUdxi;
    Basis::template interpParamGradient<Phys::vars_per_node>(pt, vars,
                                                             dUdxi.get_data());
    A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim> dUdx;
    A2D::MatMatMult(dUdxi, Jinv, dUdx);

    // for (int i = 0; i < 4; i++) {
    //   printf("dUdxi[%d] = %.8e\n", i, dUdxi[i]);
    // }
    // for (int i = 0; i < 4; i++) {
    //   printf("dUdx[%d] = %.8e\n", i, dUdx[i]);
    // }

    // Compute disp grad pert from qdot (unit vector)
    const int pertNode = ideriv % Basis::num_nodes;
    const int pertVar = ideriv / Basis::num_nodes;
    A2D::Mat<T, Phys::vars_per_node, 2> dUdxi_dot;
    // Put the gradient of the pertNode'th basis function in the pertVar'th row
    // of dUdxi_dot
    Basis::getBasisGrad(pertNode, pt, &dUdxi_dot.get_data()[2 * pertVar]);

    // Transform dUdxi_dot to physical coordinates
    A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim> dUdx_dot;
    A2D::MatMatMult(dUdxi_dot, Jinv, dUdx_dot);

    A2D::Mat<FADType, Phys::vars_per_node, Geo::spatial_dim> dUdx_fwd;
    T scale = detJ * weight;
    // Set forward seed into dUdx
    for (int ii = 0; ii < dUdx.ncomp; ii++) {
      dUdx_fwd[ii].value = dUdx[ii];
      dUdx_fwd[ii].deriv[0] = dUdx_dot[ii];
    }
    // Compute weak residual (derivative of energy w.r.t state gradient)
    A2D::Mat<FADType, Phys::vars_per_node, Geo::spatial_dim> dUdx_bar;
    Phys::template computeWeakRes<FADType>(physData, scale, dUdx_fwd, dUdx_bar);
    // Now dUdx_bar contains the forward seed of the weak res
    A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim> weak_res_dot, weak_res;
    for (int ii = 0; ii < dUdx.ncomp; ii++) {
      weak_res[ii] = dUdx_bar[ii].value;
      weak_res_dot[ii] = dUdx_bar[ii].deriv[0];
    }

    A2D::Mat<T, Phys::vars_per_node, 2> dUdxi_bar, dUdxi_bar_dot;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(weak_res, Jinv,
                                                               dUdxi_bar);
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(
        weak_res_dot, Jinv, dUdxi_bar_dot);

    Basis::template addInterpParamGradientSens<Phys::vars_per_node>(
        pt, dUdxi_bar.get_data(), res);
    Basis::template addInterpParamGradientSens<Phys::vars_per_node>(
        pt, dUdxi_bar_dot.get_data(), matCol);
  }

  // __device__ void element_jacobian(const T* xpts, const T* vars, T* jac) {}

  // template <int32_t elems_per_block = 1>
  template <class Data>
  static void add_residual(int32_t num_elements, int32_t *geo_conn,
                           int32_t *vars_conn, T *xpts, T *vars, Data *physData,
                           T *residual) {

#ifdef USE_GPU
    constexpr int elems_per_block = 32;
    dim3 block(elems_per_block, num_quad_pts);
    dim3 one_element_block(1, num_quad_pts);

    int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
    dim3 grid(nblocks);

    // constexpr int elems_per_block = 1;

    // add_residual_gpu<T, ElemGroup, elems_per_block> <<<grid,
    // block>>>(num_elements, geo_conn, vars_conn, X, soln, residual);
    add_residual_gpu<T, ElemGroup, Data, elems_per_block>
        <<<1, one_element_block>>>(num_elements, geo_conn, vars_conn, xpts,
                                   vars, physData, residual);

    gpuErrchk(cudaDeviceSynchronize());

#else  // CPU data
    // maybe a way to call add_residual_kernel as same method on CPU
    // with elems_per_block = 1
    add_residual_cpu<Data>(num_elements, geo_conn, vars_conn, xpts, vars,
                           physData, residual);
#endif
  }

  template <class Data>
  static void add_jacobian(int32_t num_vars_nodes, int32_t num_elements,
                           int32_t *geo_conn, int32_t *vars_conn, T *xpts,
                           T *vars, Data *physData, T *residual, T *mat) {

#ifdef USE_GPU
    const int elems_per_block = 8;
    dim3 block(elems_per_block, dof_per_elem, num_quad_pts);

    dim3 one_element_block(1, dof_per_elem, num_quad_pts);

    int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
    dim3 grid(nblocks);

    // add_residual_gpu<T, ElemGroup, elems_per_block> <<<grid,
    // block>>>(num_elements, geo_conn, vars_conn, X, soln, residual);
    add_jacobian_gpu<T, ElemGroup, Data, 1><<<1, one_element_block>>>(
        num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData,
        residual, mat);

    gpuErrchk(cudaDeviceSynchronize());

#else  // CPU data
    // maybe a way to call add_residual_kernel as same method on CPU
    // with elems_per_block = 1
    add_jacobian_cpu<Data>(num_vars_nodes, num_elements, geo_conn, vars_conn,
                           xpts, vars, physData, residual, mat);
#endif
  }

};  // end of ElementGroup class

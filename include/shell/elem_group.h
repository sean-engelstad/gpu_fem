#pragma once

#include "a2dcore.h"

#include "basis.h"
#include "data.h"
#include "director.h"
#include "physics.h"
#include "../base/elem_group.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroup : public BaseElementGroup<ShellElementGroup<T, Director_, Basis_, Phys_>,
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


  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_residual(
    const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
    const Data physData, T res[dof_per_elem]) {
    
    // keep in mind max of ~256 floats on single thread

    // compute node normals
    T fn[3*num_nodes], etn[num_nodes];
    // scope block for Xdn
    {
      T Xdn[9*num_nodes];
      // remove temp storage of Xdn etc. (36 floats) to backprop (recompute there)
      ShellComputeNodeNormals<T, Basis>(xpts, fn, Xdn); 

      // compute drill strains
      // removed XdinvTn, Tn, u0xn, Ctn temp storage to residual to not go over thread memory limit (removes 144 floats)
      ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
        physData, Xdn, vars, etn
      );
    }
    

    // compute director rates
    T d[3*num_nodes];
    Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

    // compute tying strain
    T ety[Basis::num_all_tying_points];
    Phys::template computeTyingStrain<Basis>(xpts, fn, vars, d, ety);

    // end of node points section

    // start of quad pt section (interpolate to quad pt level)
    // -------------------------------------------------------

    // get quadrature point
    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    // passive variables in pre-physics
    A2D::Mat<T,3,3> Tmat;

    // inputs to physics (pre-physics section)
    // -----------------------------
    A2D::ADObj<A2D::Mat<T,3,3>> u0x, u1x;
    A2D::ADObj<A2D::SymMat<T,3>> e0ty;
    A2D::ADObj<A2D::Vec<T,1>> et;
    T detXd;

    // scope block for some variables needed for interpolation only (reduces register pressure)
    // TODO : see if this is optimal set of variables here
    // -----------------------------------------------
    { // pre-physics scope block level 1
      T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];

      // interpolation of coordinates
      // Basis::template interpFields<3, 3>(pt, xpts, X.get_data()); // X not needed directly?
      Basis::template interpFields<3, 3>(pt, fn, n0);
      Basis::template interpFields<1, 1>(pt, etn, et.value().get_data());
      Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
      Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

      // shell transform (natural or ref axis)
      ShellComputeTransform<T, Data>(physData.refAxis, Xxi, Xeta, n0, Tmat.get_data());      

      { // pre-physics scope block level 2
        T gty[6];
        interpTyingStrain<T,Basis>(pt, ety, gty);

        A2D::Mat<T,3,3> XdinvT;
        // compute computational disp gradients
        detXd = ShellComputeDispGrad<T, vars_per_node, Basis>(
          pt, xpts, vars, fn, d, Xxi, Xeta, n0, Tmat.get_data(), 
          XdinvT.get_data(), u0x.value().get_data(), u1x.value().get_data()
        );

        // now transform the strain
        A2D::SymMatRotateFrame<T,3>(XdinvT, gty, e0ty.value());
        // now XdinvT goes out of scope
      } // end of pre-physics scope level 1

      // outputs are essentially u0x, u1x, e0ty, et, detXd
    } // end of pre-physics scope level 2

    // physics : disp gradients in physical space to strain energy
    // and then get sensitivities of disp gradients from energy
    T scale = detXd * weight;
    Phys::template computeWeakRes<T>(physData, scale, u0x, u1x, e0ty, et);

    // TODO : back propagate from u0x, u1x, e0ty, et
    // do we need XdinvzT?

  } // end of method add_element_quadpt_residual

  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
      const int iquad, const int ideriv, const T xpts[xpts_per_elem],
      const T vars[dof_per_elem], const Data physData, T res[dof_per_elem],
      T matCol[dof_per_elem]) {
        // TODO
  } // add_element_quadpt_jacobian_col

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
    Base::template add_residual_cpu<Data>(num_elements, geo_conn, vars_conn, xpts, vars,
                           physData, residual);
#endif
  } // end of add_residual method
};
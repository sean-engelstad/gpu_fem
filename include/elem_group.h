#pragma once

#include "basis.h"
#include "geometry.h"
#include "physics.h"

#include "a2dcore.h"

// include the kernels if on the GPU
#ifdef USE_GPU
#include "elem_group.cuh"
#endif

template <typename T, class Geo_, class Basis_, class Phys_>
class ElementGroup {
 public:
  using Geo = Geo_;
  using Basis = Basis_;
  using Phys = Phys_;
  using Quadrature = typename Geo::Quadrature;

  static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
  static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;

  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_residual(
    const int iquad,
    const T xpts[xpts_per_elem], 
    const T vars[dof_per_elem], 
    const Data physData,
    T res[dof_per_elem]
  ) {
    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    A2D::Mat<T,Geo::spatial_dim,2> J, Jinv;
    Geo::interpParamGradient(pt, xpts, J.get_data());
    T detJ;
    A2D::MatInv(J, Jinv);
    A2D::MatDet(J, detJ);

    // Compute state gradient in parametric space
    A2D::Mat<T, Phys::vars_per_node, 2> dUdxi, dUdxi_bar;
    Basis::template interpParamGradient<Phys::vars_per_node>(pt, vars, dUdxi.get_data());

    // Transform to gradient in real space dudx = dudxi * J^-1
    A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim> dUdx, dUdx_bar;
    A2D::MatMatMult(dUdxi, Jinv, dUdx);

    T scale = detJ * weight;

    // Compute weak residual (derivative of energy w.r.t state gradient)
    Phys::computeWeakRes(physData, scale, dUdx, dUdx_bar);

    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(dUdx_bar, Jinv, dUdxi_bar);

    Basis::template addInterpParamGradientSens<Phys::vars_per_node>(pt, dUdxi_bar.get_data(), res);

  }

  // __device__ void element_jacobian(const T* xpts, const T* vars, T* jac) {}

  // template <int32_t elems_per_block = 1>
  template <class Data>
  static void add_residual(int32_t num_elements, int32_t *geo_conn, int32_t *vars_conn, T *xpts, T *vars, Data *physData, T *residual) {
    using ElemGroup = ElementGroup<T, Geo, Basis, Phys>;

    #ifdef USE_GPU
      const int elems_per_block = 32;
      dim3 block(elems_per_block,3);

      int nblocks = (num_elements + elems_per_block - 1)/ elems_per_block; 
      dim3 grid(nblocks);

      // add_residual_gpu<T, ElemGroup, elems_per_block> <<<grid, block>>>(num_elements, geo_conn, vars_conn, X, soln, residual);
      add_residual_gpu<T, ElemGroup, Data, 1> <<<1, 1>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, residual);

      gpuErrchk(cudaDeviceSynchronize());

    #else // CPU data
      // maybe a way to call add_residual_kernel as same method on CPU
      // with elems_per_block = 1
      add_residual_cpu<Data>(num_elements, geo_conn, vars_conn, xpts, vars, physData, residual);
    #endif
  }

  template <class Data>
  static void add_residual_cpu(int32_t num_elements, int32_t *geo_conn, int32_t *vars_conn, T *xpts, T *vars, Data *physData, T *residual) {
        
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    for (int ielem = 0; ielem < num_elements; ielem++) {

      T elem_xpts[nxpts_per_elem];
      T elem_vars[vars_per_elem];
      T elem_res[vars_per_elem];
      Data elem_physData = physData[ielem];

      // get values for this element
      const int32_t *geo_nodes = &geo_conn[ielem*Geo::num_nodes];
      for (int inode = 0; inode < Geo::num_nodes; inode++) {
        int32_t global_inode = geo_nodes[inode];
        for (int idim = 0 ; idim < Geo::spatial_dim; idim++) {
          elem_xpts[inode*Geo::spatial_dim + idim] = xpts[global_inode*Geo::spatial_dim + idim];
        }
      }

      // for (int i = 0; i < nxpts_per_elem; i++) {
      //   printf("elem_xpts[%d] = %.8e\n", i, elem_xpts[i]);
      // }
      // return;

      const int32_t *vars_nodes = &vars_conn[ielem*Basis::num_nodes];
      for (int inode = 0; inode < Basis::num_nodes; inode++) {
        int global_inode = vars_nodes[inode];
        for (int idof = 0 ; idof < Phys::vars_per_node; idof++) {
          elem_vars[inode*Phys::vars_per_node + idof] = vars[global_inode*Phys::vars_per_node + idof];
          elem_res[inode*Phys::vars_per_node + idof] = 0.0;
        }
      }

      // done getting all elem variables

      // compute element residual
      for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
        add_element_quadpt_residual<Data>(
          iquad, elem_xpts, elem_vars, elem_physData, elem_res
        );
      }

      // add back into global res on CPU
      for (int inode = 0; inode < Basis::num_nodes; inode++) {
        int global_inode = vars_nodes[inode];
        for (int idof = 0 ; idof < Phys::vars_per_node; idof++) {
          residual[global_inode*Phys::vars_per_node + idof] += elem_res[inode*Phys::vars_per_node + idof];
        }
      }
      
    } // num_elements for loop

  }
}; // end of ElementGroup class


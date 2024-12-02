// base class for ElemGroup
#pragma once

// include the kernels if on the GPU
#ifdef USE_GPU
#include "elem_group.cuh"
#endif

// use CRTP (curiously recurring template pattern here to avoid using
// virtual functions as these aren't good on GPU)
//     namely allows defining a method in base class that calls 
//     derived class methods at compile time)
template <typename Derived, typename T, class Geo_, class Basis_, class Phys_>
class BaseElementGroup {
 public:
  using Geo = Geo_;
  using Basis = Basis_;
  using Phys = Phys_;
  using Quadrature = typename Basis::Quadrature;
  using FADType = typename A2D::ADScalar<T, 1>;

  static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
  static constexpr int32_t dof_per_elem =
      Phys::vars_per_node * Basis::num_nodes;
  static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

  // no virtual methods since not good for GPU
  // need the following methods in each subclass
  // -----------------------------------------

  // add_residual (launches kernels on GPU or calls CPU version)
  // add_element_quadpt_residual
  // add_jacobian (launches kernels on GPU or calls CPU version)
  // add_element_quadpt_jacobian_col

  template <class Data>
  static void add_residual_cpu(int32_t num_elements, int32_t *geo_conn,
                               int32_t *vars_conn, T *xpts, T *vars,
                               Data *physData, T *residual) {
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    for (int ielem = 0; ielem < num_elements; ielem++) {
      T elem_xpts[nxpts_per_elem];
      T elem_vars[vars_per_elem];
      T elem_res[vars_per_elem];
      Data elem_physData = physData[ielem];

      // get values for this element
      const int32_t *geo_nodes = &geo_conn[ielem * Geo::num_nodes];
      for (int inode = 0; inode < Geo::num_nodes; inode++) {
        int32_t global_inode = geo_nodes[inode];
        for (int idim = 0; idim < Geo::spatial_dim; idim++) {
          elem_xpts[inode * Geo::spatial_dim + idim] =
              xpts[global_inode * Geo::spatial_dim + idim];
        }
      }

      // for (int i = 0; i < nxpts_per_elem; i++) {
      //   printf("elem_xpts[%d] = %.8e\n", i, elem_xpts[i]);
      // }
      // return;

      const int32_t *vars_nodes = &vars_conn[ielem * Basis::num_nodes];
      for (int inode = 0; inode < Basis::num_nodes; inode++) {
        int global_inode = vars_nodes[inode];
        for (int idof = 0; idof < Phys::vars_per_node; idof++) {
          elem_vars[inode * Phys::vars_per_node + idof] =
              vars[global_inode * Phys::vars_per_node + idof];
          elem_res[inode * Phys::vars_per_node + idof] = 0.0;
        }
      }

      // done getting all elem variables

      // compute element residual
      for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
        Derived::template add_element_quadpt_residual<Data>(iquad, elem_xpts, elem_vars,
                                          elem_physData, elem_res);
      }

      // add back into global res on CPU
      for (int idof = 0; idof < vars_per_elem; idof++) {
        int local_inode = idof % Basis::num_nodes;
        int local_idim = idof / Basis::num_nodes;
        int iglobal =
            vars_nodes[local_inode] * Phys::vars_per_node + local_idim;
        residual[iglobal] += elem_res[idof];
      }  // end of residual assembly

    }  // num_elements for loop
  }

  template <class Data>
  static void add_jacobian_cpu(int32_t vars_num_nodes, int32_t num_elements,
                               int32_t *geo_conn, int32_t *vars_conn, T *xpts,
                               T *vars, Data *physData, T *residual, T *mat) {
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int num_vars = vars_num_nodes * Phys::vars_per_node;

    for (int ielem = 0; ielem < num_elements; ielem++) {
      T elem_xpts[nxpts_per_elem];
      T elem_vars[vars_per_elem];
      T elem_res[vars_per_elem];
      T elem_mat[vars_per_elem * vars_per_elem];
      Data elem_physData = physData[ielem];

      memset(elem_res, 0.0, vars_per_elem * sizeof(T));
      memset(elem_mat, 0.0, vars_per_elem * vars_per_elem * sizeof(T));

      // get values for this element
      const int32_t *geo_nodes = &geo_conn[ielem * Geo::num_nodes];
      for (int inode = 0; inode < Geo::num_nodes; inode++) {
        int32_t global_inode = geo_nodes[inode];
        for (int idim = 0; idim < Geo::spatial_dim; idim++) {
          elem_xpts[inode * Geo::spatial_dim + idim] =
              xpts[global_inode * Geo::spatial_dim + idim];
        }
      }

      // for (int i = 0; i < nxpts_per_elem; i++) {
      //   printf("elem_xpts[%d] = %.8e\n", i, elem_xpts[i]);
      // }
      // return;

      const int32_t *vars_nodes = &vars_conn[ielem * Basis::num_nodes];
      for (int inode = 0; inode < Basis::num_nodes; inode++) {
        int global_inode = vars_nodes[inode];
        for (int idof = 0; idof < Phys::vars_per_node; idof++) {
          elem_vars[inode * Phys::vars_per_node + idof] =
              vars[global_inode * Phys::vars_per_node + idof];
        }
      }

      // done getting all elem variables

      // compute element residual
      for (int ideriv = 0; ideriv < vars_per_elem; ideriv++) {
        A2D::Vec<T, vars_per_elem> matCol;  // initialized to zero
        for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
          Derived::template add_element_quadpt_jacobian_col<Data>(iquad, ideriv, elem_xpts,
                                                elem_vars, elem_physData,
                                                elem_res, matCol.get_data());

          // add into elem_mat first
          for (int jdof = 0; jdof < vars_per_elem; jdof++) {
            elem_mat[vars_per_elem * jdof + ideriv] += matCol[jdof];
          }

          // for (int idof = 0; idof < vars_per_elem; idof++) {
          //   printf("matCol[%d] = %.8e\n", idof, matCol[idof]);
          // }
        }
      }
      // we've computed element stiffness matrix
      // now we add into global stiffness matrix

      // assembly into global matrix
      for (int idof = 0; idof < vars_per_elem; idof++) {
        int local_inode = idof % Basis::num_nodes;
        int local_idim = idof / Basis::num_nodes;
        int iglobal =
            vars_nodes[local_inode] * Phys::vars_per_node + local_idim;
        residual[iglobal] +=
            elem_res[idof] /
            vars_per_elem;  // divide because we added into it 12 times

        for (int jdof = 0; jdof < vars_per_elem; jdof++) {
          int local_jnode = jdof % Basis::num_nodes;
          int local_jdim = jdof / Basis::num_nodes;
          int jglobal =
              vars_nodes[local_jnode] * Phys::vars_per_node + local_jdim;
          mat[num_vars * iglobal + jglobal] +=
              elem_mat[vars_per_elem * idof + jdof];
        }
      }  // end of matrix assembly double for loops

    }  // end of num_elements for loop
  }  // end of addJacobian_cpu method

};  // end of ElementGroup class

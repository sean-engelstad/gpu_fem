// base class for ElemGroup
#pragma once

#include "../linalg/vec.h"
#include "a2dcore.h"
#include "utils.h"

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

    template <class Data, template <typename> class Vec>
    static void
    add_residual_cpu(const int32_t num_elements, const Vec<int32_t> &geo_conn,
                     const Vec<int32_t> &vars_conn, const Vec<T> &xpts,
                     const Vec<T> &vars, Vec<Data> &physData, Vec<T> &res) {

        const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
        const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

        const int32_t *_geo_conn = geo_conn.getPtr();
        const int32_t *_vars_conn = vars_conn.getPtr();

        for (int ielem = 0; ielem < num_elements; ielem++) {

            Data elem_physData = physData[ielem];
            A2D::Vec<T, vars_per_elem> a2d_elem_res; // so zeroes it
            T *elem_res = a2d_elem_res.get_data();

            T elem_xpts[nxpts_per_elem];
            const int32_t *geo_elem_conn = &_geo_conn[ielem * Geo::num_nodes];
            xpts.getElementValues(Geo::spatial_dim, Geo::num_nodes,
                                  geo_elem_conn, &elem_xpts[0]);

            T elem_vars[vars_per_elem];
            const int32_t *vars_elem_conn =
                &_vars_conn[ielem * Basis::num_nodes];
            vars.getElementValues(Phys::vars_per_node, Basis::num_nodes,
                                  vars_elem_conn, &elem_vars[0]);

            // compute element residual
            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                Derived::template add_element_quadpt_residual<Data>(
                    true, iquad, elem_xpts, elem_vars, elem_physData, elem_res);
            }

            res.addElementValues(1.0, Phys::vars_per_node, Basis::num_nodes,
                                 vars_elem_conn, elem_res);

        } // num_elements for loop
    }

    template <class Data, template <typename> class Vec>
    static void add_energy_cpu(int32_t num_elements, Vec<int32_t> *geo_conn,
                               Vec<int32_t> *vars_conn, Vec<T> *xpts,
                               Vec<T> *vars, Data *physData, T &Uenergy) {

        const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
        const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

        const int32_t *_geo_conn = geo_conn->getPtr();
        const int32_t *_vars_conn = vars_conn->getPtr();

        Uenergy = 0.0; // set energy initially to zero

        for (int ielem = 0; ielem < num_elements; ielem++) {

            Data elem_physData = physData[ielem];
            A2D::Vec<T, vars_per_elem> elem_res; // so zeroes it

            T elem_xpts[nxpts_per_elem];
            const int32_t *geo_elem_conn = &_geo_conn[ielem * Geo::num_nodes];
            xpts->getElementValues(Geo::spatial_dim, Geo::num_nodes,
                                   geo_elem_conn, elem_xpts);

            T elem_vars[vars_per_elem];
            const int32_t *vars_elem_conn =
                &_vars_conn[ielem * Basis::num_nodes];
            vars->getElementValues(Phys::vars_per_node, Basis::num_nodes,
                                   vars_elem_conn, elem_vars);
            // done getting all elem variables

            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                Derived::template add_element_quadpt_energy<Data>(
                    true, iquad, elem_xpts, elem_vars, elem_physData, Uenergy);
            }

        } // num_elements for loop
    }

    template <class Data, template <typename> class Vec, class Mat>
    static void add_jacobian_cpu(int32_t vars_num_nodes, int32_t num_elements,
                                 Vec<int32_t> &geo_conn,
                                 Vec<int32_t> &vars_conn, Vec<T> &xpts,
                                 Vec<T> &vars, Vec<Data> &physData, Vec<T> &res,
                                 Mat &mat) {

        const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
        const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
        const int num_vars = vars_num_nodes * Phys::vars_per_node;

        const int32_t *_geo_conn = geo_conn.getPtr();
        const int32_t *_vars_conn = vars_conn.getPtr();

        for (int ielem = 0; ielem < num_elements; ielem++) {

            Data elem_physData = physData[ielem];
            A2D::Vec<T, vars_per_elem> a2d_elem_res; // so zeroes it
            A2D::Mat<T, vars_per_elem, vars_per_elem>
                a2d_elem_mat; // so zeroes it
            T *elem_res = a2d_elem_res.get_data();
            T *elem_mat = a2d_elem_mat.get_data();

            T elem_xpts[nxpts_per_elem];
            const int32_t *geo_elem_conn = &_geo_conn[ielem * Geo::num_nodes];
            xpts.getElementValues(Geo::spatial_dim, Geo::num_nodes,
                                  geo_elem_conn, elem_xpts);

            T elem_vars[vars_per_elem];
            const int32_t *vars_elem_conn =
                &_vars_conn[ielem * Basis::num_nodes];
            vars.getElementValues(Phys::vars_per_node, Basis::num_nodes,
                                  vars_elem_conn, elem_vars);
            // done getting all elem variables

            // compute element residual
            for (int ideriv = 0; ideriv < vars_per_elem; ideriv++) {
                A2D::Vec<T, vars_per_elem> matCol; // initialized to zero
                for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                    Derived::template add_element_quadpt_jacobian_col<Data>(
                        true, iquad, ideriv, elem_xpts, elem_vars,
                        elem_physData, elem_res,
                        &elem_mat[dof_per_elem * ideriv]);
                }
            }

            // assemble elem_mat values into global residual and matrix
            res.addElementValues(1.0 / vars_per_elem, Phys::vars_per_node,
                                 Basis::num_nodes, vars_elem_conn, elem_res);
            mat.addElementMatrixValues(1.0, ielem, Phys::vars_per_node,
                                       Basis::num_nodes, vars_elem_conn,
                                       elem_mat);

        } // end of num_elements for loop
    }     // end of addJacobian_cpu method

}; // end of ElementGroup class

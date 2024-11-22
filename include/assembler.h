#pragma once

#include <cstring>

#include "elem_group.h"

template <typename T, typename ElemGroup>
class ElementAssembler {
 public:
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;
  using Data = typename Phys::Data;
  static constexpr int32_t geo_nodes_per_elem = Geo::num_nodes;
  static constexpr int32_t vars_nodes_per_elem = Basis::num_nodes;
  static constexpr int32_t spatial_dim = Geo::spatial_dim;
  static constexpr int32_t vars_per_node = Phys::vars_per_node;

  // dummy constructor for random points (another one will be made for actual
  // connectivity)
  ElementAssembler(int32_t num_geo_nodes_, int32_t num_vars_nodes_,
                   int32_t num_elements_, int32_t *geo_conn, int32_t *vars_conn,
                   T *xpts, Data *physData)
      : num_geo_nodes(num_geo_nodes_),
        num_vars_nodes(num_vars_nodes_),
        num_elements(num_elements_) {
    // randomly generate the connectivity for the mesh
    int N = geo_nodes_per_elem * num_elements;
    h_geo_conn = new int32_t[N];
    for (int i = 0; i < N; i++) {  // deep copy connectivity
      h_geo_conn[i] = geo_conn[i];
    }

    // randomly generate the connectivity for the variables / basis
    int N2 = vars_nodes_per_elem * num_elements;
    h_vars_conn = new int32_t[N2];
    for (int i = 0; i < N2; i++) {  // deep copy connectivity
      h_vars_conn[i] = vars_conn[i];
    }

    // initialize and allocate data on the device
    int32_t num_xpts = get_num_xpts();
    h_xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
      h_xpts[ixpt] = xpts[ixpt];
    }

    // set some host data to zero
    int32_t num_vars = get_num_vars();
    h_vars = new T[num_vars];
    memset(h_vars, 0.0, num_vars * sizeof(T));

    h_residual = new T[num_vars];
    memset(h_residual, 0.0, num_vars * sizeof(T));

    h_physData = physData;

#ifdef USE_GPU

    printf("starting constructor...\n");

    cudaMalloc((void **)&d_geo_conn, N * sizeof(int32_t));
    cudaMemcpy(d_geo_conn, h_geo_conn, N * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_vars_conn, N2 * sizeof(int32_t));
    cudaMemcpy(d_vars_conn, h_vars_conn, N2 * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_xpts, num_xpts * sizeof(T));
    cudaMemcpy(d_xpts, h_xpts, num_xpts * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_vars, num_vars * sizeof(T));
    cudaMemset(d_vars, 0.0, num_vars * sizeof(T));

    cudaMalloc((void **)&d_residual, num_vars * sizeof(T));
    cudaMemset(d_residual, 0.0, num_vars * sizeof(T));

    cudaMalloc((void **)&d_physData, num_elements * sizeof(Data));
    cudaMemcpy(d_physData, h_physData, num_elements * sizeof(Data),
               cudaMemcpyHostToDevice);

    printf("finished constructor\n");
#endif  // USE_GPU
  };

  int get_num_xpts() { return num_geo_nodes * spatial_dim; }
  int get_num_vars() { return num_vars_nodes * vars_per_node; }

  void set_variables(T *vars) {
    // vars is either device array on GPU or a host array if not USE_GPU

    int32_t num_vars = vars_per_node * num_vars_nodes;
#ifdef USE_GPU
    cudaMemcpy(d_vars, vars, num_vars * sizeof(T), cudaMemcpyDeviceToDevice);
#else
    memcpy(h_vars, vars, num_vars * sizeof(T));
#endif
  }

  //  template <class ExecParameters>
  void add_residual(T *res) {
// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
    ElemGroup::template add_residual<Data>(
        num_elements, d_geo_conn, d_vars_conn, d_xpts, d_vars, d_physData, res);
#else   // USE_GPU
    ElemGroup::template add_residual<Data>(
        num_elements, h_geo_conn, h_vars_conn, h_xpts, h_vars, h_physData, res);
#endif  // USE_GPU
  };

  //  template <class ExecParameters>
  void add_jacobian(T *res, T *mat) {
// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
    ElemGroup::template add_jacobian<Data>(num_vars_nodes, num_elements,
                                           d_geo_conn, d_vars_conn, d_xpts,
                                           d_vars, d_physData, res, mat);
#else   // USE_GPU
    ElemGroup::template add_jacobian<Data>(num_vars_nodes, num_elements,
                                           h_geo_conn, h_vars_conn, h_xpts,
                                           h_vars, h_physData, res, mat);
#endif  // USE_GPU
  };

 private:
  int32_t num_geo_nodes;
  int32_t num_vars_nodes;
  int32_t num_elements;              // Number of elements of this type
  int32_t *h_geo_conn, *d_geo_conn;  // Node numbers for each element of mesh
  int32_t *h_vars_conn,
      *d_vars_conn;  // Node numbers for each element of basis points

  // Global solution and node numbers
  Data *h_physData, *d_physData;
  T *h_vars, *d_vars;
  T *h_xpts, *d_xpts;
  T *h_residual, *d_residual;
};
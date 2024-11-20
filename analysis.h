#include <cstdint>

template <uint32_t xdim = 1, uint32_t ydim = 1, uint32_t zdim = 1,
          uint32_t max_registers_per_thread = 255,
          uint32_t elements_per_block = 1>
class ExecParameters {
 public:
};

#ifdef USE_GPU
#define __SHARED__ __shared__
#else
#define __SHARED__
#endif

template <typename T>
class TriangleQuadrautre {
 public:
  // Required static data used by other classes
  static constexpr int32_t num_quad_pts = 3;
};

template <typename T, class Quadrature>
class LinearTriangleGeo {
 public:
  // Required static data used by other classes

  // Required for loading nodal coordinates
  static constexpr int32_t spatial_dim = 2;

  // Required for knowning number of spatial coordinates per node
  static constexpr int32_t num_nodes = 3;

  // Number of quadrature points
  static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

  // Data-size = spatial_dim * number of nodes
  static constexpr int32_t geo_data_size = 5 * num_quad_pts;
};

template <typename T, class Quadrature>
class QuadraticTriangleBasis {
 public:
  // Required for loading solution data
  static constexpr int32_t num_nodes = 6;

  // Parametric dimension
  static constexpr int32_t param_dim = 2;
};

template <typename T, class Quarature>
class PlaneStressPhysics {
 public:
  // Variables at each node (u, v)
  static constexpr int32_t vars_per_node = 2;
};

template <typename T, class Geo, class Basis, class Phys>
class ElementAssembler {
 public:
 private:
  int32_t num_elements;  // Number of elements of this type
  int32_t *nodes;        // Node numbers for each element

  // Global solution and node numbers
  T *soln;
  T *X;
  T *residual;  //
};

template <typename T, class Geo, class Basis, class Phys>
class ElementGroup {
 public:
  template <int32_t elems_per_block = 1>
  void add_residual() {
    __SHARED__ T geo_data[elements_per_block][Geo::geo_data_size];
    __SHARED__ T basis_data[elements_per_block][Geo::geo_data_size];
    __SHARED__ T residual[elements_per_block][Basis::num_dof];

    // Load the data from global memory into shared memory
    load_data();

    block.add_residual
  }

  template <int32_t elems_per_block = 1>
  void add_jacobian() {
    __SHARED__ T geo_data[elements_per_block][Geo::geo_data_size];
    __SHARED__ T basis_data[elements_per_block][Geo::geo_data_size];
    __SHARED__ T jac[elements_per_block][Basis::num_dof * Basis::num_dof];
  }
};

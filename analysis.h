#include <cstdint>
#include <stdlib.h>
#include <stdio.h>

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
class TriangleQuadrature {
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

template <typename T, class Quadrature>
class PlaneStressPhysics {
 public:
  // Variables at each node (u, v)
  static constexpr int32_t vars_per_node = 2;
};

template <typename T, typename ElemGroup>
class ElementAssembler {
 public:
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;
  static constexpr int32_t nodes_per_elem = Basis::num_nodes;
  static constexpr int32_t spatial_dim = Geo::spatial_dim;
  static constexpr int32_t vars_per_node = Phys::vars_per_node;

  // dummy constructor for random points (another one will be made for actual connectivity)
  ElementAssembler(int32_t num_nodes_, int32_t num_elements_) : num_nodes(num_nodes_), num_elements(num_elements_) {
    // randomly initialize data on GPU
    #ifdef USE_GPU

      printf("starting constructor...\n");

      // randomly generate the connectivity
      int N = nodes_per_elem * num_elements;
      conn = new int[N];
      for (int i = 0; i < N; i++) {
        conn[i] = rand() % num_nodes;
      }

      // initialize and allocate data on the device
      int32_t num_xpts = num_nodes * spatial_dim;
      T *h_X = new T[num_xpts];
      for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
        h_X[ixpt] = static_cast<double>(rand()) / RAND_MAX;
      }
      
      cudaMalloc((void**)&X, num_xpts * sizeof(T));
      cudaMemcpy(X, h_X, num_xpts * sizeof(T), cudaMemcpyHostToDevice);

      int32_t num_vars = vars_per_node * num_nodes;
      cudaMalloc((void**)&soln, num_vars * sizeof(T));
      cudaMemset(soln, 0.0, num_vars * sizeof(T));

      cudaMalloc((void**)&soln, num_vars * sizeof(T));
      cudaMemset(soln, 0.0, num_vars * sizeof(T));     

      printf("finished constructor\n");

    #else
    #endif
  };

//  template <class ExecParameters>
 void add_residual(T *res) {
  ElemGroup::add_residual(num_elements, conn, soln, X, residual);
  // copies global residual to that
 };

 private:
  int32_t num_nodes;
  int32_t num_elements;  // Number of elements of this type
  int32_t *conn;        // Node numbers for each element

  // Global solution and node numbers
  T *soln;
  T *X;
  T *residual;
};

template <typename T, class ElemGroup, int32_t elems_per_block = 1>
__global__ static void add_residual_gpu(int32_t num_elements, int32_t *conn, T *soln, T *X, T *residual) {
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;

  __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
  __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];
  __SHARED__ T local_res[elems_per_block][Basis::num_nodes * Phys::vars_per_node];

  // Load the data from global memory into shared memory
  // still need exec params here for block size
  // load_data();

  printf("<<<sick GPU kernel>>>\n");

  // block.element_residual()
}

// template <int32_t elems_per_block = 1>
// static void add_jacobian_kernel() {
//   __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
//   __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];
//   __SHARED__ T jac[elems_per_block][Basis::num_dof * Basis::num_dof];
// }

template <typename T, class Geo_, class Basis_, class Phys_>
class ElementGroup {
 public:
  using Geo = Geo_;
  using Basis = Basis_;
  using Phys = Phys_;

  __device__ void element_residual() {}
  __device__ void element_kernel() {}

  template <int32_t elems_per_block = 1>
  static void add_residual(int32_t num_elements, int32_t *conn, T *soln, T *X, T *residual) {
    #ifdef USE_GPU
      dim3 block(1);
      dim3 grid(1);

      using ElemGroup = ElementGroup<T, Geo, Basis, Phys>;

      add_residual_gpu<T, ElemGroup, elems_per_block> <<<grid, block>>>(num_elements, conn, soln, X, residual);

      cudaDeviceSynchronize();

    #else // CPU data
      // maybe a way to call add_residual_kernel as same method on CPU
      // with elems_per_block = 1
      // add_residual_cpu(num_elements, conn, soln, X, residual);
    #endif
  }
};
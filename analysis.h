#include <cstdint>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include "a2dcore.h"

#ifdef USE_GPU
#include "cuda_error.h"
#endif

template <uint32_t xdim = 1, uint32_t ydim = 1, uint32_t zdim = 1,
          uint32_t max_registers_per_thread = 255,
          uint32_t elements_per_block = 1>
class ExecParameters {
 public:
};

#ifdef USE_GPU
#define __SHARED__ __shared__
#define __HOST_DEVICE__ __host__ __device__
#define __DEVICE__ __device__
#define __GLOBAL__ __global__
#else
#define __SHARED__
#define __HOST_DEVICE__
#define __DEVICE__
#define __GLOBAL__
#endif

template <typename T>
class TriangleQuadrature {
 public:
  // Required static data used by other classes
  static constexpr int32_t num_quad_pts = 3;

  // get one of the three triangle quad points
  __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
    switch (ind) {
      case 0:
        pt[0] = 0.5;
        pt[1] = 0.5;
      case 1:
        pt[0] = 0.0;
        pt[1] = 0.5;
      case 2:
        pt[0] = 0.5;
        pt[1] = 0.0;
    }
    return 1.0/3.0;
  };
};

template <typename T, class Quadrature_>
class LinearTriangleGeo {
 public:
  // Required static data used by other classes
  using Quadrature = Quadrature_;

  // Required for loading nodal coordinates
  static constexpr int32_t spatial_dim = 2;

  // Required for knowning number of spatial coordinates per node
  static constexpr int32_t num_nodes = 3;

  // Number of quadrature points
  static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

  // Data-size = spatial_dim * number of nodes
  static constexpr int32_t geo_data_size = 5 * num_quad_pts;

  // LINEAR interpolation gradient
  // static constexpr 

  // jacobian and det
  __HOST_DEVICE__ static T interpParamGradient(const T* pt, const T* xpts, T* dXdxi) {
    // pt unused here
    // interpMat static for LINEAR triangle element
    constexpr T dNdxi[2 * num_nodes] = {-1, -1, 1, 0, 0, 1};

    A2D::MatMatMultCore<T, 
        num_nodes, spatial_dim,
        num_nodes, 2, 
        spatial_dim, 2, 
        A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL, false>(xpts, dNdxi, dXdxi);
    
    // return the determinant of dX/dxi
    return dXdxi[0] *dXdxi[3] - dXdxi[1] * dXdxi[2];
  }
};

template <typename T, class Quadrature_>
class QuadraticTriangleBasis {
 public:
  using Quadrature = Quadrature_;

  // Required for loading solution data
  static constexpr int32_t num_nodes = 6;

  // Parametric dimension
  static constexpr int32_t param_dim = 2;

  __HOST_DEVICE__ static void getBasisGrad(const T* xi, T* dNdxi) {
    // compute the basis function gradients at each basis node
    
    // basis fcn 0 : N0 = 2x^2 + 4xy - 3x + 2y^2 - 3y + 1
    dNdxi[0] = 4 * xi[0] + 4 * xi[1] - 3;
    dNdxi[1] = 4 * xi[0] + 4 * xi[1] - 3;
    // basis fcn 1 : N1 = x * (2x - 1)
    dNdxi[2] = 4 * xi[0] - 1;
    dNdxi[3] = 0.0;
    // basis fcn 2 : N2 = y * (2y - 1)
    dNdxi[4] = 0.0;
    dNdxi[5] = 4 * xi[1] - 1;
    // basis fcn 3 : N3 = 4xy
    dNdxi[6] = 4 * xi[1];
    dNdxi[7] = 4 * xi[0];
    // basis fcn 4 : N4 = 4y * (-x - y + 1)
    dNdxi[8]  = -4.0 * xi[1];
    dNdxi[9] = -4.0 * xi[0] - 8.0 * xi[1] + 4.0;
    // basis fcn 5 : N5 = 4x * (-x - y + 1)
    dNdxi[10] = -8.0 * xi[0] - 4.0 * xi[1] + 4.0;
    dNdxi[11] = -4.0 * xi[0];
  }

  // don't use explicit N
  // compute U, dU/dxi
  template <int vars_per_node>
  __HOST_DEVICE__ static void interpParamGradient(const T* xi, const T* Un, T* dUdxi) {

    T dNdxi[num_nodes * 2];
    getBasisGrad(xi, dNdxi);

    A2D::MatMatMultCore<T, 
        num_nodes, vars_per_node,
        num_nodes, 2, 
        vars_per_node, 2, 
        A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL, false>(Un, dNdxi, dUdxi);

  }

  template <int vars_per_node>
  __HOST_DEVICE__ static void addInterpParamGradientSens(const T* xi, const T* dUdxi_bar, T* res) {
    T dNdxi[num_nodes * 2];
    getBasisGrad(xi, dNdxi);

    // res += dN/dxi * dUdxi_bar^T
    A2D::MatMatMultCore<T, 
        num_nodes, 2, 
        vars_per_node, 2,
        num_nodes, vars_per_node, 
        A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE, true>(dNdxi, dUdxi_bar, res);
  }
};

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



  // dummy constructor for random points (another one will be made for actual connectivity)
  ElementAssembler(int32_t num_geo_nodes_, int32_t num_vars_nodes_, int32_t num_elements_,
    int32_t *geo_conn, int32_t *vars_conn, T*xpts,
    Data *physData) : num_geo_nodes(num_geo_nodes_),
    num_vars_nodes(num_vars_nodes_), num_elements(num_elements_) {
    
    // randomly generate the connectivity for the mesh
    int N = geo_nodes_per_elem * num_elements;
    h_geo_conn = new int32_t[N];
    for (int i = 0; i < N; i++) { // deep copy connectivity
      h_geo_conn[i] = geo_conn[i];
    }

    // randomly generate the connectivity for the variables / basis
    int N2 = vars_nodes_per_elem * num_elements;
    h_vars_conn = new int32_t[N2];
    for (int i = 0; i < N2; i++) { // deep copy connectivity
      h_vars_conn[i] = vars_conn[i];
    }

    // initialize and allocate data on the device
    int32_t num_xpts = num_geo_nodes * spatial_dim;
    h_xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
      h_xpts[ixpt] = static_cast<double>(rand()) / RAND_MAX;
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

      cudaMalloc((void**)&d_geo_conn, N * sizeof(int32_t));
      cudaMemcpy(d_geo_conn, h_geo_conn, N * sizeof(int32_t), cudaMemcpyHostToDevice);
      
      cudaMalloc((void**)&d_vars_conn, N2 * sizeof(int32_t));
      cudaMemcpy(d_vars_conn, h_vars_conn, N2 * sizeof(int32_t), cudaMemcpyHostToDevice);
      
      cudaMalloc((void**)&d_xpts, num_xpts * sizeof(T));
      cudaMemcpy(d_xpts, h_xpts, num_xpts * sizeof(T), cudaMemcpyHostToDevice);

      cudaMalloc((void**)&d_vars, num_vars * sizeof(T));
      cudaMemset(d_vars, 0.0, num_vars * sizeof(T));

      cudaMalloc((void**)&d_residual, num_vars * sizeof(T));
      cudaMemset(d_residual, 0.0, num_vars * sizeof(T));     

      cudaMalloc((void**)&d_physData, num_elements * sizeof(Data));
      cudaMemcpy(d_physData, h_physData, num_elements * sizeof(Data), cudaMemcpyHostToDevice);

      printf("finished constructor\n");
    #endif // USE_GPU
  };

  int get_num_vars() {return vars_per_node * num_vars_nodes;}

  void set_variables(T* vars) {
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
    ElemGroup::template add_residual<Data>(num_elements, d_geo_conn, d_vars_conn, d_xpts, d_vars, d_physData, res);
  #else // USE_GPU
    ElemGroup::template add_residual<Data>(num_elements, h_geo_conn, h_vars_conn, h_xpts, h_vars, h_physData, res);
  #endif // USE_GPU

  // add into outer residual
  
 };

 private:
  int32_t num_geo_nodes;
  int32_t num_vars_nodes;
  int32_t num_elements;  // Number of elements of this type
  int32_t *h_geo_conn, *d_geo_conn;        // Node numbers for each element of mesh
  int32_t *h_vars_conn, *d_vars_conn;      // Node numbers for each element of basis points

  // Global solution and node numbers
  Data *h_physData, *d_physData;
  T *h_vars, *d_vars;
  T *h_xpts, *d_xpts;
  T *h_residual, *d_residual;
};

#ifdef USE_GPU

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1>
__GLOBAL__ static void add_residual_gpu(int32_t num_elements, int32_t geo_conn, int32_t vars_conn, T *xpts, T *vars, Data *physData, T *residual) {
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;

  // if you want to precompute some things?
  // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
  // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

  int local_elem = threadIdx.x;
  int global_elem = local_elem + blockDim.x * blockIdx.x; 
  bool active_thread = global_elem < num_elements;
  int local_thread = (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  

  const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
  const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

  __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
  __SHARED__ T block_vars[elems_per_block][vars_per_elem];
  __SHARED__ T block_res[elems_per_block][vars_per_elem];
  __SHARED__ Data block_data[elems_per_block];

  // load data into block shared mem using some subset of threads
  // if (active_thread && threadIdx.y == 0) {
   
  if (active_thread) { // this version copies same data 3 times (improve this..)

    // would be better if memory access is more in adjacent in memory
    for (int ixpt = threadIdx.y; ixpt < nxpts_per_elem; ixpt+=blockDim.y) {
      int local_inode = ixpt / Geo::spatial_dim;
      int local_idim = ixpt % Geo::spatial_dim;
      const int global_node_ind = geo_conn[global_elem*Geo::num_nodes+local_inode];
      int global_ixpt = Geo::spatial_dim * global_node_ind + local_idim;
      block_xpts[local_elem][ixpt] = xpts[global_ixpt];
    }

    for (int idof = threadIdx.y; idof < vars_per_elem; idof+=blockDim.y) {
      int local_inode = idof / Phys::vars_per_node;
      int local_idof = idof % Phys::vars_per_node;
      const int global_node_ind = vars_conn[global_elem*Geo::num_nodes+local_inode];
      int global_idof = Phys::vars_per_node * global_node_ind + local_idof;

      block_vars[local_elem][idof] = vars[global_idof];
      block_res[local_elem][idof] = 0.0;
    }

    if (local_thread < elems_per_block) {
      int global_elem_thread = local_thread + blockDim.x * blockIdx.x; 
      block_data[local_thread] = physData[global_elem_thread];
    }
  }
  
  __syncthreads();

  printf("<<<sick GPU kernel>>>\n");

  int iquad = threadIdx.y;

  T local_res[vars_per_elem];
  memset(local_res, 0.0, sizeof(T)*vars_per_elem);

  // debug (temporarily change vars of this block to nonzero)
  for (int i = 0; i < 12; i++) {
    block_vars[local_elem][i] = 0.12 + 0.24 * i + 0.03 * i * i;
  }

  ElemGroup::template add_element_quadpt_residual<Data>(
    iquad, block_xpts[local_elem], block_vars[local_elem], block_data[local_elem], local_res
  );

  // atomic add into global res

}

#else // USE_GPU is false (CPU section)

template <typename T, class ElemGroup, class Data>
static void add_residual_cpu(int32_t num_elements, int32_t *geo_conn, int32_t *vars_conn, T *xpts, T *vars, Data *physData, T *residual) {
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;
  using Quadrature = typename Phys::Quadrature;
  
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
      ElemGroup::template add_element_quadpt_residual<Data>(
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

#endif // USE_GPU

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
      add_residual_cpu<T, ElemGroup, Data>(num_elements, geo_conn, vars_conn, xpts, vars, physData, residual);
    #endif
  }
};
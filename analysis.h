#include <cstdint>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_error.h"
#include "a2dcore.h"

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
#else
#define __SHARED__
#define __HOST_DEVICE__
#define __DEVICE__
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

    for (int idim = 0; idim < spatial_dim; idim++) {
      for (int icomp = 0; icomp < 2; icomp++) {
        int ind = 2 * icomp + idim;
        dXdxi[ind] = 0.0;
        for (int inode = 0; inode < num_nodes; inode++) {
          dXdxi[ind] += xpts[num_nodes * idim + inode] * dNdxi[2 * inode + icomp];
        }
      }
    }
    
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
        A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL, true>(Un, dNdxi, dUdxi);

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
  __HOST_DEVICE__ static void computeWeakRes(Data elemData, T scale, PlaneMat dUdx_, PlaneMat weak_res) {
    A2D::ADObj<PlaneMat> dUdx(dUdx_);
    A2D::ADObj<A2D::SymMat<T,2>> strain, stress;
    A2D::ADObj<T> energy;

    const double mu = 0.5 * elemData.E / (1.0 + elemData.nu);
    const double lambda = 2 * mu * elemData.nu / (1.0 - elemData.nu);

    auto strain_energy_stack = A2D::MakeStack(
      A2D::MatGreenStrain<strainType>(dUdx, strain),
      A2D::SymIsotropic(mu, lambda, strain, stress),
      A2D::SymMatMultTrace(strain, stress, energy)
    );

    energy.bvalue() = 0.5 * scale * elemData.t;
    strain_energy_stack.reverse();

    weak_res = dUdx.bvalue();

  }

};

template <typename T, typename ElemGroup>
class ElementAssembler {
 public:
  using Geo = typename ElemGroup::Geo;
  using Basis = typename ElemGroup::Basis;
  using Phys = typename ElemGroup::Phys;
  using Data = typename Phys::Data;
  static constexpr int32_t nodes_per_elem = Basis::num_nodes;
  static constexpr int32_t spatial_dim = Geo::spatial_dim;
  static constexpr int32_t vars_per_node = Phys::vars_per_node;

  // dummy constructor for random points (another one will be made for actual connectivity)
  ElementAssembler(int32_t num_nodes_, int32_t num_elements_, Data *h_elemData) : num_nodes(num_nodes_), num_elements(num_elements_) {
    // randomly initialize data on GPU
    #ifdef USE_GPU

      printf("starting constructor...\n");

      // randomly generate the connectivity
      int N = nodes_per_elem * num_elements;
      int32_t *h_conn = new int32_t[N];
      for (int i = 0; i < N; i++) {
        h_conn[i] = rand() % num_nodes;
      }
      cudaMalloc((void**)&conn, N * sizeof(int32_t));
      cudaMemcpy(conn, h_conn, N * sizeof(int32_t), cudaMemcpyHostToDevice);

      // initialize and allocate data on the device
      int32_t num_xpts = num_nodes * spatial_dim;
      T *h_X = new T[num_xpts];
      for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
        h_X[ixpt] = static_cast<double>(rand()) / RAND_MAX;
      }
      
      cudaMalloc((void**)&X, num_xpts * sizeof(T));
      cudaMemcpy(X, h_X, num_xpts * sizeof(T), cudaMemcpyHostToDevice);

      int32_t num_vars = vars_per_node * num_nodes;
      cudaMalloc((void**)&vars, num_vars * sizeof(T));
      cudaMemset(vars, 0.0, num_vars * sizeof(T));

      cudaMalloc((void**)&residual, num_vars * sizeof(T));
      cudaMemset(residual, 0.0, num_vars * sizeof(T));     

      cudaMalloc((void**)&elemData, num_elements * sizeof(Data));
      cudaMemcpy(elemData, h_elemData, num_elements * sizeof(Data), cudaMemcpyHostToDevice);

      printf("finished constructor\n");

    #else
    #endif
  };

//  template <class ExecParameters>
 void add_residual(T *res) {
  ElemGroup::template add_residual<Data>(num_elements, conn, X, vars, elemData, residual);
  // copies global residual to that
 };

 private:
  int32_t num_nodes;
  int32_t num_elements;  // Number of elements of this type
  int32_t *conn;        // Node numbers for each element

  // Global solution and node numbers
  Data *elemData;
  T *vars;
  T *X;
  T *residual;
};

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1>
__global__ static void add_residual_gpu(int32_t num_elements, int32_t *conn, T *xpts, T *vars, Data *elemData, T *residual) {
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
      const int global_node_ind = conn[global_elem*Geo::num_nodes+local_inode];
      int global_ixpt = Geo::spatial_dim * global_node_ind + local_idim;
      block_xpts[local_elem][ixpt] = xpts[global_ixpt];
    }

    for (int idof = threadIdx.y; idof < vars_per_elem; idof+=blockDim.y) {
      int local_inode = idof / Phys::vars_per_node;
      int local_idof = idof % Phys::vars_per_node;
      const int global_node_ind = conn[global_elem*Geo::num_nodes+local_inode];
      int global_idof = Phys::vars_per_node * global_node_ind + local_idof;

      block_vars[local_elem][idof] = vars[global_idof];
      block_res[local_elem][idof] = 0.0;
    }

    if (local_thread < elems_per_block) {
      int global_elem_thread = local_thread + blockDim.x * blockIdx.x; 
      block_data[local_thread] = elemData[global_elem_thread];
    }
  }
  
  __syncthreads();

  printf("<<<sick GPU kernel>>>\n");

  int iquad = threadIdx.y;

  T local_res[vars_per_elem];
  memset(local_res, 0.0, sizeof(T)*vars_per_elem);

  ElemGroup::template add_element_quadpt_residual<Data>(
    iquad, block_xpts[local_elem], block_vars[local_elem], block_data[local_elem], local_res
  );

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
  using Quadrature = typename Geo::Quadrature;

  static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
  static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;

  template <class Data>
  __HOST_DEVICE__ static void add_element_quadpt_residual(
    const int iquad,
    const T xpts[xpts_per_elem], 
    const T vars[dof_per_elem], 
    const Data elemData,
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
    A2D::ADObj<A2D::Mat<T, Phys::vars_per_node, 2>> dUdxi;
    Basis::template interpParamGradient<Phys::vars_per_node>(pt, vars, dUdxi.value().get_data());

    // Transform to gradient in real space dudx = dudxi * J^-1
    A2D::ADObj<A2D::Mat<T, Phys::vars_per_node, Geo::spatial_dim>> dUdx;
    A2D::MatMatMult(dUdxi, Jinv, dUdx);

    // for (int i = 0; i < 4; i++) {
    //   printf("dUdxi[%d] = %.8e\n", i, dUdxi[i]);
    // }

    T scale = detJ * weight;

    // Compute weak residual (derivative of energy w.r.t state gradient)
    Phys::computeWeakRes(elemData, scale, dUdx.value(), dUdx.bvalue());

    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(dUdx.bvalue(), Jinv, dUdxi.bvalue());

    Basis::template addInterpParamGradientSens<Phys::vars_per_node>(pt, dUdxi.bvalue().get_data(), res);


  }

  // __device__ void element_jacobian(const T* xpts, const T* vars, T* jac) {}

  // template <int32_t elems_per_block = 1>
  template <class Data>
  static void add_residual(int32_t num_elements, int32_t *conn, T *soln, T *X, Data *elemData, T *residual) {
    #ifdef USE_GPU
      const int elems_per_block = 32;
      dim3 block(elems_per_block,3);

      int nblocks = (num_elements + elems_per_block - 1)/ elems_per_block; 
      dim3 grid(nblocks);

      using ElemGroup = ElementGroup<T, Geo, Basis, Phys>;

      // add_residual_gpu<T, ElemGroup, elems_per_block> <<<grid, block>>>(num_elements, conn, soln, X, residual);
      add_residual_gpu<T, ElemGroup, Data, 1> <<<1, 1>>>(num_elements, conn, soln, X, elemData, residual);

      gpuErrchk(cudaDeviceSynchronize());

    #else // CPU data
      // maybe a way to call add_residual_kernel as same method on CPU
      // with elems_per_block = 1
      // add_residual_cpu(num_elements, conn, soln, X, residual);
    #endif
  }
};
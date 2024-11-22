

// add_residual kernel
// -----------------------

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

// add jacobian kernel
// -------------------

// template <int32_t elems_per_block = 1>
// __HOST_DEVICE__ static void add_jacobian_gpu() {
//   __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
//   __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];
//   __SHARED__ T jac[elems_per_block][Basis::num_dof * Basis::num_dof];
// }

#include "cuda_utils.h"

// base class methods to launch kernel depending on how many elements per block
// may override these in some base classes

// add_residual kernel
// -----------------------

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1>
__GLOBAL__ void add_energy_gpu(int32_t num_elements, int32_t *geo_conn, int32_t *vars_conn, T *xpts,
                               T *vars, Data *physData, T *Uenergy) {}

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void add_residual_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                 const Vec<int32_t> vars_conn, const Vec<T> xpts, const Vec<T> vars,
                                 Vec<Data> physData, const int32_t *perm, Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // if you want to precompute some things?
    // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
    // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // accessing value crashes the kernel.. (block_xpts not initialized right)
    // printf("block_xpts:");
    // printVec<T>(12,&block_xpts[local_elem][0]);

    ElemGroup::template add_element_quadpt_residual<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], local_res);

    Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                              &block_res[local_elem][0]);

    res.addElementValuesFromShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                                   Basis::num_nodes, perm, vars_elem_conn,
                                   &block_res[local_elem][0]);
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_sectional_loads_kernel(const int32_t num_elements,
                                               const Vec<int32_t> geo_conn,
                                               const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                               const Vec<T> vars, Vec<Data> physData, const int32_t *perm, 
                                               Vec<T> loads, Vec<int> load_cts) {
    // compute sectional resultants like in plane loads, moments
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int nodes_per_elem = Basis::num_nodes;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_loads[elems_per_block][vars_per_node];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_loads[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_loads[vars_per_node];
    memset(quadpt_loads, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_sectional_loads<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_loads);

    // use nodal averaging of the stresses here, alternative is max among quadpts
    // this should make stress field smoother (can be slightly unconservative sometimes)
    // but prevents preliminary design optimization from responding to non-physical stress
    // concentrations from kinks in a structure (artifact of shell elements)

    for (int idof = 0; idof < vars_per_node; idof++) {
        atomicAdd(&block_loads[local_elem][0], 1.0 / num_quad_pts * quadpt_loads[idof]);
    }

    // now add from shared to global (same averaged vars_per_node x 1 stresses added to each node in
    // element) stress cts later used to complete the averaged at nodal level, that is nodal stress
    // = sum stresses / #element stresses added to node don't use perm here since this goes to
    // visualization and not to solve

    // parallelize across quadpt dimension of block here
    for (int i = threadIdx.y; i < vars_per_elem; i += blockIdx.y) {
        int local_inode = i / vars_per_node;
        int idof = i % vars_per_node;
        int global_inode = vars_elem_conn[local_inode];
        // note we don't use perm here since this goes to visualization not solve

        atomicAdd(&loads[vars_per_node * global_inode + idof], block_loads[local_elem][idof]);
    }

    // also add up load counts
    for (int inode = threadIdx.y; inode < nodes_per_elem; inode += blockIdx.y) {
        int global_inode = vars_elem_conn[inode];
        atomicAdd(&load_cts[global_inode], 1.0);
    }

}  // end of compute_sectional_loads_kernel

// add_jacobian_gpu kernel
// -------------------
template <typename T, class ElemGroup, class Data, int32_t elems_per_block,
          template <typename> class Vec, class Mat>
__GLOBAL__ static void add_jacobian_gpu(int32_t vars_num_nodes, int32_t num_elements,
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> physData, const int32_t *perm, Vec<T> res, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    // if you want to precompute some things?
    // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
    // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_elem2 = vars_per_elem * vars_per_elem;
    const int num_vars = vars_num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    // currently using 1416 T values per block on this element (want to be below
    // 6000 doubles shared mem)
    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ T block_mat[elems_per_block][vars_per_elem2];
    __SHARED__ Data block_data[elems_per_block];

    int nthread_yz = blockDim.y * blockDim.z;
    int thread_yz = threadIdx.y * blockDim.z + threadIdx.z;
    int global_elem_thread = local_thread + blockDim.x * blockIdx.x;

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, thread_yz, nthread_yz, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, thread_yz, nthread_yz, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));
        memset(&block_mat[local_elem][0], 0.0, vars_per_elem2 * sizeof(T));

        if (local_thread < elems_per_block) {
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    int ideriv = threadIdx.y;
    int iquad = threadIdx.z;

    // TODO : should I remove this memory? going over registers with this?
    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);
    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    // call the device function to get one column of the element stiffness
    // matrix at one quadrature point
    ElemGroup::template add_element_quadpt_jacobian_col<Data>(
        active_thread, iquad, ideriv, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], local_res, local_mat_col);

    // TODO : warp shuffle add among threads before into shared
    // this reduces number of atomic calls

    Vec<T>::copyLocalToShared(active_thread, 1.0 / blockDim.y, vars_per_elem, &local_res[0],
                              &block_res[local_elem][0]);

    // copies column into row of block_mat since sym kelem matrix (assumption)
    Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_mat_col[0],
                              &block_mat[local_elem][vars_per_elem * ideriv]);
    __syncthreads();

    // if (local_thread == 0) {
    //     printf("block_mat: ");
    //     printVec<double>(vars_per_elem2, &block_mat[local_elem][0]);
    // }

    // printf("blockMat:");
    // printVec<double>(576,block_mat[local_elem]);

    res.addElementValuesFromShared(active_thread, thread_yz, nthread_yz, Phys::vars_per_node,
                                   Basis::num_nodes, perm, vars_elem_conn,
                                   &block_res[local_elem][0]);

    mat.addElementMatrixValuesFromShared(active_thread, thread_yz, nthread_yz, 1.0, global_elem,
                                         Phys::vars_per_node, Basis::num_nodes, vars_elem_conn,
                                         &block_mat[local_elem][0]);

}  // end of add_jacobian_gpu

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_strains_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                       const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                       const Vec<T> vars, Vec<Data> physData, const int *perm,
                                       Vec<T> strains, Vec<int> strain_cts) {
    // compute the strains for the kernel
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int nodes_per_elem = Basis::num_nodes;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_strains[elems_per_block][vars_per_node];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_strains[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_strains[vars_per_node];
    memset(quadpt_strains, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_strains<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_strains);

    // use nodal averaging of the stresses here, alternative is max among quadpts
    // this should make stress field smoother (can be slightly unconservative sometimes)
    // but prevents preliminary design optimization from responding to non-physical stress
    // concentrations from kinks in a structure (artifact of shell elements)

    for (int idof = 0; idof < vars_per_node; idof++) {
        atomicAdd(&block_strains[local_elem][0], 1.0 / num_quad_pts * quadpt_strains[idof]);
    }

    // now add from shared to global (same averaged vars_per_node x 1 stresses added to each node in
    // element) stress cts later used to complete the averaged at nodal level, that is nodal stress
    // = sum stresses / #element stresses added to node don't use perm here since this goes to
    // visualization and not to solve

    // parallelize across quadpt dimension of block here
    for (int i = threadIdx.y; i < vars_per_elem; i += blockIdx.y) {
        int local_inode = i / vars_per_node;
        int idof = i % vars_per_node;
        int global_inode = vars_elem_conn[local_inode];
        // note we don't use perm here since this goes to visualization not solve

        atomicAdd(&strains[vars_per_node * global_inode + idof], block_strains[local_elem][idof]);
    }

    // also add up load counts
    for (int inode = threadIdx.y; inode < nodes_per_elem; inode += blockIdx.y) {
        int global_inode = vars_elem_conn[inode];
        atomicAdd(&strain_cts[global_inode], 1.0);
    }

}  // end of compute_sectional_loads_kernel


template <typename T, class ElemGroup, template <typename> class Vec>
__GLOBAL__ void normalize_states(Vec<T> states, Vec<int> state_cts) {
    // normalize states : stresses or strains averaged to nodal level
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;

    int global_node = blockIdx.x;
    T *loc_state = states[vars_per_node * global_node];
    int nodal_state_ct = state_cts[global_node];

    for (int idof = threadIdx.x; idof < vars_per_node; idof += blockDim.x) {
        loc_state[idof] /= nodal_state_ct;
    }

}  // end of normalize_stresses_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData, const int *perm, 
                                         T rho_KS, T *sumexp_ksFailure) {
    // computes average strains in the element (among quadpts)
    // then KS failure on those strains (could also max among quadpts but that gets more expensive
    // for optimization potentially) doesn't make sense to use nodal average strains for KS failure
    // since yield stress is an element data and is not well-defined for nodes
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_avg_strains[elems_per_block][vars_per_node];
    __SHARED__ T block_ksfail[elems_per_block];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_avg_strains[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_strains[vars_per_node];
    memset(quadpt_strains, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_strains<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_strains);

    // now average strains among all quadpts in the element
    // parallelized among threadIdx.y / blockDim.y
    for (int idof = 0; idof < vars_per_node; idof++) {
        // do we need atomicAdd here?
        block_avg_strains[local_elem][idof] += 1.0 / num_quad_pts * quadpt_strains[idof];
        // atomicAdd(&block_avg_strains[local_elem][idof], 1.0 / num_quad_pts *
        // quadpt_strains[idof]);
    }
    __syncthreads();

    // now compute KS failure for each element
    Phys::computeKSFailure(block_data[local_elem], rho_KS, &block_avg_strains[local_elem][0],
                           &block_ksfail[local_elem]);

    // Compute exp(rho_KS * block_ksfail[local_elem])
    T lane_val = exp(rho_KS * block_ksfail[local_elem]);

    // Perform warp reduction within the 32 threads
    for (int offset = elems_per_block / 2; offset > 0; offset /= 2) {
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, offset);
    }

    // Store warp-level reduction in shared memory
    __SHARED__ T warp_sum;
    if (threadIdx.x == 0) {
        warp_sum = 0.0;
    }
    __syncthreads();

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&warp_sum, lane_val);
    }
    __syncthreads();

    // Final atomicAdd from the block to global sumexp_ksFailure
    if (threadIdx.x == 0) {
        atomicAdd(sumexp_ksFailure, warp_sum);
    }

}  // end of compute_ksfailure_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_mass_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                    const Vec<T> xpts, Vec<Data> physData, const int *perm, T *total_mass) {
    // computes total mass of the structure
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_masses[elems_per_block];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    if (active_thread) {
        if (local_thread == 0) {
            memset(block_masses, 0.0, elems_per_block * sizeof(T));
        }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T *quadpt_mass;

    ElemGroup::template compute_element_quadpt_mass<Data>(
        active_thread, iquad, block_xpts[local_elem], block_data[local_elem], quadpt_mass);

    // now addup masses among each of the quadpts into shared memory
    block_masses[local_elem] += quadpt_mass;
    __syncthreads();

    // concerned this will duplicate 4x for each quadpt?
    T lane_val = block_masses[local_elem];

    // Perform warp reduction within the 32 threads
    for (int offset = elems_per_block / 2; offset > 0; offset /= 2) {
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, offset);
    }

    // Store warp-level reduction in shared memory
    __SHARED__ T warp_sum;
    if (threadIdx.x == 0) {
        warp_sum = 0.0;
    }
    __syncthreads();

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&warp_sum, lane_val);
    }
    __syncthreads();

    // Final atomicAdd from the block to global sumexp_ksFailure
    if (threadIdx.x == 0) {
        atomicAdd(total_mass, warp_sum);
    }

}  // end of compute_mass_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_mass_DVsens_kernel(const int32_t num_elements,
                                           const Vec<int32_t> elem_components,
                                           const Vec<int32_t> geo_conn, const Vec<T> xpts,
                                           Vec<Data> physData, const int *perm, Vec<T> dfdx) {
    // computes total mass of the structure
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int num_quad_pts = Quadrature::num_quad_pts;
    const int num_local_dvs = Phys::num_local_dvs;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ Data block_data[elems_per_block];
    __SHARED__ T block_dmdx[elems_per_block][num_local_dvs];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    if (active_thread) {
        memset(&block_dmdx[local_elem][0], 0.0, num_local_dvs * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    // for this element, quadpt contribution
    T local_dmdx[num_local_dvs];

    ElemGroup::template compute_element_quadpt_dmass_dx<Data>(
        active_thread, iquad, block_xpts[local_elem], block_data[local_elem], &local_dmdx[0]);

    // now addup masses among each of the quadpts into shared memory
    for (int local_dv = 0; local_dv < num_local_dvs; local_dv++) {
        // should this be atomicAdd?
        block_dmdx[local_elem][local_dv] += local_dmdx[local_dv];
    }
    __syncthreads();

    // go back and make warp reduction later
    // add from shared to global

    // dfdx is basically a (num_components, num_local_dvs) row-major matrix
    for (int local_dv = threadIdx.y; local_dv < num_local_dvs; local_dv += blockDim.y) {
        // compnent num for this element
        int icomponent = elem_components[global_elem];
        atomicAdd(&dfdx[num_local_dvs * icomponent + local_dv], block_dmdx[local_elem][local_dv]);
    }

}  // end of compute_mass_DVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_DVsens_kernel(const int32_t num_elements,
                                                const Vec<int32_t> elem_components,
                                                const Vec<int32_t> geo_conn,
                                                const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                                const Vec<T> vars, Vec<Data> physData, const int *perm,
                                                T rho_KS, T sumexp_kfail, Vec<T> dfdx) {
    // computes average strains in the element (among quadpts)
    // then KS failure on those strains (could also max among quadpts but that gets more expensive
    // for optimization potentially) doesn't make sense to use nodal average strains for KS failure
    // since yield stress is an element data and is not well-defined for nodes
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;
    const int num_local_dvs = Phys::num_dvs;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_avg_strains[elems_per_block][vars_per_node];
    __SHARED__ T block_ksfail[elems_per_block];
    __SHARED__ Data block_data[elems_per_block];
    __SHARED__ T block_dfdxe[elems_per_block][num_local_dvs];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_avg_strains[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_strains[vars_per_node];
    memset(quadpt_strains, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_strains<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_strains);

    // now average strains among all quadpts in the element
    // parallelized among threadIdx.y / blockDim.y
    for (int idof = 0; idof < vars_per_node; idof++) {
        // do we need atomicAdd here?
        block_avg_strains[local_elem][idof] += 1.0 / num_quad_pts * quadpt_strains[idof];
        // atomicAdd(&block_avg_strains[local_elem][idof], 1.0 / num_quad_pts *
        // quadpt_strains[idof]);
    }
    __syncthreads();

    // now compute KS failure for each element
    Phys::computeKSFailure(block_data[local_elem], rho_KS, &block_avg_strains[local_elem][0],
                           &block_ksfail[local_elem]);

    // now get df/dksfail_elem
    T df_dksfail_elem = exp(rho_KS * (block_ksfail[local_elem])) / sumexp_kfail;

    // now backprop that through to df/dxe for each element
    Phys::computeKSFailureDVSens(block_data[local_elem], rho_KS, &block_avg_strains[local_elem][0],
                                 df_dksfail_elem, &block_dfdxe[local_elem][0]);

    // now add df/dxe into global df/dx based on which component it's from
    for (int local_dv = threadIdx.y; local_dv < num_local_dvs; local_dv += blockDim.y) {
        // compnent num for this element
        int icomponent = elem_components[global_elem];
        atomicAdd(&dfdx[num_local_dvs * icomponent + local_dv], block_dfdxe[local_elem][local_dv]);
    }

}  // end of compute_ksfailure_DVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_SVsens_kernel(const int32_t num_elements, const int32_t *perm,
                                                const Vec<int32_t> geo_conn,
                                                const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                                const Vec<T> vars, Vec<Data> physData, T rho_KS,
                                                T sumexp_kfail, Vec<T> dfdu) {
    // computes average strains in the element (among quadpts)
    // then KS failure on those strains (could also max among quadpts but that gets more expensive
    // for optimization potentially) doesn't make sense to use nodal average strains for KS failure
    // since yield stress is an element data and is not well-defined for nodes
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;
    const int num_local_dvs = Phys::num_dvs;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_avg_strains[elems_per_block][vars_per_node];
    __SHARED__ T block_ksfail[elems_per_block];
    __SHARED__ Data block_data[elems_per_block];
    __SHARED__ T block_avg_strain_bar[elems_per_block][vars_per_node];
    __SHARED__ T block_vars_bar[elems_per_block][vars_per_elem];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_avg_strains[local_elem][0], 0.0, vars_per_node * sizeof(T));
        memset(&block_avg_strain_bar[local_elem][0], 0.0, vars_per_node * sizeof(T));
        memset(&block_vars_bar[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_strains[vars_per_node];
    memset(quadpt_strains, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_strains<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_strains);

    // now average strains among all quadpts in the element
    // parallelized among threadIdx.y / blockDim.y
    for (int idof = 0; idof < vars_per_node; idof++) {
        // do we need atomicAdd here?
        block_avg_strains[local_elem][idof] += 1.0 / num_quad_pts * quadpt_strains[idof];
        // atomicAdd(&block_avg_strains[local_elem][idof], 1.0 / num_quad_pts *
        // quadpt_strains[idof]);
    }
    __syncthreads();

    // now compute KS failure for each element
    Phys::computeKSFailure(block_data[local_elem], rho_KS, &block_avg_strains[local_elem][0],
                           &block_ksfail[local_elem]);

    // now get df/dksfail_elem
    T df_dksfail_elem = exp(rho_KS * (block_ksfail[local_elem])) / sumexp_kfail;

    // now backprop that through to df/dstrain_avg for each element
    Phys::computeKSFailureSVSens(block_data[local_elem], rho_KS, &block_avg_strains[local_elem][0],
                                 df_dksfail_elem, &block_avg_strain_bar[local_elem][0]);

    // now backprop from avg strains through vars at each individual quadpt, same backprop each
    T local_vars_bar[vars_per_elem];
    memset(local_vars_bar, 0.0, vars_per_elem * sizeof(T));
    ElemGroup::template compute_element_quadpt_strains_SVsens<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], &block_avg_strain_bar[local_elem][0], local_vars_bar);

    // add from local to shared
    for (int idof = 0; idof < vars_per_elem; idof++) {
        atomicAdd(&block_vars_bar[local_elem][idof], local_vars_bar[idof]);
    };

    // add from shared to global
    dfdu.addElementValuesFromShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                                    Basis::num_nodes, perm, vars_elem_conn,
                                    &block_vars_bar[local_elem][0]);

}  // end of compute_ksfailure_SVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_adjResProduct_kernel(const int32_t num_elements, const int32_t *perm,
                                             const Vec<int> elem_components,  const Vec<int32_t> geo_conn, 
                                             const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                             const Vec<T> vars, Vec<Data> physData, Vec<T> psi,
                                             Vec<T> dfdx) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // if you want to precompute some things?
    // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
    // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int num_local_dvs = Phys::num_dvs;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_dfdxe[elems_per_block][num_local_dvs];
    __SHARED__ T block_psi[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                                Geo::num_nodes, perm, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                                Basis::num_nodes, perm, vars_elem_conn, &block_vars[local_elem][0]);

    psi.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                               Basis::num_nodes, perm, vars_elem_conn, &block_psi[local_elem][0]);

    if (active_thread) {
        memset(&block_dfdxe[local_elem][0], 0.0, num_local_dvs * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T local_dRdx[vars_per_elem];

    // accessing value crashes the kernel.. (block_xpts not initialized right)
    // printf("block_xpts:");
    // printVec<T>(12,&block_xpts[local_elem][0]);

    for (int local_dv = 0; local_dv < num_local_dvs; local_dv++) {
        ElemGroup::template add_element_quadpt_adjResProduct<Data>(
            active_thread, iquad, local_dv, block_xpts[local_elem], block_vars[local_elem],
            block_data[local_elem], local_dRdx);

        block_dfdxe[local_elem][local_dv] +=
            A2D::VecDotCore<T, vars_per_elem>(local_dRdx, &block_psi[local_elem][0]);
    }

    // TODO : this just does it for one DV right now, in future need for loop over DVs here
    // probably.. (For stiffened panel case)

    // compute dot product with local psi
    // Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res_product[0],
    //                           &block_res_product[local_elem][0]);

    // now add df/dxe into global df/dx based on which component it's from
    for (int local_dv = threadIdx.y; local_dv < num_local_dvs; local_dv += blockDim.y) {
        // compnent num for this element
        int icomponent = elem_components[global_elem];
        atomicAdd(&dfdx[num_local_dvs * icomponent + local_dv], block_dfdxe[local_elem][local_dv]);
    }
}  // end of add_residual_gpu kernel
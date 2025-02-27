
#include "cuda_utils.h"

// base class methods to launch kernel depending on how many elements per block
// may override these in some base classes

// add_residual kernel
// -----------------------

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1>
__GLOBAL__ void add_energy_gpu(int32_t num_elements, int32_t *geo_conn,
                               int32_t *vars_conn, T *xpts, T *vars,
                               Data *physData, T *Uenergy) {}

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void
add_residual_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                 const Vec<int32_t> vars_conn, const Vec<T> xpts,
                 const Vec<T> vars, Vec<int> bcs, Vec<Data> physData, const int32_t *perm, Vec<T> res)
{

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
    int local_thread = (blockDim.x * blockDim.y) * threadIdx.z +
                       blockDim.x * threadIdx.y + threadIdx.x;

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
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y,
                                Geo::spatial_dim, Geo::num_nodes, perm, geo_elem_conn,
                                &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y,
                                Phys::vars_per_node, Basis::num_nodes,
                                perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread)
    {
        memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));

        if (local_thread < elems_per_block)
        {
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

    res.addElementValuesFromShared(active_thread, threadIdx.y, blockDim.y,
                                   Phys::vars_per_node, Basis::num_nodes,
                                   perm, vars_elem_conn, &block_res[local_elem][0]);
} // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void
compute_stresses_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                        const Vec<int32_t> vars_conn, const Vec<T> xpts,
                        const Vec<T> vars, Vec<int> bcs, Vec<Data> physData, Vec<T> stresses, Vec<int> stress_cts)
{

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread = (blockDim.x * blockDim.y) * threadIdx.z +
                       blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_stresses[elems_per_block][vars_per_node];
    __SHARED__ T stress_cts[elems_per_block][vars_per_node];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y,
                            Geo::spatial_dim, Geo::num_nodes, perm, geo_elem_conn,
                            &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y,
                            Phys::vars_per_node, Basis::num_nodes,
                            perm, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread)
    {
        memset(&block_stresses[local_elem][0], 0.0, vars_per_ndoe * sizeof(T));

        if (local_thread < elems_per_block)
        {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_stress[vars_per_node];
    memset(quadpt_stress, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_stresses<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_stress);

    // use nodal averaging of the stresses here, alternative is max among quadpts
    // this should make stress field smoother (can be slightly unconservative sometimes)
    // but prevents preliminary design optimization from responding to non-physical stress concentrations
    // from kinks in a structure (artifact of shell elements)

    for (int idof = 0; idof < vars_per_node; idof++)
    {
        atomicAdd(&block_stresses[local_elem][0], 1.0 / num_quad_pts * quadpt_stress[idof]);
    }

    // now add from shared to global (same averaged vars_per_node x 1 stresses added to each node in element)
    // stress cts later used to complete the averaged at nodal level, that is nodal stress = sum stresses / #element stresses added to node
    // don't use perm here since this goes to visualization and not to solve

    // parallelize across quadpt dimension of block here
    for (int i = threadIdx.y; i < vars_per_elem; i += blockIdx.y)
    {
        int local_inode = i / vars_per_node;
        int idof = i % vars_per_node;
        int global_inode = vars_elem_conn[local_inode];
        // note we don't use perm here since this goes to visualization not solve

        atomicAdd(&stresses[dof_per_node * global_inode + idof], block_stresses[local_elem][idof]);
    }

    // also add up stress counts
    for (int inode = threadIdx.y; inode < nodes_per_elem; inode++)
    {
        int global_inode = vars_elem_conn[inode];
        atomicAdd(&stress_cts[global_inode], 1.0);
    }

} // end of compute_stresses_kernel

template <typename T, class ElemGroup, template <typename> class Vec>
__GLOBAL__ void
normalize_stresses_kernel(Vec<T> stresses, Vec<int> stress_cts)
{

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;

    int global_node = blockIdx.x;
    T *loc_stress = stresses[vars_per_node * global_inode];
    int nodal_stress_ct = stress_cts[global_inode];

    for (int idof = threadIdx.x; idof < vars_per_node; idof += blockDim.x)
    {
        loc_stress[idof] /= nodal_stress_ct;
    }

} // end of normalize_stresses_kernel

template <typename T, class ElemGroup, template <typename> class Vec>
__GLOBAL__ void
compute_ks_failure_kernel(Vec<T> stresses, T rho_KS, T *sum_exp_stress)
{

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;

    int global_node = blockIdx.x;
    T *loc_stress = stresses[vars_per_node * global_inode];
    int nodal_stress_ct = stress_cts[global_inode];

    for (int idof = threadIdx.x; idof < vars_per_node; idof += blockDim.x)
    {
        loc_stress[idof] /= nodal_stress_ct;
    }

    // now compute local failure indices in shared memory

    // apply warp reduction

    // then global atomicSum with exp(rho*vi) into ks_stress

} // end of normalize_stresses_kernel

// add_jacobian_gpu kernel
// -------------------
template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec, class Mat>
__GLOBAL__ static void
add_jacobian_gpu(int32_t vars_num_nodes, int32_t num_elements,
                 Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                 Vec<T> vars, Vec<Data> physData, Vec<T> res, Mat mat)
{
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    // if you want to precompute some things?
    // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
    // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread = (blockDim.x * blockDim.y) * threadIdx.z +
                       blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_elem2 = vars_per_elem * vars_per_elem;
    const int num_vars = vars_num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();
    const int32_t *perm = mat.getBsrData().perm;

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
    xpts.copyElemValuesToShared(active_thread, thread_yz, nthread_yz,
                                Geo::spatial_dim, Geo::num_nodes, geo_elem_conn,
                                &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, thread_yz, nthread_yz,
                                Phys::vars_per_node, Basis::num_nodes, vars_elem_conn,
                                &block_vars[local_elem][0]);

    if (active_thread)
    {
        memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));
        memset(&block_mat[local_elem][0], 0.0, vars_per_elem2 * sizeof(T));

        if (local_thread < elems_per_block)
        {
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
        active_thread, iquad, ideriv, block_xpts[local_elem],
        block_vars[local_elem], block_data[local_elem], local_res,
        local_mat_col);

    // TODO : warp shuffle add among threads before into shared
    // this reduces number of atomic calls

    Vec<T>::copyLocalToShared(active_thread, 1.0 / blockDim.y, vars_per_elem,
                              &local_res[0], &block_res[local_elem][0]);

    // copies column into row of block_mat since sym kelem matrix (assumption)
    Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem,
                              &local_mat_col[0],
                              &block_mat[local_elem][vars_per_elem * ideriv]);
    __syncthreads();

    // if (local_thread == 0) {
    //     printf("block_mat: ");
    //     printVec<double>(vars_per_elem2, &block_mat[local_elem][0]);
    // }

    // printf("blockMat:");
    // printVec<double>(576,block_mat[local_elem]);

    res.addElementValuesFromShared(active_thread, thread_yz, nthread_yz,
                                   Phys::vars_per_node, Basis::num_nodes,
                                   perm, vars_elem_conn, &block_res[local_elem][0]);

    mat.addElementMatrixValuesFromShared(
        active_thread, thread_yz, nthread_yz, 1.0, global_elem, Phys::vars_per_node,
        Basis::num_nodes, vars_elem_conn, &block_mat[local_elem][0]);

} // end of add_jacobian_gpu
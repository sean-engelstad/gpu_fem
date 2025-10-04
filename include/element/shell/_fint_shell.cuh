#pragma once
#include "cuda_utils.h"

// add_jacobian fast kernel
// -------------------
template <typename T, class ElemGroup, class Data, template <typename> class Vec, class Mat>
__GLOBAL__ static void k_add_jacobian_fast(int32_t vars_num_nodes, int32_t num_elements,
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> physData, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

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

    __SHARED__ T block_xpts[nxpts_per_elem];
    __SHARED__ T block_vars[vars_per_elem];
    __SHARED__ T block_res[vars_per_elem];
    __SHARED__ Data block_data[1];

    int nthread_yz = blockDim.y * blockDim.z;
    int thread_yz = threadIdx.y * blockDim.z + threadIdx.z;
    int global_elem_thread = local_thread + blockDim.x * blockIdx.x;

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, thread_yz, nthread_yz, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, thread_yz, nthread_yz, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[0]);

    if (active_thread) {
        // memset may not work well on GPU
        memset(&block_res[0], 0.0, vars_per_elem * sizeof(T));

        if (local_thread < elems_per_block) {
            block_data[0] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    int ideriv = threadIdx.y;
    int iquad = threadIdx.z;

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    // call the device function to get one column of the element stiffness
    // matrix at one quadrature point

    ElemGroup::template add_elem_drill_strain_jacobian_col<Data>(
        active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
        &block_data[0], local_mat_col
    );
    __syncthreads(); // splits up registers, alternative to inlined functions

    ElemGroup::template add_elem_bending_strain_jacobian_col<Data>(
        active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
        &block_data[0], local_mat_col
    );
    __syncthreads();

    ElemGroup::template add_elem_tying_strain_jacobian_col<Data>(
        active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
        &block_data[0], local_mat_col
    );
    __syncthreads();

    /* warp shuffle here.. */

    // TODO : fast add element matrix from row method (from new)
    mat.addElementMatrixValuesFromShared(active_thread, thread_yz, nthread_yz, 1.0, global_elem,
                                         Phys::vars_per_node, Basis::num_nodes, vars_elem_conn,
                                         &block_mat[local_elem][0]);

    // printf("block_mat[512] = %.4e\n", block_mat[0][512]);

}  // end of add_jacobian_gpu

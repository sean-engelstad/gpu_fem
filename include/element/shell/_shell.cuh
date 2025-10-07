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
    using Quadrature = typename Basis::Quadrature;

    int global_elem = blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_elem2 = vars_per_elem * vars_per_elem;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[nxpts_per_elem];
    __SHARED__ T block_vars[vars_per_elem];
    // __SHARED__ T block_mat[vars_per_elem2];
    __SHARED__ Data block_data[1];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, tid, nthreads, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[0]);
    __syncthreads();

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, tid, nthreads, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[0]);
     __syncthreads();

    if (active_thread) {
        // memset may not work well on GPU
        // memset(&block_mat[0], 0.0, vars_per_elem2 * sizeof(T));
        if (tid == 0) block_data[0] = _phys_data[global_elem];
    }
    __syncthreads();

    int ideriv = threadIdx.y;
    int iquad = threadIdx.x;

    // if (global_elem == 0) {
    //     printf("tid %d; iquad %d, ideriv %d\n", tid, iquad, ideriv);
    // }

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);
    
    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    // old call
    ElemGroup::template add_element_quadpt_jacobian_col<Data>(
        active_thread, iquad, ideriv, block_xpts, block_vars,
        block_data[0], local_res, local_mat_col);

    __syncthreads();

    // call the device function to get one column of the element stiffness
    // matrix at one quadrature point
    // ElemGroup::template add_elem_drill_strain_jacobian_col<Data>(
    //     active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
    //     block_data, local_mat_col
    // );
    // __syncthreads(); // splits up & reduces registers, alternative to no_inline functions

    // ElemGroup::template add_elem_bending_strain_jacobian_col<Data>(
    //     active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
    //     block_data, local_mat_col
    // );
    // __syncthreads();

    // ElemGroup::template add_elem_tying_strain_jacobian_col<Data>(
    //     active_thread, iquad, ideriv, &block_xpts[0], &block_vars[0], 
    //     block_data, local_mat_col
    // );
    // __syncthreads();

    /* warp shuffle here.. */
    int lane = tid % 32;
    int group_start = (lane / 4) * 4;
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_mat_col[idof];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        // warp broadcast
        lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
        local_mat_col[idof] = lane_val;
    }

    int elem_block_row = ideriv / Phys::vars_per_node;
    int elem_inner_row = ideriv % Phys::vars_per_node;
    mat.addElementMatRow(active_thread, elem_block_row, elem_inner_row, global_elem, iquad, Quadrature::num_quad_pts,
        Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);

}  // end of add_jacobian_gpu

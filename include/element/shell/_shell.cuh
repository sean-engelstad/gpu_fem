#pragma once
#include "cuda_utils.h"
#include "strains/basic_utils.h"
#include "strains/strain_types.h"

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
    if (!active_thread) return;
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

    if (tid == 0) block_data[0] = _phys_data[global_elem];
    __syncthreads();

    int ideriv = threadIdx.y;
    int iquad = threadIdx.x;

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    constexpr bool fast_method = true;
    // constexpr bool fast_method = false;

    // OLD slower calls
    if constexpr (!fast_method) {
        T local_res[vars_per_elem];
        memset(local_res, 0.0, sizeof(T) * vars_per_elem);

        // used here for verification + 
        constexpr STRAIN strain = ALL;
        // constexpr STRAIN strain = DRILL;

        ElemGroup::template add_element_quadpt_jacobian_col<Data, strain>(
            active_thread, iquad, ideriv, block_xpts, block_vars,
            block_data[0], local_res, local_mat_col);
        __syncthreads();
    }

    // NEW faster call
    if constexpr (fast_method) {
        // some prelim computations with xpts and quadrature point
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T fn[nxpts_per_elem];
        ShellComputeNodeNormals<T, Basis>(block_xpts, fn);
        T detXd = getDetXd<T, Basis>(pt, block_xpts, fn);
        T scale = detXd * weight; // scale for energy derivatives
        __syncthreads();
        T XdinvT[9], Tmat[9], XdinvzT[9];
        computeShellRotations<T, Basis, Data>(pt, block_data[0].refAxis, block_xpts, fn, Tmat, XdinvT, XdinvzT);
        __syncthreads();

        T p_vars[vars_per_elem];
        memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
        p_vars[ideriv] = 1.0;

        // // compute drill strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, DRILL>(
            pt, scale, block_xpts, fn, 
            XdinvT, Tmat, XdinvzT, block_data[0],
            block_vars, p_vars, local_mat_col);
        __syncthreads();

        // // compute bending strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, BENDING>(
            pt, scale, block_xpts, fn, 
            XdinvT, Tmat, XdinvzT, block_data[0],
            block_vars, p_vars, local_mat_col);
        __syncthreads();

        // // // compute tying strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, TYING>(
            pt, scale, block_xpts, fn, 
            XdinvT, Tmat, XdinvzT, block_data[0],
            block_vars, p_vars, local_mat_col);
        __syncthreads();
    }

    // if (blockIdx.x == 0 && tid == 0) {
    //     printf("local mat col: ");
    //     for (int i = 0; i < 8; i++) {
    //         printVec<T>(3, &local_mat_col[3 * i]);
    //     }
    // }


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
    __syncthreads();

    int elem_block_row = ideriv / Phys::vars_per_node;
    int elem_inner_row = ideriv % Phys::vars_per_node;
    mat.addElementMatRow(active_thread, elem_block_row, elem_inner_row, global_elem, iquad, Quadrature::num_quad_pts,
        Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);

}  // end of add_jacobian_gpu

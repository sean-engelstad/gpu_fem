#pragma once

template <typename T, class ElemGroup, class Data,
          int32_t elems_per_block = 1,
          template <typename> class Vec, int strain_case = 1>
__GLOBAL__ void add_residual_shell_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                 const Vec<int32_t> vars_conn, const Vec<T> xpts, 
                                 const Vec<T> vars, Vec<Data> physData, Vec<T> res) {
    /* a custom kernel that loads in shell transform data */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int iquad = threadIdx.x;
    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int spatial_dim2 = Geo::spatial_dim * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;
    const int nrot_per_elem = nxpts_per_elem * Geo::spatial_dim;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];
    // __SHARED__ T block_normals[elems_per_block][nxpts_per_elem];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);
    

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    const Data *_phys_data = physData.getPtr();
    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    T Tmat[9], XdinvT[9], detXd;
    ElemGroup::template compute_shell_transforms2<Data>(active_thread, iquad, block_xpts[local_elem], block_data[local_elem], Tmat, XdinvT, detXd);

    // ElemGroup::compute_shell_normals(active_thread, block_xpts[local_elem], block_normals[local_elem]);

    // customized split up or concurrent evaluation of each shell strain
    // constexpr int strain_case = 2;
    constexpr bool no_inline = true;

    // if constexpr (strain_case == 1 || no_inline) {
    //     // drill
    //     ElemGroup::template add_element_quadpt_drill_residual<Data>(
    //         active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
    //         block_data[local_elem], Tmat, XdinvT, detXd, local_res);
    // }

    // if constexpr (strain_case == 2 || no_inline) {
    //     // bending
    //     ElemGroup::template add_element_quadpt_bending_residual<Data>(
    //         active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
    //         block_data[local_elem], Tmat, XdinvT, detXd, local_res);
    // }

    // if constexpr (strain_case == 3) {
    //     // tying
    //     // add_element_quadpt_tying_residual
        ElemGroup::template add_element_quadpt_tying_residual<Data>(
            active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
            block_data[local_elem], XdinvT, detXd,  local_res); // block_normals[local_elem],
    // }
                  

    int lane = local_thread % 32;
    int group_start = (lane / 4) * 4;
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_res[idof];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        // warp broadcast
        lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
        local_res[idof] = lane_val;
    }

    res.addElementValuesFromShared(active_thread, iquad, num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, local_res);

}  // end of add_residual_gpu kernel
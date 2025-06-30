template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void add_residual_shell_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                       const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                       const Vec<T> vars, Vec<Data> physData, const Vec<T> Tmat,
                                       const Vec<T> XdinvT, const Vec<T> detXd, Vec<T> res) {
    /* a custom kernel that loads in shell transform data */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;
    using MyBufferOptions = BufferOptions<>;

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

    // shell transform shared data
    // __SHARED__ T block_Tmat[elems_per_block][nrot_per_elem];
    // __SHARED__ T block_XdinvT[elems_per_block][nrot_per_elem];
    // __SHARED__ T block_detXd[elems_per_block][num_quad_pts];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    // load in the shell transforms.. somewhat efficient access pattern
    {
        int iwarp = local_thread / 32;
        int lane = local_thread % 32;
        int nwarps = blockDim.x * blockDim.y / 32;

        // const T *Tmat_ptr = Tmat.getPtr();
        // for (int ielem = iwarp; ielem < elems_per_block; ielem += nwarps) {
        //     // now loop over current values
        //     int c_glob_elem = blockDim.y * blockIdx.x + ielem;
        //     for (int i = lane; i < nrot_per_elem; i += 32) {
        //         block_Tmat[ielem][i] = Tmat_ptr[nrot_per_elem * c_glob_elem + i];
        //     }
        // }

        // const T *XdinvT_ptr = XdinvT.getPtr();
        // for (int ielem = iwarp; ielem < elems_per_block; ielem += nwarps) {
        //     // now loop over current values
        //     int c_glob_elem = blockDim.y * blockIdx.x + ielem;
        //     for (int i = lane; i < nrot_per_elem; i += 32) {
        //         block_XdinvT[ielem][i] = XdinvT_ptr[nrot_per_elem * c_glob_elem + i];
        //     }
        // }

        // const T *detXd_ptr = detXd.getPtr();
        // for (int ielem = iwarp; ielem < elems_per_block; ielem += nwarps) {
        //     // now loop over current values
        //     int c_glob_elem = blockDim.y * blockIdx.x + ielem;
        //     for (int i = lane; i < num_quad_pts; i += 32) {
        //         block_detXd[ielem][i] = detXd_ptr[num_quad_pts * c_glob_elem + i];
        //     }
        // }
    }

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

    // temporary testing concurrent streams
    // ElemGroup::template add_element_quadpt_drill_residual<Data>(
    //         active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
    //         block_data[local_elem], &block_Tmat[local_elem][spatial_dim2 * iquad],
    //         &block_XdinvT[local_elem][spatial_dim2 * iquad], block_detXd[local_elem][iquad],
    //         local_res);

    T loc_Tmat[9], loc_XdinvT[9], loc_detXd;
    ElemGroup::template add_element_quadpt_drill_residual<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], loc_Tmat, loc_XdinvT, loc_detXd, local_res);

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
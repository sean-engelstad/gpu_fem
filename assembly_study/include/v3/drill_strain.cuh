template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void fast_drill_strain_add_residual(const int32_t num_elements,
                                               const Vec<int32_t> geo_conn,
                                               const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               Vec<T> Tmat, Vec<T> XdinvT, 
                                               Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad_node = threadIdx.x;
    // inode is inner most dimension of block
    int inode = iquad_node % Basis::num_nodes;
    int iquad = iquad_node / Basis:num_nodes;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];
    
    // TODO : these guys are pre-computed once for each design (not on re-assembly necessarily)
    // and stored in the assembler (the global vecs)
    const int nmat = 9 * Basis::num_nodes;  // 36 for quad elements
    __SHARED__ T block_Tmat[elems_per_block][nmat];
    __SHARED__ T block_XdinvT[elems_per_block][nmat];

    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];

    T *shared_xpts = &block_xpts[local_elem][0];
    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];
    T *shared_Tmat = &block_Tmat[local_elem][0];
    T *shared_XdinvT = &block_XdinvT[local_elem][0];
    T *shared_res = &block_res[local_elem][0];

    xpts.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, shared_xpts);
    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);
    Tmat.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, 9,
                                Basis::num_nodes, vars_elem_conn, shared_Tmat);
    XdinvT.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, 9,
                                Basis::num_nodes, vars_elem_conn, shared_XdinvT);

    T local_res[vars_per_elem];
#pragma unroll
    for (int i = iquad; i < 24; i += 4) {
        shared_res[i] = 0.0;
        // local_res[i] = 0.0;
    }

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    ElemGroup::template _add_drill_strain_quadpt_residual<Data>(
        active_thread, iquad, shared_xpts, shared_vars, shared_data, shared_vars, shared_vars,
        local_res);

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt & node group of 16 in case of QuadBasis and QuadElement
            // shuffles down by 8, 4, 2, 1 to reduce every 16 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 8);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                shared_res[i] = val;
            }
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  shared_res);
        __syncthreads();
    }

    // TODO : graph coloring to reduce cost of atomicAdd? graph coloring ensures no two elements share a node in the same thread block
    // will go from 3.4e-4 to 3.2e-4 with atomicAdds (what? weird.., so graph coloring may not
    // help..)
    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, shared_res);
}  // end of add_residual_gpu kernel
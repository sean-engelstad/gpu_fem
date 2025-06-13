
#include "cuda_utils.h"

// base class methods to launch kernel depending on how many elements per block
// may override these in some base classes

// add_residual kernel
// -----------------------

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void add_energy_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn, 
                               const Vec<int32_t> vars_conn, const Vec<T> xpts, const Vec<T> vars,
                               Vec<Data> physData, T *glob_U) {

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

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    T Uelem_quadpt = 0.0;
    int iquad = threadIdx.y;
    ElemGroup::template add_element_quadpt_energy<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], Uelem_quadpt);

    __SHARED__ T block_sum;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        block_sum = 0.0;
    }
    __syncthreads();

    // first add each U_quadpt contribution from thread to full block
    if (active_thread) {
        atomicAdd(&block_sum, Uelem_quadpt);
    }
    __syncthreads();

    // then add atomicAdd once from block to global energy
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicAdd(glob_U, block_sum);
    }
}

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void add_residual_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                 const Vec<int32_t> vars_conn, const Vec<T> xpts, const Vec<T> vars,
                                 Vec<Data> physData, Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

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
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    ElemGroup::template add_element_quadpt_residual<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], local_res);

    // warp reduction across quadpts
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_res[idof];
        // warp reduction across 4 threads (need to update how to do this for triangle elements later)
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        local_res[idof] = lane_val;
    }

    if (iquad == 0) {
        res.addElementValuesFromShared(active_thread, 0, 1, Phys::vars_per_node,
                                    Basis::num_nodes, vars_elem_conn,
                                    local_res);
    }
}  // end of add_residual_gpu kernel

// add_jacobian_gpu kernel
// -------------------
template <typename T, class ElemGroup, class Data, int32_t elems_per_block,
          template <typename> class Vec, class Mat>
__GLOBAL__ static void add_jacobian_gpu(int32_t vars_num_nodes, int32_t num_elements,
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> physData, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int iquad = threadIdx.x;
    int ideriv = threadIdx.y;
    int local_elem = threadIdx.z;

    int global_elem = local_elem + blockDim.z * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int num_vars = vars_num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    int nthread_xy = blockDim.x * blockDim.y;
    int thread_xy = threadIdx.y * blockDim.x + threadIdx.x;
    int global_elem_thread = local_thread + blockDim.z * blockIdx.x;

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, thread_xy, nthread_xy, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, thread_xy, nthread_xy, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    // call the device function to get one column of the element stiffness
    // matrix at one quadrature point
    ElemGroup::template add_element_quadpt_jacobian_col_no_resid<Data>(
        active_thread, iquad, ideriv, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], local_mat_col);

    // warp reduction over quadpts for jac
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_mat_col[idof];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);
        local_mat_col[idof] = lane_val;
    }

    if (iquad == 0) {
        int nderiv = blockDim.y;
        int elem_block_row = ideriv / Phys::vars_per_node;
        int elem_inner_row = ideriv % Phys::vars_per_node;
        mat.addElementMatRow(active_thread, elem_block_row, elem_inner_row, global_elem, ideriv, nderiv,
            Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);
    }
}  // end of add_jacobian_gpu

// add_jacobian_gpu kernel
// -------------------
template <typename T, int32_t elems_per_block, class Data, 
          template <typename> class Vec>
__GLOBAL__ static void set_design_variables_gpu(const int32_t num_elements, const Vec<T> design_vars, 
        const Vec<int32_t> elem_components, Vec<Data> physData) {

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    
    const int ndvs_global = design_vars.getSize();
    const int ndvs_per_comp = Data::ndvs_per_comp;
    const int *_elem_components = elem_components.getPtr();
    const T *_design_vars = design_vars.getPtr();
    Data *_phys_data = physData.getPtr();

    // don't load to shared, just read from global and set there
    if (global_elem < num_elements) {
        // get which component the element belongs to
        int icomp = _elem_components[global_elem];

        // if (global_elem < 32)
        //     printf("icomp %d, ndvs_global %d\n", icomp, ndvs_global);

        // one thread per elem
        T loc_dvs[ndvs_per_comp];
        for (int idv = 0; idv < ndvs_per_comp; idv++) {
            loc_dvs[idv] = _design_vars[ndvs_per_comp * icomp + idv];
        }

        _phys_data[global_elem].set_design_variables(loc_dvs);

        // if (global_elem < 32) {
        //     printf("global elem %d, thick = %.4e\n", global_elem, _phys_data[global_elem].thick);
        // }
    }

}  // end of set_design_variables_gpu

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_mass_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                    const Vec<T> xpts, Vec<Data> physData, T *total_mass) {
    // computes total mass of the structure
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int iquad = threadIdx.y;
    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    int nvals = elems_per_block * Quadrature::num_quad_pts;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    T quadpt_mass = 0.0;

    ElemGroup::template compute_element_quadpt_mass<Data>(
        active_thread, iquad, block_xpts[local_elem], block_data[local_elem], &quadpt_mass);

    // printf("quadpt mass %.4e\n", quadpt_mass);

    // warp reduction.. across every group of 32 threads
    // much faster than global atomic add
    T lane_val = quadpt_mass;
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 16);  // add stride 16 in warp
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 8);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 4);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

    if (local_thread % 32 == 0) {
        // add at root of each warp to global
        atomicAdd(total_mass, lane_val);
    }

    // regular atomic add, but much slower
    // atomicAdd(total_mass, quadpt_mass);
}  // end of compute_mass_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_mass_DVsens_kernel(const int32_t num_elements,
                                           const Vec<int32_t> elem_components,
                                           const Vec<int32_t> geo_conn, const Vec<T> xpts,
                                           Vec<Data> physData, Vec<T> dfdx) {
    // computes total mass of the structure
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // iquad is consecutive threads so we can do warp reduction
    int iquad = threadIdx.x;
    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    // const int num_quad_pts = Quadrature::num_quad_pts;
    const int ndvs_per_comp = Data::ndvs_per_comp;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // for this element, quadpt contribution
    T local_dmdx[ndvs_per_comp];
    memset(local_dmdx, 0.0, ndvs_per_comp * sizeof(T));

    ElemGroup::template compute_element_quadpt_dmass_dx<Data>(
        active_thread, iquad, block_xpts[local_elem], block_data[local_elem], &local_dmdx[0]);

    // warp reduction across quadpts
    for (int idv = 0; idv < ndvs_per_comp; idv++) {
        T lane_val = local_dmdx[idv];
        // warp reduction across 4 threads (need to update how to do this for triangle elements later)
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        local_dmdx[idv] = lane_val;
    }

    // add from local to global (no shared intermediate, not necessary here)
    if (iquad == 0) { // every 4th thread
        int icomp = elem_components[global_elem];
        for (int idv = 0; idv < ndvs_per_comp; idv++) {
            atomicAdd(&dfdx[ndvs_per_comp * icomp + idv], local_dmdx[idv]);
        }
    }
}  // end of compute_mass_DVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_max_failure_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData,
                                         const T rhoKS, const T safetyFactor, T *max_failure_index) {
    /* prelim kernel to get maximum failure index, before ksMax can be computed */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_node = Phys::vars_per_node;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T quadpt_fail_index = 0.0;
    ElemGroup::template get_element_quadpt_failure_index<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, quadpt_fail_index
    );

    // recall we need to get max(sigma_i) before ks_max(sigma_i) to prevent overflow,
    //  this kernel does regular max
    
    // warp reduction for max in a warp
    T lane_val = quadpt_fail_index;
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 16));
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 8));
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 4));
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 2));
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 1));

    // atomic max back to global
    if (local_thread % 32 == 0) {
        // does atomicMax work?
        atomicMax(max_failure_index, lane_val);
    }

}  // end of compute_max_failure_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void vis_failure_index_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData,
                                         Vec<T> fail_index) {
    /* prelim kernel to get maximum failure index, before ksMax can be computed */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_node = Phys::vars_per_node;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T quadpt_fail_index = 0.0;
    T rhoKS = 100.0, safetyFactor = 1.0; // for within the quadpt
    ElemGroup::template get_element_quadpt_failure_index<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, quadpt_fail_index
    );

    // recall we need to get max(sigma_i) before ks_max(sigma_i) to prevent overflow,
    //  this kernel does regular max
    
    // warp reduction for within each element the 4 quadpts
    T lane_val = quadpt_fail_index;
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 2));
    lane_val = max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 1));

    // atomic max back to global
    if (iquad == 0) {
        fail_index[global_elem] = lane_val;
    }

}  // end of compute_max_failure_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void vis_strains_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData,
                                         Vec<T> strains) {
    /* prelim kernel to get maximum failure index, before ksMax can be computed */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T loc_strains[6];
    T rhoKS = 100.0; // for within the quadpt
    ElemGroup::template compute_element_quadpt_strains<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], loc_strains
    );

    // recall we need to get max(sigma_i) before ks_max(sigma_i) to prevent overflow,
    //  this kernel does regular max
    
    // warp reduction for within each element the 4 quadpts
    for (int i = 0; i < 6; i++) {
        T lane_val = loc_strains[i];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += max(lane_val, __shfl_down_sync(0xFFFFFFFF, lane_val, 1));
        loc_strains[i] = lane_val / num_quad_pts;
    }
    

    // atomic max back to global
    if (iquad == 0) {
        for (int i = 0; i < 6; i++) {
            strains[6 * global_elem + i] = loc_strains[i];
        }
    }

}  // end of vis_strains_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void vis_stresses_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData,
                                         Vec<T> stresses) {
    /* prelim kernel to get maximum failure index, before ksMax can be computed */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_node = Phys::vars_per_node;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T loc_stresses[6];
    ElemGroup::template compute_element_quadpt_stresses<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], loc_stresses
    );

    // recall we need to get max(sigma_i) before ks_max(sigma_i) to prevent overflow,
    //  this kernel does regular max
    
    // warp reduction for within each element the 4 quadpts
    for (int i = 0; i < 6; i++) {
        T lane_val = loc_stresses[i];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);
        loc_stresses[i] = lane_val / num_quad_pts;
    }
    

    // atomic max back to global
    if (iquad == 0) {
        for (int i = 0; i < 6; i++) {
            stresses[6 * global_elem + i] = loc_stresses[i];
        }
    }

}  // end of vis_strains_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                         const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                         const Vec<T> vars, Vec<Data> physData,
                                         const T rhoKS, const T safetyFactor, const T max_failure_index, T *ksmax_failure_index) {
    /* kernel to get ksmax, need previous max_failure_index to prevent overflow */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_node = Phys::vars_per_node;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T quadpt_fail_index = 0.0;
    ElemGroup::template get_element_quadpt_failure_index<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, quadpt_fail_index
    );

    // use the global non-smooth max to prevent overflow
    T glob_max = max_failure_index; // non KS max (non-smooth)
    T exp_quadpt_fail_index = active_thread ? exp(rhoKS * (quadpt_fail_index - glob_max)) : 0.0;

    // warp reduction for max in a warp
    T lane_val = exp_quadpt_fail_index;
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 16);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 8);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 4);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
    lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

    // atomic add back to global
    if (local_thread % 32 == 0) {
        atomicAdd(ksmax_failure_index, lane_val);
    }

}  // end of compute_ksfailure_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_DVsens_kernel(const int32_t num_elements,
                                                const Vec<int32_t> elem_components,
                                                const Vec<int32_t> geo_conn,
                                                const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                                const Vec<T> vars, Vec<Data> physData, 
                                                const T rhoKS, const T safetyFactor, const T max_fail, const T sumexp_kfail, 
                                                Vec<T> dfdx) {

    /* compute dKdx/dx partial derivatives (no indirect state variable sens included) */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // (iquad, ielem) in thraed block chosen so that iquad = 0,1,2,3 on single elem
    // are consectuve threads on the GPU
    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int ndvs_per_comp = Data::ndvs_per_comp;
    // const int vars_per_node = Phys::vars_per_node;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T quadpt_fail_index = 0.0;
    ElemGroup::template get_element_quadpt_failure_index<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, quadpt_fail_index
    );

    // now compute sensitivities
    T df_dksfail_elem = exp(rhoKS * (quadpt_fail_index - max_fail)) / sumexp_kfail;

    // now backprop that through to df/dxe for each element
    T quadpt_dv_sens[ndvs_per_comp];
    memset(quadpt_dv_sens, 0.0, ndvs_per_comp * sizeof(T));

    ElemGroup::template compute_element_quadpt_failure_dv_sens<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, df_dksfail_elem, quadpt_dv_sens
    );

    // warp reduction across quadpts
    for (int idv = 0; idv < ndvs_per_comp; idv++) {
        T lane_val = quadpt_dv_sens[idv];
        // warp reduction across 4 threads (need to update how to do this for triangle elements later)
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        quadpt_dv_sens[idv] = lane_val;
    }

    if (iquad == 0) {
        int icomp = elem_components[global_elem];
        for (int idv = 0; idv < ndvs_per_comp; idv++) {
            atomicAdd(&dfdx[ndvs_per_comp * icomp + idv], quadpt_dv_sens[idv]);
        }
    }

}  // end of compute_ksfailure_DVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_ksfailure_SVsens_kernel(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                                const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                                const Vec<T> vars, Vec<Data> physData, T rhoKS, const T safetyFactor, 
                                                T max_fail, T sumexp_kfail, Vec<T> dfdu) {
    /* compute SV sens */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // (iquad, ielem) in thraed block chosen so that iquad = 0,1,2,3 on single elem
    // are consectuve threads on the GPU
    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
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
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute the local element quadpt failure index
    T quadpt_fail_index = 0.0;
    ElemGroup::template get_element_quadpt_failure_index<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, quadpt_fail_index
    );

    // now compute sensitivities
    T df_dksfail_elem = exp(rhoKS * (quadpt_fail_index - max_fail)) / sumexp_kfail;

    // now backprop that through to df/dxe for each element
    T quadpt_du_sens[vars_per_elem];
    memset(quadpt_du_sens, 0.0, vars_per_elem * sizeof(T));

    ElemGroup::template compute_element_quadpt_failure_sv_sens<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], rhoKS, safetyFactor, df_dksfail_elem, quadpt_du_sens
    );
    
    // warp reduction across quadpts
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = quadpt_du_sens[idof];
        // warp reduction across 4 threads (need to update how to do this for triangle elements later)
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        quadpt_du_sens[idof] = lane_val;
    }

    if (iquad == 0) {
        dfdu.addElementValuesFromShared(active_thread, 0, 1, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn,
                                   quadpt_du_sens);
    }

}  // end of compute_ksfailure_SVsens_kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_adjResProduct_kernel(const int32_t num_elements,
                                                const Vec<int32_t> elem_components,
                                                const Vec<int32_t> geo_conn,
                                                const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                                const Vec<T> vars, Vec<Data> physData,
                                                const Vec<T> psi, Vec<T> dfdx) {
    /* adjoint residual product psi^T dR/dx */
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // (iquad, ielem) in thraed block chosen so that iquad = 0,1,2,3 on single elem
    // are consectuve threads on the GPU
    int local_elem = threadIdx.y;
    int iquad = threadIdx.x;
    int global_elem = local_elem + blockDim.y * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int ndvs_per_comp = Data::ndvs_per_comp;
    // const int vars_per_node = Phys::vars_per_node;
    // const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_psi[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);
    psi.copyElemValuesToShared(active_thread, iquad, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_psi[local_elem][0]);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.x;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // compute element quadpt psi_e^T dR_e/dx adjoint res product
    T quadpt_dv_sens[ndvs_per_comp];
    memset(quadpt_dv_sens, 0.0, ndvs_per_comp * sizeof(T));

    ElemGroup::template compute_element_quadpt_adj_res_product<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], block_psi[local_elem], quadpt_dv_sens
    );

    // warp reduction across quadpts
    for (int idv = 0; idv < ndvs_per_comp; idv++) {
        T lane_val = quadpt_dv_sens[idv];
        // warp reduction across 4 threads (need to update how to do this for triangle elements later)
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        quadpt_dv_sens[idv] = lane_val;
    }

    if (iquad == 0) {
        int icomp = elem_components[global_elem];
        for (int idv = 0; idv < ndvs_per_comp; idv++) {
            atomicAdd(&dfdx[ndvs_per_comp * icomp + idv], quadpt_dv_sens[idv]);
        }
    }
}  // end of compute_adjResProduct_kernel
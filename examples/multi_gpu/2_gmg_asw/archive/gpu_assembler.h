#pragma once

#include <cstring>

#include "chrono"
#include "cuda_utils.h"
#include "mesh/TACSMeshLoader.h"

#ifdef USE_GPU
#include "gpu_assembler.cuh"
#endif  // USE_GPU

// linear algebra formats
#include "gpumat.h"
#include "gpuvec.h"
#include "linalg/vec.h"
#include "multigpu_context.h"
#include "optimization/analysis_function.h"

template <typename ElemGroup, typename T_, class Partitioner, typename Basis_, typename Phys_>
class GPUElementAssembler {
    // assembler for a single element type with multiple GPUs

   public:
    using T = T_;
    using Basis = Basis_;
    using Geo = typename Basis_::Geo;
    using Phys = Phys_;
    using Data = typename Phys::Data;
    using Vec = GPUVec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;
    using DerivedAssembler = ElemGroup;
    // TODO : add back functions later..
    // using MyFunction = AnalysisFunction<T, Vec>;

    static constexpr int32_t geo_nodes_per_elem = Geo::num_nodes;
    static constexpr int32_t vars_nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t spatial_dim = Geo::spatial_dim;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

    // function declarations (to make easier to use)
    // ------------------------
    GPUElementAssembler() = default;  // for pointers

    GPUElementAssembler(MultiGPUContext *ctx_, const Partitioner *part_, int32_t num_nodes_,
                        int32_t num_elements_, HostVec<T> &xpts, HostVec<int> &bcs,
                        HostVec<Data> &compData, int32_t num_components_ = 1,
                        HostVec<int> elem_component = HostVec<int>(1))
        : ctx(ctx_),
          part(part_),
          cublasHandles(ctx_->cublasHandles),
          streams(ctx_->streams),
          ngpus(part_->ngpus),
          num_nodes(num_nodes_),
          num_elements(num_elements_),
          num_components(num_components_) {
        // main constructor

        //         int32_t num_vars = get_num_vars();
        //         this->vars = Vec<T>(num_vars);
        //         this->accel = Vec<T>(num_vars);

        //         // on host (TODO : if need to deep copy entries to device?)
        //         // TODO : should probably do factorization explicitly instead of
        //         // implicitly upon construction
        //         bsr_data = BsrData(num_elements, num_vars_nodes, Basis::num_nodes,
        //         Phys::vars_per_node,
        //                            vars_conn.getPtr());

        //         int ndvs = get_num_dvs();

        // #ifdef USE_GPU

        //         // convert everything to device vecs
        //         this->geo_conn = geo_conn.createDeviceVec();
        //         this->vars_conn = vars_conn.createDeviceVec();
        //         this->xpts = xpts.createDeviceVec();
        //         this->bcs = bcs.createDeviceVec();
        //         this->compData =
        //             compData.createDeviceVec(false);  // false means it just does copy no malloc
        //         this->elem_components = elem_components.createDeviceVec();
        //         this->dvs = DeviceVec<T>(ndvs);

        // #else  // not USE_GPU

        //         // on host just copy normally
        //         this->geo_conn = geo_conn;
        //         this->vars_conn = vars_conn;
        //         this->xpts = xpts;
        //         this->bcs = bcs;
        //         this->compData = compData;
        //         this->elem_components = elem_components;
        //         this->dvs = HostVec<T>(ndvs);

#endif  // end of USE_GPU or not USE_GPU check
    }

    void moveBsrDataToDevice() { this->bsr_data = bsr_data.createDeviceBsrData(); }
    void moveBsrDataToHost() { this->bsr_data = bsr_data.createHostBsrData(); }

    static DerivedAssembler createFromBDF(int ngpus_, TACSMeshLoader &mesh_loader,
                                          Data single_data) {
        // TODO : make the partition as well..

        // int vars_per_node = Phys::vars_per_node;  // input

        // int num_nodes, num_elements, num_bcs, num_components;
        // int *elem_conn, *bcs, *elem_components;
        // T *xpts;

        // mesh_loader.getAssemblerCreatorData(vars_per_node, num_nodes, num_elements, num_bcs,
        //                                     num_components, elem_conn, bcs, elem_components,
        //                                     xpts);

        // // make HostVec objects here for Assembler
        // HostVec<int> elem_conn_vec(vars_nodes_per_elem * num_elements, elem_conn);
        // HostVec<int> bcs_vec(num_bcs, bcs);
        // HostVec<int> elem_components_vec(num_elements, elem_components);
        // HostVec<T> xpts_vec(spatial_dim * num_nodes, xpts);
        // HostVec<Data> compData_vec(num_components, single_data);

        // // printf("num_components = %d\n", num_components);

        // // call base constructor
        // return ElemGroup(num_nodes, num_nodes, num_elements, elem_conn_vec, elem_conn_vec,
        // xpts_vec,
        //                  bcs_vec, compData_vec, num_components, elem_components_vec);
    }

    __HOST__ void apply_bcs(Vec<T> &vec, bool can_print = false) { vec.apply_bcs(bcs); }

    void apply_bcs(Mat &mat, bool can_print = false) { mat.apply_bcs(bcs); }

    DeviceVec<T> createGPUVec(T *data = nullptr, bool randomize = false, bool can_print = false) {
        // TODO : change this to GPUvec

        // auto h_vec = createVarsHostVec(data, randomize);
        // auto d_vec = h_vec.createDeviceVec(true, can_print);
        // return d_vec;
    }

    void set_variables(Vec<T> &newVars) { newVars.copyValuesTo(this->vars); }
    void set_component_data(Vec<Data> &newCompData) { newCompData.copyValuesTo(compData); }

    template <int elems_per_block = 8>
    void add_residual(Vec *res) {
        res->zero();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = loc_nnodes[g];
            int loc_nelems = loc_num_elements[g];

            T *loc_elem_comps = loc_elem_components[g];
            int *loc_row_conn = row_loc_elem_conn[g];
            int *loc_col_conn = cols_loc_elem_conn[g];
            T *loc_xpts_ptr = loc_xpts[g];
            T *loc_vars_ptr = loc_vars[g];
            T *loc_comp_data_ptr = loc_comp_data[g];
            T *loc_res_ptr = loc_res[g];

            dim3 block(num_quad_pts, elems_per_block);
            int nblocks = (loc_nelems + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);

            k_add_multigpu_residual_fast<T, elems_per_block, ElemGroup>
                <<<grid, block, 0, streams[g]>>>(loc_num_nodes, loc_nelems, loc_elem_comps,
                                                 loc_row_conn, loc_col_conn, loc_xpts_ptr,
                                                 loc_vars_ptr, loc_comp_data_ptr, loc_res_ptr);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();
    }

    template <int elems_per_block = 1>
    void add_jacobian(Vec *res, Mat *mat, bool can_print = false) {
        mat->zeroValues();

        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);

        dim3 block(num_quad_pts, cols_per_elem, elems_per_block);
        int elem_cols_per_block = cols_per_elem * elems_per_block;

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = loc_nnodes[g];
            int loc_nelems = loc_num_elements[g];

            T *loc_elem_comps = loc_elem_components[g];
            T *loc_xpts_ptr = loc_xpts[g];
            T *loc_vars_ptr = loc_vars[g];
            T *loc_comp_data_ptr = loc_comp_data[g];

            int *loc_elem_ind_map = mat->getLocalElemIndMap(g);
            T *loc_mat_vals = mat->getLocalVals(g);

            int nblocks =
                (loc_nelems * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;

            dim3 grid(nblocks);

            k_add_multigpu_jacobian_fast<T, elems_per_block, ElemGroup>
                <<<grid, block, 0, streams[g]>>>(loc_num_nodes, loc_nelems, cols_per_elem,
                                                 loc_elem_comps, loc_xpts_ptr, loc_vars_ptr,
                                                 loc_comp_data_ptr, loc_elem_ind_map, loc_mat_vals);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();
    }

    // util functions
    BsrData &getBsrData() { return bsr_data; }
    Vec<T> getXpts() { return xpts; }
    Vec<T> getVars() { return vars; }
    Vec<int> getBCs() { return bcs; }
    Vec<int> getConn() { return vars_conn; }
    Vec<Data> getCompData() { return compData; }
    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    int get_num_dvs() { return num_components * Data::ndvs_per_comp; }
    int get_num_components() { return num_components; }
    int get_num_nodes() { return num_vars_nodes; }
    int get_num_elements() { return num_elements; }
    void get_elem_components(Vec<int> &elem_comp_out) {
        elem_components.copyValuesTo(elem_comp_out);
    }
    Vec<int> getElemComponents() { return elem_components; }

    HostVec<T> createVarsHostVec(T *data, bool randomize);
    void setBsrData(BsrData new_bsr_data) { this->bsr_data = new_bsr_data; }

    // void permuteVec(Vec<T> vec) {
    //     if (bsr_data.perm) {
    //         vec.permuteData(bsr_data.block_dim, bsr_data.perm);
    //     } else {
    //         printf("bsr data has no iperm pointer\n");
    //     }
    // }
    // void invPermuteVec(Vec<T> vec) {
    //     if (bsr_data.iperm) {
    //         vec.permuteData(bsr_data.block_dim, bsr_data.iperm);
    //     } else {
    //         printf("bsr data has no iperm pointer\n");
    //     }
    // }

    void free() {
        if (is_free) return;
        is_free = true;  // now it's freed

        geo_conn.free();
        vars_conn.free();
        bcs.free();
        elem_components.free();
        xpts.free();
        vars.free();
        compData.free();
        bsr_data.free();
        dvs.free();
    }

   protected:
    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;

    int ngpus = 0;

    bool is_free = false;
    int32_t num_nodes, num_elements, num_components;

    Vec<int32_t> geo_conn, vars_conn;
    Vec<int> bcs, elem_components;
    Vec<T> xpts, vars;
    Vec<Data> compData;
    BsrData bsr_data;
};  // end of ElementAssembler class declaration

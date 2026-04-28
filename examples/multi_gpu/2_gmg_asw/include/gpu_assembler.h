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

    GPUElementAssembler(MultiGPUContext *ctx_, int32_t num_nodes_, int32_t num_elements_,
                        HostVec<int> *h_elem_conn_, HostVec<T> *xpts, HostVec<int> *bcs_,
                        HostVec<Data> *compData, int32_t num_components_ = 1,
                        HostVec<int> *elem_component = new HostVec<int>(1))
        : ctx(ctx_),
          cublasHandles(ctx_->cublasHandles),
          streams(ctx_->streams),
          bcs(bcs_),
          ngpus(ctx_->ngpus),
          num_nodes(num_nodes_),
          num_elements(num_elements_),
          num_components(num_components_) {
        // main constructor

        // make the multi-GPU domain decomp partitioner
        part = new Partitioner(ngpus, num_nodes, num_elements, vars_nodes_per_elem,
                               h_elem_conn->getPtr());

        // get host versions for output
        h_xpts = xpts;
        h_elem_conn = h_elem_conn_;
        h_elem_components = elem_component;
        h_compData = compData;
        h_vars = new HostVec<T>(num_nodes * vars_per_node);

        d_xpts = new Vec(ctx, part, spatial_dim);
        d_vars = new Vec(ctx, part, vars_per_node);
        d_res = new Vec(ctx, part, vars_per_node);

        // set xpts from host
        d_xpts->setValuesFromHost(xpts->getPtr());

        h_loc_elem_components = new int *[ngpus];
        d_loc_elem_components = new int *[ngpus];
        int *elem_comp_ptr = elem_component->getPtr();
        for (int g = 0; g < ngpus; g++) {
            int local_nelems = part->getLocalNumElements(g);
            h_loc_elem_components[g] = new int[local_nelems];
            int start_elem = part->getStartElem(g);
            for (int le = 0; le < local_nelems; le++) {
                int e = le + start_elem;
                h_loc_elem_components[g][le] = elem_comp_ptr[e];
            }

            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMalloc(&d_loc_elem_components[g], local_nelems * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_elem_components[g], h_loc_elem_components[g],
                                  local_nelems * sizeof(int), cudaMemcpyHostToDevice));
        }
        ctx->sync();

        // copy compData pointer to all GPUs (not subdivided, since usually #components small this
        // is fine)
        Data *h_comp_data_ptr = compData->getPtr();
        d_loc_comp_data = new Data *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMalloc(&d_loc_comp_data[g], num_components * sizeof(Data)));
            CHECK_CUDA(cudaMemcpy(d_loc_comp_data[g], h_comp_data_ptr[g],
                                  num_components * sizeof(Data), cudaMemcpyHostToDevice));
        }
        ctx->sync();

        allocate_reduced_bcs();
#endif  // end of USE_GPU or not USE_GPU check
    }

    Partitioner *getPartitioner() { return part; }
    // void moveBsrDataToDevice() { this->bsr_data = bsr_data.createDeviceBsrData(); }
    // void moveBsrDataToHost() { this->bsr_data = bsr_data.createHostBsrData(); }

    static DerivedAssembler createFromBDF(MultiGPUContext *ctx_, TACSMeshLoader &mesh_loader,
                                          Data single_data) {
        int vars_per_node = Phys::vars_per_node;  // input

        int num_nodes, num_elements, num_bcs, num_components;
        int *elem_conn, *bcs, *elem_components;
        T *xpts;

        mesh_loader.getAssemblerCreatorData(vars_per_node, num_nodes, num_elements, num_bcs,
                                            num_components, elem_conn, bcs, elem_components, xpts);

        // make HostVec objects here for Assembler
        HostVec<int> elem_conn_vec(vars_nodes_per_elem * num_elements, elem_conn);
        HostVec<int> bcs_vec(num_bcs, bcs);
        HostVec<int> elem_components_vec(num_elements, elem_components);
        HostVec<T> xpts_vec(spatial_dim * num_nodes, xpts);
        HostVec<Data> compData_vec(num_components, single_data);

        // printf("num_components = %d\n", num_components);

        // call base constructor
        return ElemGroup(ctx_, num_nodes, num_nodes, num_elements, &elem_conn_vec, &xpts_vec,
                         &bcs_vec, &compData_vec, num_components, &elem_components_vec);
    }

    void apply_bcs(Vec *vec) { vec->apply_bcs(n_owned_bcs, d_owned_bcs, n_local_bcs, d_local_bcs); }

    void apply_bcs(Mat *mat) { mat->apply_bcs(n_owned_bcs, d_owned_bcs, n_local_bcs, d_local_bcs); }

    DeviceVec<T> createGPUVec(T *h_data = nullptr) {
        Vec *d_vec = Vec(ctx, part, vars_per_node);
        d_vec->setValuesFromHost(h_data);
        return d_vec;
    }

    void set_variables(Vec<T> &newVars) { newVars.copyValuesTo(this->vars); }
    void set_component_data(Vec<Data> &newCompData) { newCompData.copyValuesTo(compData); }

    template <int elems_per_block = 8>
    void add_residual(Vec *res) {
        res->zero();
        xpts->expandToLocal();
        vars->expandToLocal();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_xpts->getExpandedNodes(g);
            int loc_nelems = part->getLocalNumElements(g);

            T *loc_elem_comps = loc_elem_components[g];
            int *loc_row_conn = mat->getRowRedElemConn(g);
            int *loc_col_conn = mat->getColRedElemConn(g);

            // local expanded vec on input, reduced or only owned vec on output
            T *loc_xpts_ptr = d_xpts->getLocalPtr(g);
            T *loc_vars_ptr = d_vars->getLocalPtr(g);
            T *loc_res_ptr = d_res->getRedPtr(g);
            Data *loc_comp_data_ptr = d_loc_comp_data[g];

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
    void add_jacobian(Mat *mat) {
        mat->zeroValues();
        xpts->expandToLocal();
        vars->expandToLocal();

        int cols_per_elem = 24;  // for 1st order element

        dim3 block(num_quad_pts, cols_per_elem, elems_per_block);
        int elem_cols_per_block = cols_per_elem * elems_per_block;

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_xpts->getExpandedNodes(g);
            int loc_nelems = part->getLocalNumElements(g);

            T *loc_xpts_ptr = d_xpts->getLocalPtr(g);
            T *loc_vars_ptr = d_vars->getLocalPtr(g);
            int *loc_elem_comps = loc_elem_components[g];
            Data *loc_comp_data_ptr = d_loc_comp_data[g];

            // use this connectivity for extracting xpts, vars
            int *loc_elem_conn_ptr = mat->getColRedElemConn(g);
            int *loc_elem_ind_map = mat->getLocalElemIndMap(g);
            T *loc_mat_vals = mat->getLocalVals(g);

            int nblocks =
                (loc_nelems * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;

            dim3 grid(nblocks);

            k_add_multigpu_jacobian_fast<T, elems_per_block, ElemGroup>
                <<<grid, block, 0, streams[g]>>>(
                    loc_num_nodes, loc_nelems, cols_per_elem, loc_elem_comps, loc_elem_conn_ptr,
                    loc_xpts_ptr, loc_vars_ptr, loc_comp_data_ptr, loc_elem_ind_map, loc_mat_vals);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();
    }

    // util functions
    // BsrData &getBsrData() { return bsr_data; }
    HostVec<T> *getXpts() { return h_xpts; }
    HostVec<T> *getVars() { return h_vars; }
    // Vec<int> getBCs() { return bcs; }
    HostVec<int> *getConn() { return h_elem_conn; }
    HostVec<Data> *getCompData() { return h_compData; }
    HostVec<int> *getElemComponents() { return h_elem_components; }
    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    int get_num_dvs() { return num_components * Data::ndvs_per_comp; }
    int get_num_components() { return num_components; }
    int get_num_nodes() { return num_vars_nodes; }
    int get_num_elements() { return num_elements; }
    // void get_elem_components(HostVec<int> &elem_comp_out) {
    //     elem_components.copyValuesTo(elem_comp_out);
    // }

    // HostVec<T> createVarsHostVec(T *data, bool randomize);
    // void setBsrData(BsrData new_bsr_data) { this->bsr_data = new_bsr_data; }

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

    void allocate_reduced_bcs() {
        // divide bcs into row and col local bcs (row bcs can be used on owned vec, col on expanded
        // vec with ghost aka local vec)
        nbcs = bcs->getSize();
        int *h_bcs_ptr = bcs->getPtr();
        h_owned_bcs = new int *[ngpus];
        h_local_bcs = new int *[ngpus];
        d_owned_bcs = new int *[ngpus];
        d_local_bcs = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            int owned_nnodes = part->getNumOwnedNodes(g);
            int *h_owned_nodes = part->getOwnedNodesPtr(g);
            int local_nnodes = part->getNumLocalNodes(g);
            int *h_local_nodes = part->getLocalNodesPtr(g);

            bool *is_owned = new bool[num_nodes];
            memset(is_owned, false, num_nodes * sizeof(bool));
            int *owned_red_ind = new int[num_nodes];
            memset(owned_red_ind, 0, num_nodes * sizeof(int));
            for (int n = 0; n < owned_nnodes; n++) {
                int node = h_owned_nodes[n];
                is_owned[node] = true;
                owned_red_ind[node] = n;
            }
            std::vector<int> g_owned_bcs;
            for (int i = 0; i < nbcs; i++) {
                int bc_dof = h_bcs_ptr[i];
                int bc_node = bc_dof / vars_per_node;
                int bc_dim = bc_dof % vars_per_node;

                if (is_owned[bc_node]) {
                    owned_node = owned_red_ind[bc_node];
                    owned_dof = vars_per_node * owned_node + bc_dim;
                    g_owned_bcs.push_back(owned_dof);
                }
            }

            bool *is_local = new bool[num_nodes];
            memset(is_local, false, num_nodes * sizeof(bool));
            int *local_red_ind = new int[num_nodes];
            memset(local_red_ind, 0, num_nodes * sizeof(int));
            for (int n = 0; n < local_nnodes; n++) {
                int node = h_local_nodes[n];
                is_local[node] = true;
                local_red_ind[node] = n;
            }
            std::vector<int> g_local_bcs;
            for (int i = 0; i < nbcs; i++) {
                int bc_dof = h_bcs_ptr[i];
                int bc_node = bc_dof / vars_per_node;
                int bc_dim = bc_dof % vars_per_node;

                if (is_local[bc_node]) {
                    local_node = local_red_ind[bc_node];
                    local_dof = vars_per_node * local_node + bc_dim;
                    g_local_bcs.push_back(local_dof);
                }
            }

            // now assign owned and local bcs to host + device
            h_owned_bcs[g] = new int[g_owned_bcs.size()];
            for (int i = 0; i < g_owned_bcs.size(); i++) {
                int bc = g_owned_bcs[i];
                h_owned_bcs[g][i] = bc;
            }

            h_local_bcs[g] = new int[g_local_bcs.size()];
            for (int i = 0; i < g_local_bcs.size(); i++) {
                int bc = g_local_bcs[i];
                h_local_bcs[g][i] = bc;
            }

            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMalloc(&d_owned_bcs[g], g_owned_bcs.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_owned_bcs[g], h_owned_bcs[g], g_owned_bcs.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_bcs[g], g_local_bcs.size() * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_bcs[g], h_local_bcs[g], g_local_bcs.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }

        ctx->sync();
    }

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

    HostVec<int> *bcs;
    int nbcs;
    int **h_owned_bcs, **d_owned_bcs;
    int **h_local_bcs, **d_local_bcs;
    int *n_owned_bcs, *n_local_bcs;

    HostVec<int> *h_elem_conn, *h_elem_components;
    HostVec<T> *h_xpts, *h_vars;
    HostVec<Data> *h_compData;

    Vec *d_xpts, *d_vars, *d_res;
    int **h_loc_elem_components, **d_loc_elem_components;
    Data **d_loc_comp_data;
};  // end of ElementAssembler class declaration

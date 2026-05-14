#pragma once

#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "assembler/gpu_assembler.h"
#include "cuda_utils.h"
#include "element/shell/_shell.cuh"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/multigpu_context.h"

template <typename T, class Assembler_, class Partition, class IEVSplit>
class MultiGPUBDDC_LUSolver {
    using Assembler = Assembler_;
    using Director = typename Assembler::Director;
    using Basis = typename Assembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename Assembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    using VEC = GPUvec<T, PARTITION>;
    using MAT = GPUbsrmat<T, PARTITION>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    MultiGPUBDDC_LUSolver(MultiGPUContext *ctx_, Partition *part_, Assembler *assembler_, Mat *mat_,
                          const IEVSplit &split_) ctx(ctx_),
        part(part_), mat(mat_), cublasHandles(ctx_->cublasHandles),
        cusparseHandles(ctx_->cusparseHandles), streams(ctx_->streams) {
        ngpus = ctx->ngpus;
        num_elements = assembler.get_num_elements();
        num_nodes = assembler.get_num_nodes();
        N = num_nodes * vars_per_node;

        block_dim = mat->getBlockDim();
        block_dim2 = block_dim * block_dim;

        d_xpts = assembler->getDeviceXpts();
        d_vars = assembler->getDeviceVars();
        d_loc_elem_components = assembler->getDeviceElemComponents();
        d_compData = assembler->getDeviceCompData();
        assembler->getLocalDeviceBCs(n_owned_bcs, n_local_bcs, d_owned_bcs, d_local_bcs);

        import_splitting();
        build_IE_I_V_maps();
        build_IEV_sparsity();
        make_iev_bcs();  // TBD
        allocate_vectors();
    }

    void free() {
        // TBD
    }
    int getLambdaSize() const { return ngam * block_dim; }

    void update_after_assembly(Vec *vars) {
        vars->copyTo(d_vars);
        assemble_subdomains();
        factorIEsubdomains();
        factorIsubdomains();
        assemble_coarse_problem();
        factorCoarseVertex();
    }

    void mat_vec(Vec *gam_in, Vec *gam_out) {
        gam_out->zeroValues();

        addVecGamtoIEV(gam_in, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        addVecIEtoI(f_IE, f_I, 1.0, 0.0);

        solveSubdomainI(f_I, u_I);

        addVecItoIE(u_I, u_IE, 1.0, 0.0);
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 1.0, f_IEV);

        addVecIEVtoGam(f_IEV, gam_out, 1.0, 0.0);
    }

    bool solve(Vec *gam_rhs, Vec *gam, bool check_conv = false) {
        constexpr bool SCALED = true;

        addVecGamtoIEV<SCALED>(gam_rhs, f_IEV, 1.0, 0.0);

        addVecIEVtoVc<SCALED>(f_IEV, f_V, 1.0, 0.0);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        addVecIEtoIEV(f_IE, f_IEV, -1.0, 1.0);
        zeroInteriorIE(f_IE);

        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);

        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);
        solveCoarse(f_V, u_V);

        addVecVctoIEV(u_V, temp_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, temp_IEV, -1.0, 0.0, f_IEV);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);

        u_IE->zeroAll();
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 1.0);
        addVecVctoIEV<SCALED>(u_V, u_IEV, 1.0, 1.0);

        addVecIEVtoGam<SCALED>(u_IEV, gam, 1.0, 0.0);

        return false;
    }

   private:
    static int *copy_vec(const std::vector<int> &v) {
        if (v.empty()) return nullptr;

        int *out = new int[v.size()];
        std::memcpy(out, v.data(), v.size() * sizeof(int));
        return out;
    }

    void import_splitting() {
        // sgpu = single GPU
        sgpu_num_subdomains = split.num_subdomains;

        // Original/single-GPU splitting copied directly from split
        sgpu_I_nnodes = split.I_nnodes;
        sgpu_IE_nnodes = split.IE_nnodes;
        sgpu_IEV_nnodes = split.IEV_nnodes;
        sgpu_Vc_nnodes = split.Vc_nnodes;
        sgpu_V_nnodes = split.V_nnodes;
        sgpu_lam_nnodes = split.lam_nnodes;

        sgpu_elem_sd_ind = copy_vec(split.elem_sd_ind);
        sgpu_node_class_ind = copy_vec(split.node_class_ind);
        sgpu_node_nsd = copy_vec(split.node_nsd);

        sgpu_IEV_sd_ptr = copy_vec(split.IEV_sd_ptr);
        sgpu_IEV_sd_ind = copy_vec(split.IEV_sd_ind);
        sgpu_IEV_nodes = copy_vec(split.IEV_nodes);
        sgpu_IEV_elem_conn = copy_vec(split.IEV_elem_conn);

        // d_sgpu_IEV_elem_conn =
        //     HostVec<int>(num_elements * nodes_per_elem, sgpu_IEV_elem_conn).createDeviceVec();

        // make partition for IEV conn from IEV splitting (uses those subdomains to assign labels)
        part_IEV =
            new Partition(ngpus, sgpu_IEV_nnodes, num_elements, nodes_per_elem, sgpu_IEV_elem_conn,
                          part->num_components, part->h_elem_components, split);

        create_multigpu_splitting();

        d_xpts_IEV = new Vec(ctx, part_IEV, block_dim);
        d_vars_IEV = new Vec(ctx, part_IEV, block_dim);
        mat_IEV = new Mat(ctx, part_IEV, block_dim);
    }

    void create_multigpu_splitting() {
        // determine which subdomains belong to which GPUs from the partition..

        subdomain_gpu_ind = new int[sgpu_num_subdomains];
        memset(subdomain_gpu_ind, -1, sgpu_num_subdomains * sizeof(int));
        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int gpu = part->find_owned_gpu_from_elem(e);
            subdomain_gpu_ind[s] = gpu;
        }

        // get local nnodes and nelems on each GPU
        local_nnodes = new int[ngpus];
        local_nelems = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            local_nnodes[g] = part->local_nnodes[g];
            local_nelems[g] = part->local_nelems[g];
        }

        // compute glob to local elem map
        int *elem_ctr = new int[ngpus];
        memset(elem_ctr, 0, ngpus * sizeof(int));
        glob_loc_elem_map = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            glob_loc_elem_map[g] = new int[local_nelems[g]];
        }
        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int g = subdomain_gpu_ind[s];
            int ered = elem_ctr[g]++;
            glob_loc_elem_map[g][e] = ered;
        }

        // compute elem_sd_ind on each local GPU
        elem_sd_ind = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            elem_sd_ind[g] = new int[local_nelems[g]];
        }
        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int g = subdomain_gpu_ind[s];
            int ered = glob_loc_elem_map[g][e];
            elem_sd_ind[g][ered] = s;
        }

        // get num subdomains in each GPU
        num_subdomains = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> subdomains;
            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int s = elem_sd_ind[g][ered];
                subdomains.insert(s);
            }
            num_subdomains[g] = subdomains.size();
        }
        sd_cts = new int[ngpus];
        memset(sd_cts, 0, ngpus * sizeof(int));
        red_subdomains = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            red_subdomains[g] = new int[num_subdomains[g]];
            std::unordered_set<int> subdomains;
            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int s = elem_sd_ind[g][ered];
                subdomains.insert(s);
            }
            std::vector<int> sd_vec(subdomains.begin(), subdomains.end());
            for (int sred = 0; sred < sd_vec.size(); sred++) {
                red_subdomains[g][sred] = s;
            }
        }

        // classify the nodes
        node_class_ind = new int *[ngpus];
        // node_nsd = new int *[ngpus];
        I_nnodes = new int[ngpus];
        IE_nnodes = new int[ngpus];
        IEV_nnodes = new int[ngpus];
        Vc_nnodes = new int[ngpus];
        V_nnodes = new int[ngpus];
        lam_nnodes = new int[ngpus];
        ngam = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            node_class_ind[g] = new int[local_nnodes[g]];
            node_nsd[g] = new int[local_nnodes[g]];
            I_nnodes[g] = 0;
            IE_nnodes[g] = 0;
            IEV_nnodes[g] = 0;
            Vc_nnodes[g] = 0;
            V_nnodes[g] = 0;
            lam_nnodes[g] = 0;

            for (int l = 0; l < local_nnodes[g]; l++) {
                int n = part->h_local_nodes[g][l];
                int node_class = sgpu_node_class_ind[n];
                node_class_ind[g][l] = node_class;
                if (node_class == INTERIOR) {
                    I_nnodes[g] += 1;
                    IE_nnodes[g] += 1;
                    IEV_nnodes[g] += 1;
                } else if (node_class == EDGE) {
                    lam_nnodes[g] += 1;
                    IE_nnodes[g] += nsd;
                    IEV_nnodes[g] += nsd;
                } else {
                    Vc_nnodes[g] += 1;
                    V_nnodes[g] += nsd;
                    IEV_nnodes[g] += nsd;
                }
            }

            ngam[g] = lam_nnodes[g] + Vc_nnodes[g];  // for BDDC E+V basically (full interface)
        }

        // don't think I also need IEV_sd_ptr and IEV_sd_ind (temp arrays and we can just get this
        // from the single gpu ones)

        // so just fill out these two arrays on each local GPU
        IEV_nodes = new int *[ngpus];
        IEV_loc_to_glob = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            IEV_nodes[g] = new int[IEV_nnodes[g]];
            IEV_loc_to_glob[g] = new int[IEV_nnodes[g]];
        }
        int *IEV_cts = new int[ngpus];
        memset(IEV_cts, 0, ngpus * sizeof(int));
        for (int s = 0; s < sgpu_num_subdomains; s++) {
            int gpu = subdomain_gpu_ind[s];
            for (int iev = sgpu_IEV_sd_ptr[s]; iev < sgpu_IEV_sd_ptr[s + 1]; iev++) {
                int n = sgpu_IEV_nodes[iev];
                int iev_red = IEV_cts[gpu]++;
                int n_red = part->global_to_local[g][n];
                IEV_nodes[g][iev_red] = n_red;
                IEV_loc_to_glob[g][iev_red] = iev;
            }
        }

        // then fill out IEV_elem_conn
        IEV_elem_conn = new int *[ngpus];
        elem_loc_to_glob = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            IEV_elem_conn[g] = new int[nodes_per_elem * local_nelems[g]];
            elem_loc_to_glob[g] = new int[local_nelems[g]];
        }
        int *elem_red_cts = new int[ngpus];
        memset(elem_red_cts, 0, ngpus * sizeof(int));
        elem_glob_to_loc = new int[num_elements];
        for (int e = 0; e < num_elements; e++) {
            int g = part->h_elem_assigned_gpu[e];
            int ered = elem_red_cts[g]++;
            elem_glob_to_loc[e] = ered;
            elem_loc_to_glob[g][ered] = e;
        }
        // now fill out the IEV_elem_conn
        for (int g = 0; g < ngpus; g++) {
            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int e = elem_loc_to_glob[g][ered];
                int *sgpu_lnodes = &sgpu_IEV_elem_conn[nodes_per_elem * e];
                int *lnodes = &IEV_elem_conn[g][nodes_per_elem * ered];
                for (int l = 0; l < nodes_per_elem; l++) {
                    int n = sgpu_lnodes[l];
                    int nred = part->global_to_local[g][n];
                    lnodes[l] = nred;
                }
            }
        }
    }
    void build_IE_I_V_maps() {
        // TODO;
        build_Vc_and_gam_maps();
    }
    void build_Vc_and_gam_maps() {}
    void build_IEV_sparsity() {}
    void allocate_vectors() {}
    void clear_host_data() {}
    void assemble_subdomains() {
        addVec_globalToIEV(d_xpts, d_xpts_IEV, 3, 1.0, 0.0);
        addVec_globalToIEV(d_vars, d_vars_IEV, block_dim, 1.0, 0.0);

        add_IEV_jacobian();
    }
    void add_IEV_jacobian() {
        int cols_per_elem = 24;  // for 1st order element

        d_xpts_IEV->expandToLocal();
        d_vars_IEV->expandToLocal();

        dim3 block(num_quad_pts, cols_per_elem, elems_per_block);
        int elem_cols_per_block = cols_per_elem * elems_per_block;

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_xpts_IEV->getExpandedNodes(g);
            int loc_nelems = part->getLocalNumElements(g);

            // can't use Vec objects here since not same number of nodes?
            // or I could create it from IEV_elem_conn partitioners? we'll see
            T *loc_xpts_ptr = d_xpts_IEV->getLocalPtr(g);
            T *loc_vars_ptr = d_vars_IEV->getLocalPtr(g);
            int *loc_elem_comps = d_loc_elem_components[g];
            Data *loc_comp_data_ptr = d_loc_comp_data[g];

            // local element connectivity, used for both rows and columns
            int *loc_elem_conn_ptr = mat_IEV->getLocalElemConn(g);
            int *loc_elem_ind_map = mat_IEV->getLocalElemIndMap(g);
            T *loc_mat_vals = mat_IEV->getLocalVals(g);

            int nblocks =
                (loc_nelems * cols_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;

            dim3 grid(nblocks);

            k_add_multigpu_jacobian_fast<T, elems_per_block, ElemGroup>
                <<<grid, block, 0, streams[g]>>>(
                    loc_num_nodes, loc_nelems, cols_per_elem, loc_elem_comps, loc_elem_conn_ptr,
                    loc_xpts_ptr, loc_vars_ptr, loc_comp_data_ptr, loc_elem_ind_map, loc_mat_vals);

            CHECK_CUDA(cudaGetLastError());
        }
    }
    void assemble_coarse_problem() {
        // done on single GPU
        S_VV->zeroValues();
        copyKmat_IEVtoSvv();
        computeSvvInverseTerm();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    void copyKmat_IEVtoIE() {}
    void copyKmat_IEVtoI() {}
    void copyKmat_IEVtoSvv() {
        // TBD: may also need kmat_IEV on root GPU to ensure copy to S_VV on root GPU (can't be
        // partitioned maybe) or if there's some way I can assemble that part directly into Svv?
    }
    void computeSvvInverseTerm() {
        // TODO: this is problem with using CuDSS, how do we do the inverse term ghost transfer
        // then to root a ton of times? Isn't that really expensive?
    }
    void sparseMatVec(Mat *A, Vec *x, T alpha, T beta, Vec *y) {}
    void factorIEsubdomains() {}
    void factorIsubdomains() {}
    void assemble_coarse_problem() {}
    void factorCoarseVertex() {}
    void solveSubdomainIE(Vec *rhs_in, Vec *sol_out) {}
    void solveSubdomainI(Vec *rhs_in, Vec *sol_out) {}
    void solveCoarse(Vec *rhs_in, Vec *sol_out) {}
    void zeroInteriorIE(Vec *x) {}
    template <bool SCALED = false>
    void addVecIEVtoGam(const Vec *vec_IEV, Vec *vec_gam, T alpha, T beta) {}
    template <bool SCALED = false>
    void addVecGamtoIEV(const Vec *vec_gam, Vec *vec_IEV, T alpha, T beta) {}
    void addVecGamtoIE(const Vec *gam, Vec *vec_IE, T a, T b) {}
    void addVecIEtoGam(const Vec *vec_IE, Vec *gam, T a, T b) {}
    template <bool scaled = false>
    void addVec_globalToIEV(Vec *x_global, Vec *y_iev, int vars_per_node_in, T a, T b) {}
    void addVecIEVtoIE(const Vec *x, Vec *y, T a, T b) {}
    void addVecItoIEV(const Vec *x, Vec *y, T a, T b) {}
    void addVecIEVtoI(const Vec *x, Vec *y, T a, T b) {}
    template <bool scaled = false>
    void addVecIEVtoVc(const Vec *x, Vec *y, T a, T b) {}
    void addVecIEtoIEV(const Vec *x, Vec *y, T a, T b, int vars_per_node = -1) {}
    template <bool scaled = false>
    void addVecVctoIEV(const Vec *x, Vec *y, T a, T b, int vars_per_node = -1) {}
    void addVecItoIE(const Vec *x, Vec *y, T a, T b) {}
    void addVecIEtoI(const Vec *x, Vec *y, T a, T b) {}
    void addVecLamtoIE(const Vec *lam, Vec *vec_IE, T a, T b) {}
    void addVecIEtoLam(const Vec *vec_IE, Vec *lam, T a, T b) {}
    void addGlobalSoln(const Vec *u_IE, const Vec *u_V, Vec *soln) {}

   private:
    int num_elements, num_nodes, N, block_dim, block_dim2;

    Assembler *assembler = nullptr;
    Mat *mat = nullptr;
    Partition *part = nullptr;
    Partition *part_IEV = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;

    Vec *d_xpts, *d_vars;
    int **d_loc_elem_components;
    int *n_owned_bcs, *n_local_bcs;
    int **d_owned_bcs, **d_local_bcs;
    Data **d_loc_comp_data;

    // single GPU subdomains
    int sgpu_num_subdomains;
    int sgpu_I_nnodes, sgpu_IE_nnodes, sgpu_IEV_nnodes, sgpu_Vc_nnodes, sgpu_V_nnodes,
        sgpu_lam_nnodes;
    int *sgpu_elem_sd_ind, *sgpu_node_class_ind, *sgpu_node_nsd;
    int *sgpu_IEV_sd_ptr, *sgpu_IEV_sd_ind, *sgpu_IEV_nodes, *sgpu_IEV_elem_conn;

    // multi GPU subdomains
    int ngpus;
    int *subdomain_gpu_ind;
    int *num_subdomains;
    int *I_nnodes, *IE_nnodes, *IEV_nnodes, *Vc_nnodes, *V_nnodes, *lam_nnodes;
    int **elem_sd_ind, **node_class_ind, **node_nsd;
    int **IEV_nodes, **IEV_elem_conn;

    Vec *d_xpts_IEV, *d_vars_IEV;
    Mat *mat_IEV;
};
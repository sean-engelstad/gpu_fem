#pragma once
#include "_unstruct_utils.h"
#include "_unstructured.cuh"
#include "gpumat.h"
#include "gpuvec.h"
#include "multigpu_context.h"

template <typename T, class Assembler, class Partitioner, class Basis>
class MultiGPUUnstructuredProlongation {
   public:
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;
    using Basis = typename Assembler::Basis;

    MultiGPUUnstructuredProlongation(MultiGPUContext *ctx_, Partitioner *fine_part_,
                                     Partitioner *coarse_part_, Assembler *fine_assembler_,
                                     Assembler *crs_assembler_, int block_dim_, Mat *fine_mat_,
                                     Mat *crs_mat_, int ELEM_MAX_ = 10)
        : ctx(ctx_), fine_part(fine_part_), coarse_part(coarse_part_) {
        fine_num_nodes = fine_part->num_nodes;
        coarse_num_nodes = coarse_part->num_nodes;
        fine_num_elements = fine_part->num_elements;
        coarse_num_elements = coarse_part->num_elements;
        ngpus = ctx->ngpus;
        cublasHandles = ctx->cublasHandles;
        streams = ctx->streams;
        block_dim = block_dim_;
        nxe_fine = nxe_fine_, nxe_coarse = nxe_coarse_;
        weights = new Vec(ctx, fine_part, block_dim);
        // matrices only stored since they contain reduced elem_conn needed for prolong assembly
        fine_mat = fine_mat_, coarse_mat = crs_mat_;
        fine_assembler = fine_assembler_, crs_assembler = crs_assembler_;
        fine_xpts = fine_assembler->getDeviceXpts();
        crs_xpts = crs_assembler->getDeviceXpts();
        ELEM_MAX = ELEM_MAX_;

        descr_P = new cusparseMatDescr_t[ngpus];
        descr_PT = new cusparseMatDescr_t[ngpus];
        for (int g = 0; g < ngpus; g++) {
            descr_P[g] = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_P[g]));
            CHECK_CUSPARSE(cusparseSetMatType(descr_P[g], CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_P[g], CUSPARSE_INDEX_BASE_ZERO));
            descr_PT[g] = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_PT[g]));
            CHECK_CUSPARSE(cusparseSetMatType(descr_PT[g], CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_PT[g], CUSPARSE_INDEX_BASE_ZERO));
        }

        construct_nz_pattern();
        assemble_matrices();
    }

    void construct_nz_pattern() {
        fine_xpts->expandToLocal();
        crs_xpts->expandToLocal();
        d_n2e_ptr = new int *[ngpus];
        d_n2e_elems = new int *[ngpus];
        d_n2e_xis = new T *[ngpus];
        P_bsr_data = new BsrData[ngpus];
        PT_bsr_data = new BsrData[ngpus];
        d_P_vals = new DeviceVec<T>(ngpus);
        d_PT_vals = new DeviceVec<T>(ngpus);

        for (int g = 0; g < ngpus; g++) {
            // get local connectivity and xpts for each GPU
            int *h_fine_loc_elem_conn = fine_mat->getHostLocalElemConn(g);
            int *h_crs_loc_elem_conn = coarse_mat->getHostLocalElemConn(g);

            int fine_loc_nnodes = fine_xpts->getExpandedNodes();
            int crs_loc_nnodes = crs_xpts->getExpandedNodes();
            T *h_fine_loc_xpts = fine_xpts->getLocalVecOnHost(g);
            T *h_crs_loc_xpts = crs_xpts->getLocalVecOnHost(g);

            int fine_nelems = fine_part->getLocalNumElements(g);
            int coarse_nelems = coarse_part->getLocalNumElements(g);
            int *h_fine_elem_comp = fine_assembler->getLocalElemComponents(g);
            int *h_crs_elem_comp = crs_assembler->getLocalElemComponents(g);

            // assumes elements on this partition line up from coarse to fine
            // aka fine partition is subset of coarse partition and vice versa (equal boundaries)
            // True if using TacsComponentPartitioner
            init_unstructured_grid_maps<T, Basis, true, true>(
                h_fine_loc_xpts, h_crs_loc_xpts, fine_loc_nnodes, crs_loc_nnodes,
                h_fine_loc_elem_conn, h_crs_loc_elem_conn, fine_nelems, coarse_nelems,
                h_fine_elem_comp, h_crs_elem_comp, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                P_bsr_data[g], PT_bsr_data[g], d_P_vals[g], d_PT_vals[g], ELEM_MAX);
        }
    }

    void assemble_matrices() {
        // get matrix values (no permutations, only singleGPU coarse direct uses it but done
        // internally). Also apply_bcs done to vecs in gmg solver so not needed here
        d_coarse_weights = new DeviceVec<T>(ngpus);

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int *d_crs_loc_elem_conn = coarse_mat->getLocalElemConn(g);
            int fine_loc_nnodes = fine_xpts->getExpandedNodes();
            int *d_fine_iperm = P_bsr_data[g].iperm;
            int *d_coarse_iperm = PT_bsr_data[g].iperm;
            int N_loc_coarse = crs_xpts->getExpandedSize(g);
            d_coarse_weights[g] = DeviceVec<T>(N_loc_coarse);

            dim3 block(32);
            dim3 grid((fine_loc_nnodes + 31) / 32);
            int *d_P_rowp = P_bsr_data[g].rowp;
            int *d_P_cols = P_bsr_data[g].cols;
            k_prolong_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                fine_loc_nnodes, d_fine_iperm, d_P_rowp, d_P_cols, block_dim, d_P_vals[g]);

            // assemble PT mat
            int *d_PT_rowp = PT_bsr_data[g].rowp;
            int *d_PT_cols = PT_bsr_data[g].cols;
            k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                fine_loc_nnodes, d_fine_iperm, d_PT_rowp, d_PT_cols, block_dim, d_PT_vals[g],
                d_coarse_weights[g]);
            CHECK_CUDA(cudaGetLastError());
        }

        // weights to normalize P? not in original unstruct because num_attached_elems does it?
    }

    void prolongate(Vec *coarse_in, Vec *fine_out) {
        coarse_in->expandToLocal();
        fine_out->zeroAll();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_in->getLocalPtr(g);
            T *loc_fine = fine_out->getLocalPtr(g);

            T a = 1.0, b = 0.0;
            int mb = P_bsr_data[g].mb, nb = P_bsr_data[g].nb;
            int *d_P_rowp = P_bsr_data[g].rowp;
            int *d_P_cols = P_bsr_data[g].cols;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandles[g], CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                                          P_bsr_data[g].nnzb, &a, descr_P[g], d_P_vals[g], d_P_rowp,
                                          d_P_cols, block_dim, loc_coarse, &b, loc_fine));
        }
        ctx->sync();
        fine_out->reduceFromLocal();
    }

    void restrict_vec(Vec *fine_in, Vec *coarse_out) {
        fine_in->expandToLocal();
        coarse_out->zeroAll();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_out->getLocalPtr(g);
            T *loc_fine = fine_in->getLocalPtr(g);

            T a = 1.0, b = 0.0;
            int mb = PT_bsr_data[g].mb, nb = PT_bsr_data[g].nb;
            int *d_P_rowp = PT_bsr_data[g].rowp;
            int *d_P_cols = PT_bsr_data[g].cols;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandles[g], CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                nb, PT_bsr_data[g].nnzb, &a, descr_PT[g], d_PT_vals[g], d_PT_rowp, d_PT_cols,
                block_dim, loc_fine, &b, loc_coarse));
        }
        ctx->sync();
        coarse_out->reduceFromLocal();

        // TODO : add normalize part for nonlinear structures (restrict with normalize)
    }

    void free() {
        // TBD
    }

   private:
    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;
    MultiGPUContext *ctx = nullptr;
    Partitioner *fine_part = nullptr;
    Partitioner *coarse_part = nullptr;

    Assembler *fine_assembler, *crs_assembler;

    int ngpus, block_dim;
    int fine_num_nodes, coarse_num_nodes;
    int fine_num_elements, coarse_num_elements;
    int nxe_coarse, nxe_fine;
    Vec *weights;
    Mat *fine_mat, *coarse_mat;
    Vec *fine_xpts, *crs_xpts;

    // matrix values
    cusparseMatDescr_t *descr_P, *descr_PT;
    BsrData *P_bsr_data, *PT_bsr_data;
    DeviceVec<T> *d_P_vals, *d_PT_vals, *d_coarse_weights;
    int **d_n2e_ptr, **d_n2e_elems;
    T **d_n2e_xis;
    int ELEM_MAX = 10;
};
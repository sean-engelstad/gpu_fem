#pragma once
#include "gpumat.h"
#include "gpuvec.h"
#include "multigpu_context.h"

template <typename T, class Partition, class Basis, ProlongationGeom geom>
class MultiGPUStructuredProlongation {
   public:
    using Vec = GPUvec<T, Partitioner>;
    // using Mat = GPUbsrmat<T, Partitioner>;

    MultiGPUStructuredProlongation(MultiGPUContext *ctx_, Partitioner *fine_part_,
                                   Partitioner *coarse_part_, int nxe_fine_, int nxe_coarse_)
        : ctx(ctx_), fine_part(fine_part_), coarse_part(coarse_part_) {
        fine_num_nodes = fine_part->num_nodes;
        coarse_num_nodes = coarse_part->num_nodes;
        fine_num_elements = fine_part->num_elements;
        coarse_num_elements = coarse_part->num_elements;
        ngpus = ctx->ngpus;
        cublasHandles = ctx->cublasHandles;
        streams = ctx->streams;
        block_dim = ctx->block_dim;
        nxe_fine = nxe_fine_, nxe_coarse = nxe_coarse_;
        weights = new Vec(ctx, fine_part, block_dim);

        compute_row_sum_weights();
    }

    void compute_row_sum_weights() {
        weights->zero();
        weights->zeroLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_weights = weights->getLocalPtr(g);

            dim3 block(32);
            dim3 grid((loc_nelems + block.x - 1) / block.x);
            k_structured_weights<T, Partition, Basis, geom><<<grid, block, 0, streams[g]>>>(
                block_dim, nxe_coarse, nxe_fine, loc_fine_nelems, loc_weights);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        // reduce add then broadcasts from owned to local
        weights->reduceFromLocal();
        weights->expandToLocal();
    }

    void prolongate(Vec *coarse_in, Vec *fine_out) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_coarse = coarse_in->getLocalPtr(g);
            T *loc_fine = fine_out->getLocalPtr(g);
            T *loc_weights = weights->getLocalPtr(g);

            dim3 block(32);
            dim3 grid((loc_nelems + block.x - 1) / block.x);
            k_structured_prolongate<T, Partition, Basis, geom>
                <<<grid, block, 0, streams[g]>>>(block_dim, nxe_coarse, nxe_fine, loc_fine_nelems,
                                                 loc_weights, loc_coarse, loc_fine);
            CHECK_CUDA(cudaGetLastError());
        }

        ctx->sync();
    }

    void restrict_vec(Vec *fine_in, Vec *coarse_out) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_fine = fine_in->getLocalPtr(g);
            T *loc_coarse = coarse_out->getLocalPtr(g);
            T *loc_weights = weights->getLocalPtr(g);

            dim3 block(32);
            dim3 grid((loc_nelems + block.x - 1) / block.x);
            k_structured_restrict<T, Partition, Basis, geom>
                <<<grid, block>>>(block_dim, nxe_coarse, nxe_fine, loc_fine_nelems, loc_weights,
                                  loc_fine, loc_coarse);
            CHECK_CUDA(cudaGetLastError());
        }

        ctx->sync();
    }

   private:
    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;
    MultiGPUContext *ctx = nullptr;
    Partitioner *fine_part = nullptr;
    Partitioner *coarse_part = nullptr;

    int ngpus, block_dim;
    int fine_num_nodes, coarse_num_nodes;
    int fine_num_elements, coarse_num_elements;
    int nxe_coarse, nxe_fine;
    Vec *weights;
};
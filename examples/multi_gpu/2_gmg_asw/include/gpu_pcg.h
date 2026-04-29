#pragma once
#include "gpumat.h"
#include "gpuvec.h"
#include "multigpu_context.h"

template <typename T, class Partitioner, class PRECOND>
class GPU_PCG {
   public:
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;

    GPU_PCG(MultiGPUContext *ctx_, const Partitioner *part_, Mat *A_, PRECOND *M_, int N_,
            int block_dim_ = 6)
        : ctx(ctx_), part(part_), A(A_), M(M_), N(N_), block_dim(block_dim_) {
        resid = new Vec(ctx, part, block_dim);
        w = new Vec(ctx, part, block_dim);
        p = new Vec(ctx, part, block_dim);
        z = new Vec(ctx, part, block_dim);
    }

    ~GPU_PCG() {
        delete resid;
        delete w;
        delete p;
        delete z;
    }

    int solve(Vec *rhs, Vec *x, int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8,
              int print_freq = 50, bool can_print = true) {
        T a = 0.0;
        T b = 0.0;

        ctx->sync();
        auto start = std::chrono::high_resolution_clock::now();

        rhs->copyTo(resid);

        a = -1.0;
        b = 1.0;
        A->mult(a, x, b, resid);

        T init_resid_norm = resid->norm();

        if (can_print) {
            printf("PCG init_resid = %.8e\n", init_resid_norm);
        }

        M->solve(resid, z);
        z->copyTo(p);

        bool converged = false;
        int total_iter = 0;

        for (int j = 0; j < max_iter; j++, total_iter++) {
            a = 1.0;
            b = 0.0;
            A->mult(a, p, b, w);

            T rz0 = resid->dotProd(z);
            T wp0 = w->dotProd(p);
            T alpha = rz0 / wp0;

            x->axpy(alpha, p);

            a = -alpha;
            resid->axpy(a, w);

            M->solve(resid, z);

            T rz1 = resid->dotProd(z);
            T beta = rz1 / rz0;

            p->scale(beta);
            a = 1.0;
            p->axpy(a, z);

            T resid_norm = resid->norm();

            if (can_print && (j % print_freq == 0)) {
                printf("PCG [%d] = %.8e\n", j, resid_norm);
            }

            if (std::abs(resid_norm) < abs_tol + init_resid_norm * rel_tol) {
                converged = true;

                if (can_print) {
                    printf("\tPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
                }

                break;
            }
        }

        ctx->sync();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;

        if (can_print) {
            printf("\tfinished PCG in %.4e sec\n", dt);
        }

        return converged ? total_iter + 1 : -total_iter;
    }

    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    Mat *A = nullptr;
    PRECOND *M = nullptr;

    int N = 0;
    int block_dim = 0;

    Vec *resid = nullptr;
    Vec *w = nullptr;
    Vec *p = nullptr;
    Vec *z = nullptr;
};
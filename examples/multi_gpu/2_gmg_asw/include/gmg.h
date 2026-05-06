#pragma once

#include <vector>

#include "multigpu_context.h"

template <typename T, class PARTITION, class ASSEMBLER, class SMOOTHER, class PROLONGATION,
          class COARSE_SOLVER>
class MultiGPUGeometricMultigrid {
   public:
    using VEC = GPUVec<T, PARTITION>;

    MultiGPUGeometricMultigrid(MultiGPUContext *ctx_, std::vector<ASSEMBLER *> assemblers_,
                               std::vector<MAT *> mats_, std::vector<SMOOTHER *> smoothers_,
                               std::vector<PROLONGATION *> prolongations_,
                               COARSE_SOLVER *coarse_solver_, int MAX_STEPS_ = 500, T rtol_ = 1e-6,
                               T atol_ = 1e-30, bool PRINT_ = false, int print_freq_ = 10,
                               T line_search_min_ = 1e-2, T line_search_max_ = 2.0)
        : ctx(ctx_),
          assemblers(assemblers_),
          mats(mats_),
          smoothers(smoothers_),
          prolongations(prolongations_),
          coarse_solver(coarse_solver_) {
        nlevels = assemblers.size();
        MAX_STEPS = MAX_STEPS_;
        line_search_min = line_search_min_;
        line_search_max = line_search_max_;
        print_freq = print_freq_;
        PRINT = PRINT_;
        rtol = rtol_, atol = atol_;

        // make vecs on each level
        for (int level = 0; level < nlevels; level++) {
            d_defects[level] = assemblers[level]->createGPUVec();
            d_solns[level] = assemblers[level]->createGPUVec();
            d_temp[level] = assemblers[level]->createGPUVec();
            d_temp_defect[level] = assemblers[level]->createGPUVec();
        }
    }

    void solve(VEC *rhs, FINE_VEC *soln) {
        // V-cycle solve here..
        OBSERVED_STEPS = 0;

        // somehow set fine grid defect from rhs
        rhs->copyTo(d_defects[0]);
        T init_defect_norm = d_defects[0]->norm();
        T converged_nrm = atol + rtol * init_defect_norm;
        T final_defect_norm = init_defect_norm;
        // V-cycle steps
        for (int STEP = 0; STEP < MAX_STEPS; STEP++) {
            // restrict + pre-smooth from fine to coarse
            for (int level = 0; level < nlevels - 1; level++) {
                // pre-smooth (solve here is equivalent to smoothDefect)
                smoothers[level]->solve(d_defects[level], d_solns[level]);

                // restrict
                prolongations[level]->restrict_vec(d_defects[level], d_defects[level + 1]);
                assemblers[level + 1]->apply_bcs(d_defects[level + 1]);
            }

            // coarse solve
            coarse_solver->solve(d_defects[nlevels - 1], d_solns[nlevels - 1]);

            // prolong + post-smooth back up from coarse to fine
            for (int level = nlevels - 2; level >= 0; level--) {
                // initial prolong
                prolongations[level]->prolongate(d_solns[level + 1], d_temp[level]);
                assemblers[level]->apply_bcs(d_temp[level]);

                // line search on d_temp
                mats[level]->mult(d_temp[level], d_temp_defect[level]);
                T dp1 = d_temp[level]->dotProd(d_defects[level]);
                T dp2 = d_temp[level]->dotProd(d_temp_defect[level]);
                T omega = dp1 / dp2;
                omega = std::max(line_search_min, std::min(line_search_max, omega));
                d_solns[level]->axpy(omega, d_temp[level]);
                d_defects[level]->axpy(-omega, d_temp_defect[level]);

                // post-smooth
                smoothers[level]->solve(d_defects[level], d_solns[level]);
            }

            // convergence check
            T defect_nrm = d_defects[0]->norm();
            final_defect_norm = defect_nrm;
            if (STEP % print_freq == 0 && PRINT) {
                printf("V-cycle step %d, ||defect|| = %.3e\n", STEP, defect_nrm);
            }
            if (defect_nrm < converged_nrm) {
                if (PRINT) {
                    printf(
                        "V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                        i_vcycle + 1, defect_nrm, init_defect_nrm);
                }
                OBSERVED_STEPS = STEP + 1;
                break;
            }
        }

        // copy final solution back to output
        d_solns[0]->copyTo(soln);
    }

   private:
    int nlevels;
    MultiGPUContext *ctx;
    std::vector<ASSEMBLER *> assemblers;
    std::vector<MAT *> mats;
    std::vector<SMOOTHER *> smoothers;
    std::vector<PROLONGATION *> prolongations;
    COARSE_SOLVER *coarse_solver;
    int MAX_STEPS;

    // also need some vectors at each level too?
    std::vector<VEC *> d_defects;
    std::vector<VEC *> d_solns;
    std::vector<VEC *> d_temp;
    std::vector<VEC *> d_temp_defect;

    T line_search_min, line_search_max, rtol, atol;
    int print_freq, OBSERVED_STEPS;
    bool PRINT;
};

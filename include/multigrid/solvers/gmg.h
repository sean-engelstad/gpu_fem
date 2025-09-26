#pragma once

// original multilevel Vcycle solver
// V1 I should say..

template <class GRID>
class GeometricMultigridSolver {
    /* a standard geomtric multigrid solver class */
    using T = double;

   public:
    GeometricMultigridSolver() = default;

    template <class Basis>
    void init_unstructured(int ELEM_MAX = 4) {
        /* initialize unstructured grid maps */
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel].template init_unstructured_grid_maps<Basis>(grids[ilevel + 1], ELEM_MAX);
        }
        // setup = true;
    }

    int getNumLevels() { return grids.size(); }

    double get_memory_usage_mb() {
        // get total memory usage of each Kmat across all grids
        double total_mem = 0.0;
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            total_mem += grids[ilevel].get_memory_usage_mb();
        }
        return total_mem;
    }

    void set_design_variables(DeviceVec<T> dvs) {
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].assembler.set_design_variables(dvs);
        }
    }

    void vcycle_solve(int starting_level, int pre_smooth, int post_smooth, int n_vcycles = 100, bool print = false,
                      T atol = 1e-6, T rtol = 1e-6, T omega = 1.0, bool double_smooth = false,
                      int print_freq = 1, bool symmetric = false, bool time = false, bool debug = false) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        if (print) printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_vcycles;

        // printf("V-cycle : print_freq %d\n", print_freq);

        int n_levels = getNumLevels();
        // if (print) printf("n_levels %d\n", n_levels);

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            // printf("V cycle step %d\n", i_vcycle);

            // go down each level smoothing and restricting until lowest level
            for (int i_level = starting_level; i_level < n_levels; i_level++) {
                int exp_smooth_factor = double_smooth ? 1 << i_level : 1;  // power of 2 more
                // int exp_smooth_factor = 1.0;  // only smooth post-steps if this

                // if not last  (pre-smooth)
                if (i_level < n_levels - 1) {
                    // if (print) printf("\tlevel %d pre-smooth\n", i_level);

                    if (time) CHECK_CUDA(cudaDeviceSynchronize());
                    auto pre_smooth_time = std::chrono::high_resolution_clock::now();

                    grids[i_level].smoothDefect(pre_smooth * exp_smooth_factor, debug,
                                                pre_smooth * exp_smooth_factor - 1, omega, symmetric);

                    if (time) {
                        CHECK_CUDA(cudaDeviceSynchronize());
                        auto post_smooth_time = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> smooth_time1 =
                            post_smooth_time - pre_smooth_time;
                        printf("\tsmoothDefect[%d]-down in %.2e sec\n", i_level,
                               smooth_time1.count());
                    }
                    auto pre_restr_time = std::chrono::high_resolution_clock::now();

                    // restrict defect
                    grids[i_level + 1].restrict_defect(
                        grids[i_level].nelems, grids[i_level].d_iperm, grids[i_level].d_defect);

                    if (time) {
                        CHECK_CUDA(cudaDeviceSynchronize());
                        auto post_restr_time = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> restr_time = post_restr_time - pre_restr_time;
                        printf("\trestr-defect[%d] in %.2e sec\n", i_level, restr_time.count());
                    }

                } else {
                    if (debug) printf("\t--level %d full-solve\n", i_level);
                    // printf("pre-direct-solve\n");
                    if (time) CHECK_CUDA(cudaDeviceSynchronize());
                    auto pre_direct_time = std::chrono::high_resolution_clock::now();

                    // coarsest grid full solve
                    grids[i_level].direct_solve(false);

                    if (time) {
                        CHECK_CUDA(cudaDeviceSynchronize());
                        auto post_direct_time = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> direct_time =
                            post_direct_time - pre_direct_time;
                        printf("\tdirect-solve[%d] in in %.2e sec\n", i_level, direct_time.count());
                    }
                }
            }

            // now go back up the hierarchy
            for (int i_level = n_levels - 2; i_level >= starting_level; i_level--) {
                if (time) CHECK_CUDA(cudaDeviceSynchronize());
                auto pre_prolong_time = std::chrono::high_resolution_clock::now();

                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);
                // if (print) printf("\tlevel %d post-smooth\n", i_level);

                // TEMP DEBUG a smoothed prolongation..
                // grids[i_level].smoothed_prolongate(grids[i_level + 1].d_iperm,
                //                                    grids[i_level + 1].d_soln, 30);
                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_prolong_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> prolong_time =
                        post_prolong_time - pre_prolong_time;
                    printf("\tprolong-soln[%d] up in in %.2e sec\n", i_level, prolong_time.count());
                }
                auto pre_smooth_up_time = std::chrono::high_resolution_clock::now();

                // post-smooth
                // printf("post-smooth on level %d\n", i_level);
                int exp_smooth_factor =
                    double_smooth ? 1 << i_level : 1;  // power of 2 more smoothing at each level..
                grids[i_level].smoothDefect(post_smooth * exp_smooth_factor, debug,
                                            post_smooth * exp_smooth_factor - 1, omega, symmetric);

                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_smooth_up_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> smooth_up_time =
                        post_smooth_up_time - pre_smooth_up_time;
                    printf("\tpost-smooth[%d] up in in %.2e sec\n", i_level,
                           smooth_up_time.count());
                }
            }

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[starting_level].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            if (i_vcycle % print_freq == 0 && print)
                printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());

            if (defect_nrm < atol + rtol * init_defect_nrm && print) {
                printf("V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_vcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_vcycle + 1;
                break;
            }
        }

        if (print) printf("done with v-cycle solve from start level %d, conv %.2e to %.2e ||defect|| in %d steps\n",
               starting_level, init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void fcycle_solve(int starting_level, int pre_smooth, int post_smooth, int n_fcycles = 100, bool print = false,
                      T atol = 1e-6, T rtol = 1e-6, T omega = 1.0, bool double_smooth = false,
                      int print_freq = 1, bool symmetric = false, bool time = false, bool debug = false) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        if (print) printf("F-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_fcycles;

        // printf("V-cycle : print_freq %d\n", print_freq);

        int n_levels = getNumLevels();
        // if (print) printf("n_levels %d\n", n_levels);
        int i_level = starting_level;

        for (int i_cycle = 0; i_cycle < n_fcycles; i_cycle++) {

            // pre-smooth and restrict
            if (i_level < n_levels - 1) {
                grids[i_level].smoothDefect(pre_smooth, debug, pre_smooth - 1, omega, symmetric);

                // restrict defect
                grids[i_level + 1].restrict_defect(
                    grids[i_level].nelems, grids[i_level].d_iperm, grids[i_level].d_defect);
            }

            // call inner F-cycle 1 time
            if (i_level < n_levels - 1) {
                int pre_smooth_l = double_smooth ? pre_smooth * 2 : pre_smooth;
                int post_smooth_l = double_smooth ? post_smooth * 2 : post_smooth;

                fcycle_solve(i_level + 1, pre_smooth_l, post_smooth_l, 1, false, atol, rtol, omega, double_smooth);
            } else {
                grids[i_level].direct_solve(false);
            }

            // post-smooth and restrict
            if (i_level < n_levels - 1) {
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

                grids[i_level].smoothDefect(post_smooth, debug, post_smooth, omega, symmetric);
            }

            // then call V-cycle at this level (one iteration)
            vcycle_solve(i_level, pre_smooth, post_smooth, 1, false, atol, rtol, omega, double_smooth);

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[i_level].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            if (i_cycle % print_freq == 0 && print)
                printf("F-cycle step %d, ||defect|| = %.3e\n", i_cycle, defect_nrm);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());

            if (defect_nrm < atol + rtol * init_defect_nrm && print) {
                printf("F-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_cycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_cycle + 1;
                break;
            }
        }

        if (print) printf("done with F-cycle solve from start level %d, conv %.2e to %.2e ||defect|| in %d steps\n",
               starting_level, init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void wcycle_solve(int i_level, int pre_smooth, int post_smooth, int n_wcycles = 100,
                      bool print = false, T atol = 1e-6, T rtol = 1e-6, T omega = 1.0) {
        /* W-cycle may have greater performance than V-cycle? what about also F-cycle? */

        bool is_outer_call = n_wcycles > 2 && i_level == 0;
        // printf("i_level %d, n_wcycles %d, is outer call %d\n", i_level, n_wcycles,
        // is_outer_call);
        T init_defect_nrm, fin_defect_nrm;
        if (is_outer_call) {  // only for outer call..
            // init defect nrm
            init_defect_nrm = grids[0].getDefectNorm();
            printf("W-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);
            fin_defect_nrm = init_defect_nrm;
        }

        int n_steps = n_wcycles;
        int n_levels = getNumLevels();
        // if (print) printf("n_levels %d\n", n_levels);

        for (int i_wcycle = 0; i_wcycle < n_wcycles; i_wcycle++) {
            // pre-smooth and restrict if not last level
            if (i_level < n_levels - 1) {
                // pre-smooth at this level
                grids[i_level].smoothDefect(pre_smooth, print, pre_smooth - 1, omega);

                // restrict defect
                grids[i_level + 1].restrict_defect(grids[i_level].nelems, grids[i_level].d_iperm,
                                                   grids[i_level].d_defect);
            }

            // solve this level or call lower level W-cycle (recursively)
            if (i_level < n_levels - 1) {
                // recursive call to lower level (2 inner W-cycles each step..)
                wcycle_solve(i_level + 1, pre_smooth, post_smooth, 2, print, atol, rtol, omega);
            } else {
                // coarsest grid full solve
                if (print) printf("\t--level %d full-solve\n", i_level);
                grids[i_level].direct_solve(false);
            }

            // now post-smooth and back up the hierarchy (if not direct solve inner level)
            if (i_level < n_levels - 1) {
                // prolongate from lower level
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

                // post smooth
                bool rev_colors = true;
                grids[i_level].smoothDefect(post_smooth, print, post_smooth - 1, omega, rev_colors);
            }

            // compute fine grid defect of V-cycle (outer call only)++
            T defect_nrm;
            if (is_outer_call) {
                defect_nrm = grids[i_level].getDefectNorm();
                fin_defect_nrm = defect_nrm;
                printf("W-cycle step %d, ||defect|| = %.3e\n", i_wcycle, defect_nrm);
            }

            if (defect_nrm < atol + rtol * init_defect_nrm && is_outer_call) {
                printf("W-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_wcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_wcycle + 1;
                break;
            }
        }

        if (is_outer_call) {
            printf("done with W-cycle solve, conv %.2e to %.2e ||defect|| in %d steps\n",
                   init_defect_nrm, fin_defect_nrm, n_steps);
        }
    }  // end of wcycle_solve method

    void free() {
        for (int ilevel = 0; ilevel < n_levels; ilevel++) {
            grids[ilevel].free();
        }
    }

    int nxe, n_levels;
    bool setup;
    std::vector<GRID> grids;
};
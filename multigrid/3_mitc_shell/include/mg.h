#pragma once

template <class GRID>
class ShellMultigrid {
    /* shell elem geomtric multigrid solver class */
    using T = double;

   public:
    ShellMultigrid() = default;

    template <class Basis>
    void init_unstructured(int ELEM_MAX = 4) {
        /* initialize unstructured grid maps */
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel].template init_unstructured_grid_maps<Basis>(grids[ilevel + 1], ELEM_MAX);
        }
        // setup = true;
    }

    int getNumLevels() { return grids.size(); }

    void vcycle_solve(int pre_smooth, int post_smooth, int n_vcycles = 100, bool print = false,
                      T atol = 1e-6, T rtol = 1e-6, T omega = 1.0, bool double_smooth = false) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_vcycles;

        int n_levels = getNumLevels();
        // if (print) printf("n_levels %d\n", n_levels);

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            // printf("V cycle step %d\n", i_vcycle);

            // go down each level smoothing and restricting until lowest level
            for (int i_level = 0; i_level < n_levels; i_level++) {
                // int exp_smooth_factor = double_smooth ? 1<<i_level : 1; // power of 2 more
                // smoothing at each level..
                int exp_smooth_factor = 1.0;  // only on post-steps..

                // if not last  (pre-smooth)
                if (i_level < n_levels - 1) {
                    // if (print) printf("\tlevel %d pre-smooth\n", i_level);

                    // pre-smooth; TODO : do fast version later.. but let's demo with slow version
                    // first
                    grids[i_level].smoothDefect(pre_smooth * exp_smooth_factor, print,
                                                pre_smooth * exp_smooth_factor - 1, omega);

                    // restrict defect
                    grids[i_level + 1].restrict_defect(
                        grids[i_level].nelems, grids[i_level].d_iperm, grids[i_level].d_defect);

                } else {
                    if (print) printf("\t--level %d full-solve\n", i_level);

                    // coarsest grid full solve
                    grids[i_level].direct_solve(false);
                }
            }

            // now go back up the hierarchy
            for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
                // get coarse-fine correction from coarser grid to this grid
                // printf("prolongate from grid %d => %d\n", i_level + 1, i_level);
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);
                // if (print) printf("\tlevel %d post-smooth\n", i_level);

                // post-smooth
                // printf("post-smooth on level %d\n", i_level);
                int exp_smooth_factor =
                    double_smooth ? 1 << i_level : 1;  // power of 2 more smoothing at each level..
                bool rev_colors = true;
                grids[i_level].smoothDefect(post_smooth * exp_smooth_factor, print,
                                            post_smooth * exp_smooth_factor - 1, omega, rev_colors);
            }

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[0].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);

            if (defect_nrm < atol + rtol * init_defect_nrm) {
                printf("V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_vcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_vcycle + 1;
                break;
            }
        }

        printf("done with v-cycle solve, conv %.2e to %.2e ||defect|| in %d steps\n",
               init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void wcycle_solve(int i_level, int pre_smooth, int post_smooth, int n_wcycles = 100,
                      bool print = false, T atol = 1e-6, T rtol = 1e-6, T omega = 1.0) {
        /* W-cycle may have greater performance than V-cycle? what about also F-cycle? */

        bool is_outer_call = n_wcycles > 2 && i_level == 0;
        printf("i_level %d, n_wcycles %d, is outer call %d\n", i_level, n_wcycles, is_outer_call);
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

            // compute fine grid defect of V-cycle (outer call only)
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
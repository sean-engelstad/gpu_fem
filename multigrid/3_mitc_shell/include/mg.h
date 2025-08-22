#pragma once

template <class GRID>
class ShellMultigrid {
    /* shell elem geomtric multigrid solver class */
    using T = double;

   public:
    ShellMultigrid() = default;

    int getNumLevels() { return grids.size(); }

    void vcycle_solve(int pre_smooth, int post_smooth, int n_vcycles = 100, bool print = false,
                      int inner_solve_iters = 100, T atol = 1e-6, T rtol = 1e-6) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

        int n_levels = getNumLevels();
        // if (print) printf("n_levels %d\n", n_levels);

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            // printf("V cycle step %d\n", i_vcycle);

            // go down each level smoothing and restricting until lowest level
            for (int i_level = 0; i_level < n_levels; i_level++) {
                // if not last  (pre-smooth)
                if (i_level < n_levels - 1) {
                    // if (print) printf("\tlevel %d pre-smooth\n", i_level);

                    // pre-smooth; TODO : do fast version later.. but let's demo with slow version
                    // first
                    grids[i_level].multicolorBlockGaussSeidel_slow(pre_smooth, print,
                                                                   pre_smooth - 1);

                    // restrict defect
                    grids[i_level + 1].restrict_defect(
                        grids[i_level].nelems, grids[i_level].d_elem_conn, grids[i_level].d_iperm,
                        grids[i_level].d_defect);

                } else {
                    if (print) printf("\t--level %d full-solve\n", i_level);

                    // coarsest grid full solve
                    grids[i_level].direct_solve(false);
                }
            }

            // now go back up the hierarchy
            for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

                // if (print) printf("\tlevel %d post-smooth\n", i_level);

                // post-smooth
                grids[i_level].multicolorBlockGaussSeidel_slow(post_smooth, print, post_smooth - 1);
            }

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[0].getDefectNorm();
            printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);

            if (defect_nrm < atol + rtol * init_defect_nrm) {
                printf("V-cycle GMG converged in %d steps\n", i_vcycle + 1);
                break;
            }
        }
    }

    void free() {
        for (int ilevel = 0; ilevel < n_levels; ilevel++) {
            grids[ilevel].free();
        }
    }

    int nxe, n_levels;
    bool setup;
    std::vector<GRID> grids;
};
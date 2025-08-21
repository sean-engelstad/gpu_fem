#include "grid.h"

template <class Grid>
class ShellMultigrid {
    /* shell elem geomtric multigrid solver class */
    using GRID = ShellGrid;
    using T = double;

   public:
    ShellMultigrid(int nxe_) : nxe(nxe_) {
        // number of levels for nxe_ being finest grid num elements in x-dir (TODO : adapt this from plate case later..)
        int log2_nxe = 0, nxe_copy = nxe;
        while (nxe_copy >>= 1) ++log2_nxe;
        n_levels = log2_nxe - 1;

        // check nxe is power of 2 (for good geom multigrid property, convenience)
        int nxe_power_2 = 1 << log2_nxe;
        assert(nxe_power_2 == nxe);
    }

    void addGrid(int ilevel, Grid &grid) {
        grids[ilevel] = grid;
    }

    void vcycle_solve(int pre_smooth, int post_smooth, int n_vcycles = 100, bool print = false,
                      int inner_solve_iters = 100, T atol = 1e-6, T rtol = 1e-6) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        printf("init defect nrm, ||defect|| = %.2e\n", init_defect_nrm);

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            // printf("V cycle step %d\n", i_vcycle);

            // go down each level smoothing and restricting until lowest level
            for (int i_level = 0; i_level < n_levels; i_level++) {
                // if not last  (pre-smooth)
                if (i_level < n_levels - 1) {
                    if (print) printf("\tlevel %d pre-smooth\n", i_level);

                    // pre-smooth; TODO : do fast version later.. but let's demo with slow version first
                    grids[i_level].multicolorBlockGaussSeidel_slow(pre_smooth, print, pre_smooth - 1);

                    // restrict defect
                    grids[i_level + 1].restrict_defect(grids[i_level].d_perm,
                                                       grids[i_level].d_defect);
                } else {
                    if (print) printf("\tlevel %d full-solve\n", i_level);

                    // coarsest grid full solve
                    grids[i_level].direct_solve(print);
                }
            }

            // now go back up the hierarchy
            for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

                if (print) printf("\tlevel %d post-smooth\n", i_level);

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
    GRID *grids;
};
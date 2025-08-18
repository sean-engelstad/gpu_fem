#include "include/pde.h"

int main() {
    // make poisson solver and try and run it..

    using T = float; // double
    using GRID = PoissonSolver<T>;

    // set finest grid size
    // int nxe = 1024;
    int nxe = 32;

    int log2_nxe = 0, nxe_copy = nxe;
    while (nxe_copy >>= 1) ++log2_nxe;
    int n_levels = log2_nxe - 1;
    printf("nxe %d, log2(nxe) %d, n_levels %d\n", nxe, log2_nxe, n_levels);

    GRID *grids = new GRID[n_levels]; 
    int c_nxe = nxe;
    for (int ilevel = 0; ilevel < n_levels; ilevel++, c_nxe /= 2) {
        printf("level %d, making poisson solver with nxe %d elems\n", ilevel, c_nxe);
        grids[ilevel] = GRID(c_nxe);
    }

    /* DEBUG before full V-cycles */
    // T grid1_def_nrm = grids[n_levels-2].getDefectNorm();

    // // test restriction on coarsest grid
    // grids[n_levels-1].restrict_defect(grids[n_levels-2].d_defect);
    // // cudaDeviceSynchronize();

    // T *h_coarse_defect0 = grids[n_levels-1].d_defect.createHostVec().getPtr();
    // printf("coarsest grid defect: ");
    // printVec<T>(grids[n_levels-1].N, h_coarse_defect0);

    // T grid2_def_nrm = grids[n_levels-1].getDefectNorm(); 

    // printf("grid 1 |defect| = %.2e => grid 2 |defect| = %.2e\n", grid1_def_nrm, grid2_def_nrm);

    // return 0;
    /* end of DEBUG section */

    // now try multigrid V-cycle solves here

    // /* try solve here.. */
    int pre_smooth = 3, post_smooth = 3;
    T omega = 2.0 / 3.0;
    bool print = true;
    // int n_vcycles = 100;
    int n_vcycles = 1;

    for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
        printf("V cycle step %d\n", i_vcycle);

        // go down each level smoothing and restricting until lowest level
        for (int i_level = 0; i_level < n_levels; i_level++) {
            // if not last  (pre-smooth)
            if (i_level < n_levels - 1) {
                printf("\tlevel %d pre-smooth\n", i_level);

                // pre-smooth
                grids[i_level].dampedJacobiDefect(pre_smooth, omega, print, pre_smooth - 1);

                // restrict defect
                grids[i_level + 1].restrict_defect(grids[i_level].d_defect);
            } else {
                printf("\tlevel %d full-solve\n", i_level);

                // print the defect here..
                // T *h_coarse_defect = grids[i_level].d_defect.createHostVec().getPtr();
                // printf("h_coarse_defect: ");
                // printVec<T>(grids[i_level].N, h_coarse_defect);

                // full-solve on last grid (of current defect)
                grids[i_level].dampedJacobiDefect(100, omega, print, 99);
            }
        }

        // now go back up the hierarchy
        for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
            // get coarse-fine correction from coarser grid to this grid
            grids[i_level].prolongate(grids[i_level + 1].d_soln);

            printf("\tlevel %d post-smooth\n", i_level);

            // post-smooth
            grids[i_level].dampedJacobiDefect(post_smooth, omega, print, post_smooth - 1);
        }

        // compute fine grid defect of V-cycle
        T defect_nrm = grids[0].getDefectNorm();
        printf("\tend of v-cycle step %d, ||defect|| = %.2e\n", i_vcycle, defect_nrm);

    }

    // free
    for (int ilevel = 0; ilevel < n_levels; ilevel++) {
        grids[ilevel].free();
    }

    return 0;
};
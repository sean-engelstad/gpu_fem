#pragma once
#include "../solve_utils.h"

// cusparse directLU solves with multigrid (V-cycle style preconditioner for two levels)
// only works from coarsest grid to next level

template <class GRID>
class CusparseMGDirectLU : public BaseSolver {
public:
    CusparseMGDirectLU(GRID *grid_) : grid(grid_) { }

    void solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {

        // approximate fine grid solve using V-cycle and coarse direct solves
        bool is_perm = false;
        grid->setDefect(rhs, is_perm);
        grid->zeroSolution();

        // coarse grid directLU solve
        grid->direct_solve(false);

        // get solution out
        grid->getSolution(soln, is_perm);
    }

    void free() {return;} // nothing

private:
    GRID *grid;
    SolverOptions options;
    // right now the directLU solve code is stored on the coarse grid.. may pull it off of there when I cleanup the code (and put it here)
};
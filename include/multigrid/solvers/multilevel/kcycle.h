// generic Kcycle solver
#pragma once

#include "../solve_utils.h"

template <class GRID, class CoarseSolver, class SubspaceSolver, class KrylovSolver>
class MultilevelKcycleSolver {
public:
    using T = double;

    MultilevelKcycleSolver() = default;

    // TODO : put some of these methods for multilevel in a base multilevel solver class
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

    // create the hierarchy of solvers bottom-up
    void init_solvers(SolverOptions inner_subspace_options, SolverOptions outer_subspace_options, 
        SolverOptions inner_krylov_options, SolverOptions outer_krylov_options, 
        bool just_outer_krylov = false) {
            
        // after you've created each grid, call this method to make the solver hierarchy (though you can do this yourself if you wanted to)
        int nlevels = getNumLevels();
        int isolver = 0;
        for (int ilevel = nlevels - 1; ilevel >= 0; ilevel--) {
            if (ilevel == nlevels-1) {
                // create coarse grid direct solver
                BaseSolver *coarse_solver = new CoarseSolver(&grids[ilevel]);
                solvers.push_back(coarse_solver);
            } else {
                if (just_outer_krylov) {
                    // first make the subspace solver
                    auto subspace_options = ilevel == 0 ? outer_subspace_options : inner_subspace_options;
                    BaseSolver *subspace_solver = new SubspaceSolver(&grids[ilevel], &grids[ilevel+1], solvers[isolver-1], subspace_options);
                    if (ilevel != 0) solvers.push_back(subspace_solver);

                    // then make the krylov solver at this level with the subspace solver as preconditioner
                    if (ilevel == 0) {
                        auto krylov_options = ilevel == 0 ? outer_krylov_options : inner_krylov_options;
                        BaseSolver *krylov_solver = new KrylovSolver(&grids[ilevel], subspace_solver, krylov_options, ilevel);
                        solvers.push_back(krylov_solver);
                        outer_solver = krylov_solver;
                    }

                } else {  
                    // all levels have krylov
                    // first make the subspace solver
                    auto subspace_options = ilevel == 0 ? outer_subspace_options : inner_subspace_options;
                    BaseSolver *subspace_solver = new SubspaceSolver(&grids[ilevel], &grids[ilevel+1], solvers[isolver-1], subspace_options);

                    // then make the krylov solver at this level with the subspace solver as preconditioner
                    auto krylov_options = ilevel == 0 ? outer_krylov_options : inner_krylov_options;
                    BaseSolver *krylov_solver = new KrylovSolver(&grids[ilevel], subspace_solver, krylov_options, ilevel);
                    solvers.push_back(krylov_solver);
                    if (ilevel == 0) {
                        outer_solver = krylov_solver;
                    }
                }
            }
            isolver++;
        }
    }

    void solve(DeviceVec<T> rhs, DeviceVec<T> soln) {
        bool check_conv = true; // only checks conv on outer solver
        outer_solver->solve(rhs, soln, check_conv);
    }

    void solve() {
        this->solve(grids[0].d_defect, grids[0].d_soln);
    }

    void free() {
        int n_levels = getNumLevels();
        for (int ilevel = 0; ilevel < n_levels; ilevel++) {
            grids[ilevel].free();
        }
    }

// private:
    BaseSolver *outer_solver;
    std::vector<BaseSolver*> solvers; // stored coarse to fine 
    std::vector<GRID> grids; // stored fine to coarse
};
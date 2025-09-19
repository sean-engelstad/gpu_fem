#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// multigrid imports
#include "multigrid/grid.h"
#include "multigrid/fea.h"
#include "multigrid/mg.h"
#include "multigrid/interface.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

class TacsGpuMultigridSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, false>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // multigrid objects
    using Prolongation = StructuredProlongation<CYLINDER>;
    using GRID = ShellGrid<Assembler, Prolongation, MULTICOLOR_GS_FAST2>;
    using MG = ShellMultigrid<GRID>;
    using StructSolver = TacsMGInterface<T, Assembler, MG>; 

    // functions
    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    TacsGpuMultigridSolver(double rhoKS = 100.0, double safety_factor = 1.5, double load_mag = 100.0,
        int nxe = 100, int nx_comp = 5, int ny_comp = 5, double SR = 50.0) {
        // 1) Build mesh & assembler
        assert(nxe % nx_comp == 0); // evenly divisible by number of elems_per_comp
        int nye = nxe;
        assert(nye % ny_comp == 0);

        // start building multigrid object
        auto mg = MG();

        // get nxe_min for not exactly power of 2 case
        int pre_nxe_min = nxe > 32 ? 32 : 4;
        int nxe_min = pre_nxe_min;
        for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
            nxe_min = c_nxe;
        }

        // make each grid
        for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
            // make the assembler
            int c_nhe = c_nxe;
            double L = 1.0, R = 0.5, thick = L / SR;
            double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
            // double rho = 2500, ys = 350e6;
            bool imperfection = false; // option for geom imperfection
            int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
            auto assembler = createCylinderAssembler<Assembler>(c_nxe, c_nhe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop, rho, ys, nx_comp, ny_comp);
            constexpr int load_case = 3;
            double Q = load_mag; // load magnitude
            T *my_loads = getCylinderLoads<T, Physics, load_case>(c_nxe, c_nhe, L, R, Q);
            printf("making grid with nxe %d\n", c_nxe);

            // make the grid
            bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
            bool reorder = true; // to allow multicolor smoothing
            auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
            mg.grids.push_back(grid); // add new grid
        }

        // now make the solver interface
        bool print = true;
        solver = std::make_unique<StructSolver>(mg, print);
        T atol = 1e-6, rtol = 1e-6;
        int n_cycles = 200, pre_smooth = 2, post_smooth = 2, print_freq = 3;
        solver->set_mg_solver_settings(rtol, atol, n_cycles, pre_smooth, post_smooth, print_freq);

        // get struct loads on finest grid
        auto fine_grid = mg.grids[0];
        d_loads = DeviceVec<T>(fine_grid.N);
        mg.grids[0].getDefect(d_loads);
        

        // initialize any vecs needed at this level 
        auto &assembler = mg.grids[0].assembler;
        nvars = assembler.get_num_vars();
        int nn = assembler.get_num_nodes();
        soln = DeviceVec<T>(nvars);
        ndvs = assembler.get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, /*initial=*/0.02);

        // 5) Functions
        mass = std::make_unique<DMass>();
        ksfail = std::make_unique<DKSFail>(rhoKS, safety_factor);

        dvs_changed = true;
        first_solve = true;
    }

    void set_design_variables(const std::vector<T> &dvs) {
        /* check if dvs changed before running new analysis (make sure this works right) */
        dvs_changed = (dvs.size() != prev_dvs.size());
        if (!dvs_changed) {
            for (int i = 0; i < dvs.size(); i++) {
                if (dvs[i] != prev_dvs[i]) {
                    dvs_changed = true;
                    break;
                }
            }
        }

        // if (first_solve) {clear
        //     dvs_changed = true;
        //     first_solve = false;
        // }
        dvs_changed = true; // debug

        if (dvs_changed) {
            prev_dvs = dvs;
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            solver->set_design_variables(d_dvs);
        }
    }

    int get_num_vars() const { return nvars; }
    int get_num_dvs() const { return ndvs; }
    void writeSolution(const std::string &filename) const { solver->writeSoln(filename); }

    void solve() {
        if (dvs_changed) {
            printf("design changed, new solve\n");

            // debugging
            // auto h_loads = d_loads.createHostVec();
            // printf("h_loads:");
            // printVec<T>(h_loads.getSize(), h_loads.getPtr());

            solver->solve(d_loads);
            solver->copy_solution_out(soln);
        } else {
            // reload old state
            printf("design didn't change, reload vals\n");
            solver->copy_solution_in(soln);
        }
    }

    T evalFunction(const std::string &name) {
        if (name == "mass")
            return solver->evalFunction(*mass);
        else if (name == "ksfailure")
            return solver->evalFunction(*ksfail);
        throw std::invalid_argument("Unknown func");
    }

    // fills host array of length ndvs
    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        double *dptr;
        if (name == "mass") {
            solver->solve_adjoint(*mass);
            dptr = mass->dv_sens.getPtr();
        } else if (name == "ksfailure") {
            solver->solve_adjoint(*ksfail);
            dptr = ksfail->dv_sens.getPtr();
        } else {
            throw std::invalid_argument("Unknown func");
        }
        CHECK_CUDA(cudaMemcpy(out_h_sens, dptr, ndvs * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void free() {
        solver->free();
        // assembler->free();
        d_loads.free();
        d_dvs.free();
        // tear down MPI if *we* initialized it
        int mpi_inited = 0;
        MPI_Finalized(&mpi_inited);
        if (!mpi_inited) {
            MPI_Finalize();
        }
    }

   private:
    // std::unique_ptr<Assembler> assembler;
    std::unique_ptr<StructSolver> solver;
    std::unique_ptr<DMass> mass;
    std::unique_ptr<DKSFail> ksfail;

    int ndvs = 0, nvars = 0;
    DeviceVec<T> d_loads, d_dvs;
    std::vector<T> prev_dvs;
    bool dvs_changed;
    bool first_solve;
    DeviceVec<T> soln;
};
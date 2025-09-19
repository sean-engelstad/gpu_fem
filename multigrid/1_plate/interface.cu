// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/fea.h"
#include "multigrid/mg.h"
#include <string>
#include <chrono>

// optimization with GMG imports
#include "multigrid/interface.h"

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void multigrid_plate_solve(int nxe, double SR, int n_vcycles) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // multigrid objects
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2; // this is much faster than other two methods (MULTICOLOR_GS_FAST is about 2.6x slower at high DOF)
    using Prolongation = StructuredProlongation<PLATE>;
    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;
    using MGInterface = TacsMGInterface<T, Assembler, MG>; // for optimization
    
    auto mg = MG();

    // get nxe_min for not exactly power of 2 case
    int pre_nxe_min = nxe > 32 ? 32 : 4;
    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }
    
    // set the number of design variables (can increase later..)
    int nxe_dv = 4, nye_dv = 4;

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / nxe_dv, nye_per_comp = c_nye/nye_dv; 
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
        bool reorder = true; // color reorder
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    // now make the solver interface
    bool print = true;
    auto interface = MGInterface(mg, print);
    T atol = 1e-6, rtol = 1e-6;
    int n_cycles = 200, pre_smooth = 1, post_smooth = 1, print_freq = 3;
    interface.set_mg_solver_settings(rtol, atol, n_cycles, pre_smooth, post_smooth, print_freq);

    // get struct loads on finest grid
    auto fine_grid = mg.grids[0];
    DeviceVec<T> d_loads(fine_grid.N);
    mg.grids[0].getDefect(d_loads);

    // get initial dvs
    int ndvs = mg.grids[0].assembler.get_num_dvs();
    T thick = 1.0 / SR;
    auto d_dvs = DeviceVec<T>(ndvs, thick);

    // now do a linear static solve with GMG
    interface.solve(d_loads);
    interface.writeSoln("out/plate_mg1.vtk");

    // compute the function values
    T mass_val = interface.evalFunction(mass);
    T ksfail_val = interface.evalFunction(ksfail);
    printf("mass %.2e, ksfail %.2e\n", mass_val, ksfail_val);

    // try solving an adjoint problem
    auto mass = Mass<T, DeviceVec>();
    T rhoKS = 100.0, safety_factor = 1.5;
    auto ksfail = KSFailure<T, DeviceVec>(rhoKS, safety_factor);
    interface.evalFunction(ksfail);
    interface.solve_adjoint(ksfail);
    interface.writeAdjointSolution("out/plate_mg_adj1.vtk");

    // compute the design gradient
    T *dptr = ksfail->dv_sens.getPtr();
    T *h_dvgrad = new T[10];
    cudaMemcpy(h_dvgrad, dptr, 10 * sizeof(T), cudaMemcpyDeviceToHost);
    printf("h_dvgrad: ");
    printVec<T>(10, h_dvgrad);

    // try setting design variables and solving again
    d_dvs.setFullVecToConstValue(thick * 2.0);
    interface.set_design_variables(d_dvs);
    interface.solve(d_loads);
    interface.writeSoln("out/plate_mg2.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = false;
    int nxe = 256; // default value
    // int nxe = 64;
    double SR = 100.0; // default
    int n_vcycles = 50;

    multigrid_plate_solve(nxe, SR, n_vcycles);

    return 0;

    
}
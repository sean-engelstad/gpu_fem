/* develop fast block-GS multicolor using small matrix for testing purposes.. */
// 7 node test matrix with 2x2 block dim for multicoloring (aka 14 DOF).. that I've made up

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
#include "../../include/grid.h"
#include "../../include/fea.h"
#include "../../include/mg.h"
#include <string>
#include <chrono>

// #include <cusparse_v2.h>
// #include "cublas_v2.h"
// #include "cuda_utils.h"
// #include "linalg/vec.h"
// #include "solvers/linear_static/_cusparse_utils.h"

int main() {

    // fine grid input
    int nxe = 32;
    
    // shells
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
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    using Prolongation = StructuredProlongation<PLATE>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();

    // generate two grids (fine and coarse
    std::vector<GRID> grids;

    // fine mesh distortions
    int m_fine = 3, n_fine = 3;
    T x_frac_f = 0.25, y_frac_f = 0.25, shear_frac_f = 0.7;

    // coarse mesh distortions
    int m_coarse = 2, n_coarse = 2;
    T x_frac_c = 0.8, y_frac_c = 0.5, shear_frac_c = 0.5;

    double SR = 50.0;

    // make each grid
    int nxe_min = nxe / 2;
    int i_level = 0;
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)

        // distortion 
        bool is_fine = c_nxe == nxe;
        int m = is_fine ? m_fine : m_coarse;
        int n = is_fine ? n_fine : n_coarse;
        T x_frac = is_fine ? x_frac_f : x_frac_c;
        T y_frac = is_fine ? y_frac_f : y_frac_c;
        T shear_frac = is_fine ? shear_frac_f : shear_frac_c;

        auto assembler = createPlateDistortedAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp, 
            m, n, x_frac, y_frac, shear_frac);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        bool full_LU = true; // temp so we can see analyses
        // bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
        grids.push_back(*GRID::buildFromAssembler(assembler, my_loads, full_LU, true));
        i_level++;
    }

    // now run an analysis real quick.. writeout the solution..
    for (int i = 0; i < 2; i++) {
        grids[i].direct_solve();

        auto h_soln = grids[i].d_soln.createPermuteVec(6, grids[i].d_perm).createHostVec();
        printToVTK<Assembler, HostVec<T>>(grids[i].assembler, h_soln, "grid_" + std::to_string(i) + ".vtk");
    }

    // now try prolongation.. with unstruct mesh style (even though technically they are very different distorted struct meshes, still need same general method)
    // do this on the host first..
    

    return 0;
};
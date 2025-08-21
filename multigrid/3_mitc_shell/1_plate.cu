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
#include "include/grid.h"
#include "include/fea.h"
#include <string>

/* command line args:
    [direct/mg] [--nxe int] [--SR float]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void direct_plate_solve(int nxe, double SR) {
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

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bool full_LU = true;

    if (full_LU) {
        bsr_data.AMD_reordering();
        bsr_data.compute_full_LU_pattern(fillin, print);
    } else {
        /*
        RCM and reorderings actually hurt GMRES performance on the plate case
        because the matrix already has a nice banded structure => RCM increases bandwidth (which means it just doesn't work well for this problem
        as it's whole point is to decrease matrix bandwidth)
        */

        bsr_data.AMD_reordering();
        // bsr_data.RCM_reordering();
        // bsr_data.qorder_reordering(1.0);
        
        bsr_data.compute_ILUk_pattern(5, fillin);
        // bsr_data.compute_full_LU_pattern(fillin, print); // reordered full LU here for debug
    }
    // printf("perm:");
    // printVec<int>(bsr_data.nnodes, bsr_data.perm);
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    if (full_LU) {
        CUSPARSE::direct_LU_solve(kmat, loads, soln);
    } else {
        int n_iter = 200, max_iter = 400;
        T abs_tol = 1e-11, rel_tol = 1e-14;
        bool print = true;
        CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);

        // CUSPARSE::GMRES_DR_solve<T, false>(kmat, loads, soln, 50, 10, 65, abs_tol, rel_tol, true, false, 5);
    }

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}

void multigrid_plate_solve(int nxe, double SR) {
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

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    // make the shell grid
    auto grid = ShellGrid::buildFromAssembler<Assembler>(assembler, my_loads);

    // try doing multicolor block-GS iterations here (precursor to doing multgrid first)
    // int n_iters = 3;
    // int n_iters = 10;
    int n_iters = 1000;
    bool print = true;
    int print_freq = 1;
    // T omega = 1.0; // TODO : may need somewhat damping for higher SR?
    T omega = 0.7;
    grid.multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega);

    // multigrid solve here..

    // print some of the data of host residual
    auto h_soln = grid.d_soln.createPermuteVec(6, grid.Kmat.getIPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_mg.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = false;
    int nxe = 256; // default value
    double SR = 100.0; // default

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        }  else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value]" << std::endl;
            return 1;
        }
    }

    // done reading arts, now run stuff
    if (is_multigrid) {
        multigrid_plate_solve(nxe, SR);
    } else {
        direct_plate_solve(nxe, SR);
    }

    return 0;

    
}
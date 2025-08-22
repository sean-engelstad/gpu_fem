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
#include "include/mg.h"
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

void multigrid_plate_debug(int nxe, double SR) {
    // geometric multigrid method, debug individual steps on single grid here..

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
    using GRID = ShellGrid<PlateProlongation>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    // make the shell grid
    GRID *grid = GRID::buildFromAssembler<Assembler>(assembler, my_loads);

    // make a second grid here
    auto coarse_assembler = createPlateAssembler<Assembler>(nxe / 2, nxe / 2, Lx, Ly, E, nu, thick, rho, ys, 
        nxe_per_comp / 2, nye_per_comp / 2);
    T *my_coarse_loads = getPlateLoads<T, Physics>(nxe / 2, nye / 2, Lx, Ly, Q);

    // make the shell grid
    GRID *coarse_grid = GRID::buildFromAssembler<Assembler>(coarse_assembler, my_coarse_loads);

    // solve on coarse grid first..
    coarse_grid->direct_solve();
    auto h_coarse_soln = coarse_grid->d_soln.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_soln, "out/plate_coarse_direct.vtk");

    auto h_coarse_soln2 = coarse_grid->d_soln.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_soln2, "out/plate_coarse_direct2.vtk");

    // DEBUG
    // int *h_c_iperm = DeviceVec<int>(coarse_grid->nnodes, coarse_grid->d_iperm).createHostVec().getPtr();
    // printf("h_ coarse iperm: ");
    // printVec<int>(coarse_grid->nnodes, h_c_iperm);
    // return;

    // fine defect here..
    auto h_fine_defect = grid->d_defect.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_fine_defect, "out/plate_fine_defect_0.vtk");

    // try prolongation
    grid->prolongate(coarse_grid->d_iperm, coarse_grid->d_soln);

    // print some of the data of host residual
    auto h_soln = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_cf_soln.vtk");

    // fine defect here..
    auto h_fine_defect1 = grid->d_defect.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_fine_defect1, "out/plate_fine_defect_1.vtk");

    // // does printing soln again change it?
    // auto h_soln3 = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assembler, h_soln3, "out/plate_cf_soln2.vtk");

    // try defect restriction
    coarse_grid->restrict_defect(grid->nelems, grid->d_elem_conn, grid->d_iperm,
                        grid->d_defect);

    // print some of the data of host residual
    auto h_coarse_defect = coarse_grid->d_defect.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_defect, "out/plate_fc_defect.vtk");

    // // try doing multicolor block-GS iterations here (precursor to doing multgrid first)
    // int n_iters = 3;
    int n_iters = 10;
    // int n_iters = 1000;
    bool print = true;
    int print_freq = 1;
    T omega = 1.0; // TODO : may need somewhat damping for higher SR?
    // T omega = 0.7; // only seem to need damping for very small DOF (for full solve, still smoothes otherwise)
    grid->multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega);

    // print some of the data of host residual
    auto h_soln2 = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln2, "out/plate_mg.vtk");
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

    // multigrid objects
    using Prolongation = PlateProlongation;
    using GRID = ShellGrid<Prolongation>;
    using MG = ShellMultigrid<GRID>;

    Assembler *fine_assembler;
    auto mg = MG();

    // make each grid
    for (int c_nxe = nxe; c_nxe >= 4; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);

        if (c_nxe == nxe) {
            fine_assembler = &assembler;
        }
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        auto grid = *GRID::buildFromAssembler<Assembler>(assembler, my_loads);
        mg.grids.push_back(grid); // add new grid
    }

    printf("starting v cycle solve\n");
    int pre_smooth = 2, post_smooth = 2;
    // int n_vcycles = 30;
    int n_vcycles = 3;
    // bool print = false;
    bool print = true;
    T atol = 1e-6, rtol = 1e-6;
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, atol, rtol);
    printf("done with v-cycle solve\n");


    // print some of the data of host residual
    int *d_iperm = mg.grids[0].Kmat.getIPerm();
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_iperm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(*fine_assembler, h_soln, "out/plate_mg.vtk");
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
        // multigrid_plate_debug(nxe, SR);
    } else {
        direct_plate_solve(nxe, SR);
    }

    return 0;

    
}
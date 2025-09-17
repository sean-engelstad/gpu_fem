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

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
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

void direct_solve(int nxe, double SR) {
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

    int nhe = nxe;
    double L = 1.0, R = 0.5, thick = L / SR;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto assembler = createCylinderAssembler<Assembler>(nxe, nhe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);

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
    constexpr bool compressive = false;
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T, Physics, compressive>(nxe, nhe, L, R, Q);

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
    }

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/cylinder.vtk");
}

void multigrid_solve(int nxe, double SR, int n_vcycles) {
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
    // const SMOOTHER smoother = MULTICOLOR_GS;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;
    
    using Prolongation = StructuredProlongation<CYLINDER>;
    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();

    auto mg = MG();

    // int nxe_min = 4;
    // int nxe_min = nxe / 2; // two level
    int nxe_min = 32; // maybe really coarse cylinders have such horrible discretization (like diamond)
    // that it affects the ovr conv, so make min size much larger..

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nhe = c_nxe;
        double L = 1.0, R = 0.5, thick = L / SR;
        double E = 70e9, nu = 0.3;
        // double rho = 2500, ys = 350e6;
        bool imperfection = false; // option for geom imperfection
        int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
        auto assembler = createCylinderAssembler<Assembler>(c_nxe, c_nhe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
        constexpr bool compressive = false;
        double Q = 1.0; // load magnitude
        T *my_loads = getCylinderLoads<T, Physics, compressive>(c_nxe, c_nhe, L, R, Q);
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2) {
            reorder = true;
        }
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    // // bool debug = false;
    // bool debug = true;
    // if (debug) {
    //     int *d_perm1 = mg.grids[0].d_perm;
    //     int *d_perm2 = mg.grids[1].d_perm;
    //     mg.grids[1].direct_solve(false);
    //     auto h_solnc1 = mg.grids[1].d_soln.createPermuteVec(6, d_perm2).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_solnc1, "out/cylinder_coarse_soln.vtk");

    //     mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    //     auto h_soln1 = mg.grids[0].d_temp_vec.createPermuteVec(6, d_perm1).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln1, "out/cylinder_mg_cf.vtk");

    //     // plot orig fine defect
    //     auto h_fdef = mg.grids[0].d_defect.createPermuteVec(6, d_perm1).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fdef, "out/cylinder_mg_fine_defect.vtk");

    //     // now try restrict defect
    //     mg.grids[1].restrict_defect(
    //                         mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);
    //     auto h_def2 = mg.grids[1].d_defect.createPermuteVec(6, d_perm2).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_def2, "out/cylinder_mg_restrict.vtk");
    //     return;
    // }  

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting v cycle solve\n");
    // int pre_smooth = 1, post_smooth = 1;
    int pre_smooth = 1, post_smooth = 2;
    // int pre_smooth = 2, post_smooth = 2; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    // int pre_smooth = 4, post_smooth = 4;
    // bool print = true;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    T omega = 1.0;
    // bool double_smooth = false;
    bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, atol, rtol, omega, double_smooth);
    // 0 input means starts on outer level (fine grid W-cycle)
    // mg.wcycle_solve(0, pre_smooth, post_smooth, n_vcycles, print, atol, rtol, omega); // try stronger W-cycle solve here on cylinder
    printf("done with v-cycle solve\n");

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    printf("cylinder GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/cylinder_mg.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = false;
    int nxe = 256; // default value
    double SR = 100.0; // default
    int n_vcycles = 50;

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
        } else if (strcmp(arg, "--nvcyc") == 0) {
            if (i + 1 < argc) {
                n_vcycles = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nvcyc\n";
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
        multigrid_solve(nxe, SR, n_vcycles);
    } else {
        direct_solve(nxe, SR);
    }

    return 0;

    
}
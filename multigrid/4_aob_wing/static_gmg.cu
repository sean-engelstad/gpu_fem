
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/fea.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

/* argparse options:
[mg/direct/debug] [--level int]
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

std::string time_string(int itime) {
    std::string _time = std::to_string(itime);
    if (itime < 10) {
        return "00" + _time;
    } else if (itime < 100) {
        return "0" + _time;
    } else {
        return _time;
    }
}

void solve_linear_multigrid(MPI_Comm &comm, int level, double SR, int nsmooth) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

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

    // old smoothers
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;
    // const SMOOTHER smoother = MULTICOLOR_GS;
    // const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = MULTICOLOR_GS_FAST2; // fastest (faster than MULTICOLOR_GS_FAST by about 2.6x at high DOF)
    // const SMOOTHER smoother = DAMPED_JACOBI;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2_JUNCTION;

    const SCALER scaler = LINE_SEARCH;

    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Basis, is_bsr>; 

    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    using MG = GeometricMultigridSolver<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();

    // make each wing multigrid object.. (highest mesh level is finest, this is flipped from MG object's convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.01;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // TODO : run optimized design from AOB case

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2 
            || smoother == MULTICOLOR_GS_FAST2_JUNCTION) {
            reorder = true;
        } else if (smoother == DAMPED_JACOBI) {
            reorder = false;
        }
        // printf("reorder %d\n", reorder);
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        printf("begin unstructured map\n");
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = mg.grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting v cycle solve\n");
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    // best was V(4,4) before
    // bool print = false;
    bool print = true;
    T atol = 1e-6, rtol = 1e-6;
    T omega = 1.5; // good GS-SSOR parameter (speedups up convergence)
    // T omega = 1.0;
    // T omega = 0.85;
    if (smoother == LEXIGRAPHIC_GS) omega = 1.4;
    if (smoother == DAMPED_JACOBI) omega = 0.7; // damped jacobi diverges on wingbox
    int n_cycles = 200;
    if (SR > 100.0) n_cycles = 1000;

    bool time = false;
    // bool time = true;

    bool symmetric = false;
    // bool symmetric = true;

    // int print_freq = 1;
    int print_freq = 5;

    // bool double_smooth = false;
    bool double_smooth = true; // false
    mg.vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, omega, double_smooth, print_freq, symmetric, time);
    // mg.wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, omega);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = mg.get_memory_usage_mb();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    // double check with true resid nrm
    T resid_nrm = mg.grids[0].getResidNorm();
    printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
}

void solve_linear_direct(MPI_Comm &comm, int level, double SR) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "meshes/aob_wing_L" + std::to_string(level) + ".bdf";
  mesh_loader.scanBDFFile(fname.c_str());

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
  using GRID = ShellGrid<Assembler, UnstructuredProlongation<Basis,true>, MULTICOLOR_GS_FAST2_JUNCTION, NONE>;

  //   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
  double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // TODO : set this in from optimized design from AOB case

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();

//   // TRY INSTEAD Mc REORDERING
//   int num_colors, *_color_rowp, *nodal_num_comps, *node_geom_ind;
//   GRID::get_nodal_geom_indices(assembler, nodal_num_comps, node_geom_ind);
//   bsr_data.multicolor_junction_reordering_v2(node_geom_ind, num_colors, _color_rowp);

  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 10.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start1 = std::chrono::high_resolution_clock::now();

  // solve the linear system
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end1 - start1;

  size_t bytes_per_double = sizeof(double);
  double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
  printf("direct LU solve uses memory(MB) %.2e in %.2e sec\n", mem_mb, solve_time.count());

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/aob_direct_L" + std::to_string(level) + ".vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 0; // level mesh to solve..
    bool is_multigrid = true;
    // bool is_debug = false;
    double SR = 50.0;
    // int nsmooth = 4;
    int nsmooth = 6; // typically faster right now

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--SR double] [--nsmooth int]" << std::endl;
            return 1;
        }
    }

    // solve linear with directLU solve
    if (is_multigrid) { // && !is_debug) {
        solve_linear_multigrid(comm, level, SR, nsmooth);
    // } else if (is_multigrid && is_debug) {
    //     solve_linear_multigrid_debug(comm, level, SR);
    } else {
        solve_linear_direct(comm, level, SR);
    }

    // TBD multigrid solve..

    MPI_Finalize();
    return 0;
};

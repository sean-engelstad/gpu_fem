
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// local multigrid imports
#include "include/grid.h"
#include "include/fea.h"
#include "include/mg.h"
#include <string>
#include <chrono>

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void solve_linear_multigrid(MPI_Comm &comm, int level) {
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

    // multigrid objects
    // const SMOOTHER smoother = MULTICOLOR_GS;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    using Prolongation = UnstructuredProlongation<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    std::vector<GRID> direct_grids;

    // make each wing multigrid object.. (highest mesh level is finest, this is flipped from MG object's convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/naca_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
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

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
            reorder = true;
        }
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid

        if (i == level) {
            // also makethe true fine grid
            TACSMeshLoader mesh_loader2{comm};
            mesh_loader2.scanBDFFile(fname.c_str());
            auto assembler2 = Assembler::createFromBDF(mesh_loader2, Data(E, nu, thick));
            auto direct_fine_grid = *GRID::buildFromAssembler(assembler2, my_loads, true, true);
            direct_grids.push_back(direct_fine_grid); 
        }
    }

    if (!Prolongation::structured) {
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        // printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    // -------------------------------
    bool debug = false;
    // bool debug = true;

    if (debug) {
        // check n2e ptr for fine nodes 4909, 1098, 1097
        // int fine_nodes[7] = {4912, 1099, 1100, 4910, 4909, 1098, 1097};
        // int fine_nodes[4] = {14803, 15947, 15948, 15977};
        // int fine_nodes[2] = {11132, 11133};
        // int fine_nodes[3] = {24064, 24095, 24125};
        // int fine_nodes[2] = {1539, 5316};
        int fine_nodes[3] = {11102, 14081, 14064};

        GRID &fine_grid = mg.grids[0];
        int n2e_nnz = fine_grid.n2e_nnz;
        int *h_n2e_ptr = DeviceVec<int>(fine_grid.nnodes + 1, fine_grid.d_n2e_ptr).createHostVec().getPtr();
        int *h_n2e_elems = DeviceVec<int>(n2e_nnz, fine_grid.d_n2e_elems).createHostVec().getPtr();
        T *h_n2e_xis = DeviceVec<T>(2 * n2e_nnz, fine_grid.d_n2e_xis).createHostVec().getPtr();
        int ncoarse_elems = fine_grid.ncoarse_elems;
        int *h_coarse_conn = DeviceVec<int>(4 * ncoarse_elems, fine_grid.d_coarse_conn).createHostVec().getPtr();

        for (int i = 0; i < 3; i++) {
            int _fine_node = fine_nodes[i];
            // int fine_node = _fine_node - 1;
            int fine_node = _fine_node;
            int n_celems = h_n2e_ptr[fine_node + 1] - h_n2e_ptr[fine_node];
            printf("fine node %d with %d connected celems\n", _fine_node, n_celems);

            for (int jp = h_n2e_ptr[fine_node]; jp < h_n2e_ptr[fine_node + 1]; jp++) {
                int celem = h_n2e_elems[jp];
                T xi = h_n2e_xis[2 * jp], eta = h_n2e_xis[2 * jp + 1];
                int k = jp - h_n2e_ptr[fine_node];
                printf("\tcelem %d/%d = %d has xi %.2e, eta %.2e and cnodes ", k + 1, n_celems, celem, xi, eta);

                int *loc_elem_conn = &h_coarse_conn[4 * celem];
                for (int elem_node = 0; elem_node < 4; elem_node++) {
                    int cnode = loc_elem_conn[elem_node];
                    printf("%d ", cnode);
                }
                printf("\n");
            }
        }

        // print the number of celems for each fine node to file..
        T *h_n2e_cts = new T[6 * fine_grid.nnodes];
        for (int inode = 0; inode < fine_grid.nnodes; inode++) {
            for (int idof = 0; idof < 6; idof++) {
                h_n2e_cts[6 * inode + idof] = h_n2e_ptr[inode + 1] - h_n2e_ptr[inode];
            }
        }
        auto h_n2e_cts_vec = HostVec<T>(fine_grid.N, h_n2e_cts);
        printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_n2e_cts_vec, "out/wing_num_celems.vtk");
    }


    // TEMP DEBUG unstructured
    if (debug) {
        // mg.grids[1].restrict_defect(
        //                     mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);

        mg.grids[1].direct_solve(false);
        int n_smooth = 0; // regular prolong
        // int n_smooth = 1;
        // int n_smooth = 3;
        mg.grids[0].prolongate_debug(mg.grids[1].d_iperm, mg.grids[1].d_soln, "out/wing_", ".vtk", n_smooth);
        int *d_perm1 = mg.grids[0].d_perm;
        auto h_soln_mg = mg.grids[0].d_soln.createPermuteVec(6, d_perm1).createHostVec();

        printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln_mg, "out/wing_soln_update.vtk");     

        // compare to true fine grid soln
        direct_grids[0].direct_solve(false);
        int *d_perm2 = direct_grids[0].d_perm;
        auto h_true_soln = direct_grids[0].d_soln.createPermuteVec(6, d_perm2).createHostVec();
        printToVTK<Assembler,HostVec<T>>(direct_grids[0].assembler, h_true_soln, "out/wing_true_soln.vtk");    


        // somehow compare the soln update to the true host solution?
        return;
    }

    // if (debug) {
    //     int *d_perm1 = mg.grids[0].d_perm;
    //     int *d_perm2 = mg.grids[1].d_perm;

    //     // plot orig fine defect
    //     auto h_fdef = mg.grids[0].d_defect.createPermuteVec(6, d_perm1).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fdef, "out/wing_mg_fine_defect.vtk");

    //     // now try restrict defect
    //     mg.grids[1].restrict_defect(
    //                         mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);
    //     auto h_def2 = mg.grids[1].d_defect.createPermuteVec(6, d_perm2).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_def2, "out/wing_mg_restrict.vtk");

    //     // coarse solve
    //     mg.grids[1].direct_solve(false);
    //     auto h_solnc1 = mg.grids[1].d_soln.createPermuteVec(6, d_perm2).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_solnc1, "out/wing_coarse_soln.vtk");

    //     // prolongate
    //     mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    //     auto h_soln1 = mg.grids[0].d_temp_vec.createPermuteVec(6, d_perm1).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln1, "out/wing_mg_cf.vtk");     
        
    //     // plot new fine defect
    //     auto h_fdef2 = mg.grids[0].d_defect.createPermuteVec(6, d_perm1).createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fdef2, "out/wing_mg_new_fine_defect.vtk");
        
    //     return;
    // }  
    // -------------------------------

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = mg.grids[0].getResidNorm();

    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting v cycle solve\n");
    // int pre_smooth = 1, post_smooth = 1;
    int pre_smooth = 2, post_smooth = 2;
    // int pre_smooth = 4, post_smooth = 4;
    // bool print = false;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    T omega = 1.0;
    // T omega = 0.8; // may need lower omega to handle junctions better? less magnification there?
    // T omega = 0.7;
    // T omega = 0.1;
    // int n_vcycles = 50;
    int n_vcycles = 100;

    bool double_smooth = false;
    // bool double_smooth = true; // false
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, atol, rtol, omega, double_smooth);
    printf("done with v-cycle solve\n");

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // double check with true resid nrm
    T resid_nrm = mg.grids[0].getResidNorm();
    printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/wing_mg.vtk");
}

void solve_linear_direct(MPI_Comm &comm, int level) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "meshes/naca_wing_L" + std::to_string(level) + ".bdf";
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

  double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  int ndvs = assembler.get_num_dvs();
  printf("ndvs %d\n", ndvs);
  T thick2 = 1e-2;
  HostVec<T> h_dvs(ndvs, thick2);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // temp debug, double check bcs
//   auto d_bcs_vec = assembler.getBCs();
//   int n_bcs = d_bcs_vec.getSize();
//   int *h_bcs = d_bcs_vec.createHostVec().getPtr();
//   printf("# bcs %d, bcs: ", n_bcs);
//   printVec<int>(n_bcs, h_bcs);

  // T mass = assembler._compute_mass();
  // printf("mass %.4e\n", mass);

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
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

  // solve the linear system
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/naca_direct_L" + std::to_string(level) + ".vtk");

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

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level value]" << std::endl;
            return 1;
        }
    }

    // solve linear with directLU solve
    if (is_multigrid) {
        solve_linear_multigrid(comm, level);
    } else {
        solve_linear_direct(comm, level);
    }

    // TBD multigrid solve..

    MPI_Finalize();
    return 0;
};

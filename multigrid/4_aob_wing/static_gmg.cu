
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
#include "multigrid/mg.h"
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

    // using Prolongation = UnstructuredProlongation<Basis>;
    using Prolongation = UnstructuredProlongationFast<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    // std::vector<GRID> direct_grids;

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

        // TODO : get optimized design from NACA case
        // // set reasonable design variables (optional, otherwise const thick..)
        // int ndvs = assembler.get_num_dvs(); // 32 components
        // // TODO : make thinner later

        // // internal struct and skin/OML thicknesses
        // T its_thick = 0.5666 / SR, skin_thick = 0.5666 / SR;
        // // T its_thick = 0.1, skin_thick = 1.0;
        // // T its_thick = 0.008, skin_thick = 0.03;
        // // T its_thick = 0.001, skin_thick = 0.01;

        // bool is_int_struct[32] = {1, 1, 0, 1,   0, 0, 0, 1,   1, 1, 0, 1,   0, 0, 0, 1,
        //     1, 0, 0, 1,   0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 1 };
        // T *h_dvs_ptr = new T[32];
        // for (int j = 0; j < 32; j++) {
        //     if (is_int_struct[j]) {
        //         h_dvs_ptr[j] = its_thick;
        //     } else {
        //         h_dvs_ptr[j] = skin_thick;
        //     }
        // }
        // auto h_dvs = HostVec<T>(32, h_dvs_ptr);
        // auto global_dvs = h_dvs.createDeviceVec();
        // assembler.set_design_variables(global_dvs);

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
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    // T omega = 2.0;
    // T omega = 1.8;
    // T omega = 1.7;
    // T omega = 1.6;
    T omega = 1.5;
    // T omega = 1.4;
    // T omega = 1.3;
    // T omega = 1.2;
    // T omega = 1.0;
    // T omega = 0.85;
    if (smoother == LEXIGRAPHIC_GS) omega = 1.4;
    if (smoother == DAMPED_JACOBI) omega = 0.7; // damped jacobi diverges on wingbox
    int n_cycles = 200;

    bool time = false;
    // bool time = true;

    // bool double_smooth = false;
    bool double_smooth = true; // false
    mg.vcycle_solve(pre_smooth, post_smooth, n_cycles, print, atol, rtol, omega, double_smooth, time);
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

void solve_linear_multigrid_debug(MPI_Comm &comm, int level, double SR) {
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
    // const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = MULTICOLOR_GS_FAST2;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2_JUNCTION;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    const SCALER scaler = LINE_SEARCH;

    // using Prolongation = UnstructuredProlongation<Basis>;
    using Prolongation = UnstructuredProlongationFast<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    std::vector<GRID> direct_grids;

    // make each wing multigrid object.. (highest mesh level is finest, this is flipped from MG object's convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)

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

        // TODO : get optimized design from NACA case..
        // // set reasonable design variables (optional, otherwise const thick..)
        // int ndvs = assembler.get_num_dvs(); // 32 components
        // // TODO : make thinner later
        // T its_thick = 0.5666 / SR, skin_thick = 0.5666 / SR;
        // // T its_thick = 0.1; // (internal struct thick - ribs / spars)
        // // T skin_thick = 1.0;
        // bool is_int_struct[32] = {1, 1, 0, 1,   0, 0, 0, 1,   1, 1, 0, 1,   0, 0, 0, 1,
        //     1, 0, 0, 1,   0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 1 };
        // T *h_dvs_ptr = new T[32];
        // for (int j = 0; j < 32; j++) {
        //     if (is_int_struct[j]) {
        //         h_dvs_ptr[j] = its_thick;
        //     } else {
        //         h_dvs_ptr[j] = skin_thick;
        //     }
        // }
        // auto h_dvs = HostVec<T>(32, h_dvs_ptr);
        // auto global_dvs = h_dvs.createDeviceVec();
        // assembler.set_design_variables(global_dvs);

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
            // reorder = true; // leads to RCM reordering
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

    // bool pre_debug = false;
    // // bool pre_debug = true;

    // if (pre_debug) {
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

    // also compute and printout which nodes are which color
    auto h_color_rowp = mg.grids[0].h_color_rowp;
    int *d_perm = mg.grids[0].d_perm;
    int *h_perm = DeviceVec<int>(fine_grid.nnodes, d_perm).createHostVec().getPtr();
    int num_colors = h_color_rowp.getSize() - 1;
    T *h_node_colors = new T[6 * fine_grid.nnodes]; // unpermuted
    int N = mg.grids[0].N;
    for (int icolor = 0; icolor < num_colors; icolor++) {
        for (int jp = h_color_rowp[icolor]; jp < h_color_rowp[icolor + 1]; jp++) {
            int perm_inode = jp;
            int inode = h_perm[perm_inode];

            for (int idof = 0; idof < 6; idof++) {
                int ind = 6 * inode + idof;
                h_node_colors[ind] = icolor;
            }
        }
    }
    auto h_node_colors_vec = HostVec<T>(fine_grid.N, h_node_colors);
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_node_colors_vec, "out/wing_colors.vtk");

    //     return;
    // }

    bool write = true;
    if (write) {
        auto h_fine_defectn1 = mg.grids[0].d_defect.createPermuteVec(6, mg.grids[0].Kmat.getPerm()).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fine_defectn1, "out/0_wing_fine_defect0.vtk");
    }

    // // TEMP debug, test prolongate matrix-vec..
    // printf("test prolongate\n");
    // mg.grids[1].direct_solve(false); // false for don't print
    // mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    // auto h_soln_debug = mg.grids[0].d_soln.createPermuteVec(6, mg.grids[0].Kmat.getPerm()).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln_debug, "out/0_test_prolong.vtk");

    printf("starting v cycle solve\n");
    // init defect nrm
    T init_defect_nrm = mg.grids[0].getDefectNorm();
    printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

    T atol = 1e-6, rtol = 1e-6;
    // int pre_smooth = 1, post_smooth = 1;
    int pre_smooth = 2, post_smooth = 2;
    bool print = true;
    T omega = 1.0;

    // int n_vcycles = 5;
    int n_vcycles = 50;

    int n_steps = n_vcycles;
    T fin_defect_nrm = init_defect_nrm;

    int n_levels = 2;

    for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
        // printf("V cycle step %d\n", i_vcycle);
        std::string file_suffix = "_" + time_string(i_vcycle) + ".vtk";

        // auto h_fine_defectn1 = mg.grids[0].d_defect.createPermuteVec(6, mg.grids[0].Kmat.getIPerm()).createHostVec();
        // printToVTK<Assembler,HostVec<T>>(assemblers[0], h_fine_defectn1, "out/0_plate_fine_defect0.vtk");

        // go down each level smoothing and restricting until lowest level
        for (int i_level = 0; i_level < n_levels; i_level++) {

            std::string file_prefix = "out/level" + std::to_string(i_level) + "_";

            // if not last  (pre-smooth)
            if (i_level < n_levels - 1) {
                // if (print) printf("\tlevel %d pre-smooth\n", i_level);

                // prelim defect
                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 0.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "pre1_start" + file_suffix, xpts_shift);
                }

                // pre-smooth; TODO : do fast version later.. but let's demo with slow version
                // first
                printf("GS on level %d\n", i_level);
                mg.grids[i_level].smoothDefect(pre_smooth, print, pre_smooth - 1, omega);

                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 1.5};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "pre2_smooth" + file_suffix, xpts_shift);
                }

                // restrict defect
                printf("restrict defect from level %d => %d\n", i_level, i_level + 1);
                mg.grids[i_level + 1].restrict_defect(
                    mg.grids[i_level].nelems, mg.grids[i_level].d_iperm,
                    mg.grids[i_level].d_defect);
                CHECK_CUDA(cudaDeviceSynchronize()); // needed to make this work right?

                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level+1].d_defect.createPermuteVec(6, mg.grids[i_level+1].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 3.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level+1].assembler, h_fine_defect00, file_prefix + "pre3_restrict" + file_suffix, xpts_shift);
                }

            } else {
                if (print) printf("\t--level %d full-solve\n", i_level);

                // prelim defect
                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 0.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "full_pre1_start" + file_suffix, xpts_shift);
                }

                // coarsest grid full solve
                mg.grids[i_level].direct_solve(false); // false for don't print

                // prelim soln
                if (write) {
                    auto h_soln = mg.grids[i_level].d_soln.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 1.5};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "full_pre2_soln" + file_suffix, xpts_shift);
                }
            }
        }

        // now go back up the hierarchy
        for (int i_level = n_levels - 2; i_level >= 0; i_level--) {

            std::string file_prefix = "out/level" + std::to_string(i_level) + "_";

            // prelim defect
            if (write) {
                auto h_soln = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -6, 0.0};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "post1_defect" + file_suffix, xpts_shift);
            }

            // get coarse-fine correction from coarser grid to this grid
            mg.grids[i_level].prolongate_debug(mg.grids[i_level + 1].d_iperm, mg.grids[i_level + 1].d_soln, file_prefix, file_suffix, 0, -6);
            if (print) printf("\tlevel %d post-smooth\n", i_level);

            CHECK_CUDA(cudaDeviceSynchronize()); // TODO : needed to make this work right?, adding write statements improved conv..

            if (write) {
                auto h_soln = mg.grids[i_level].d_soln.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -6, 6.0};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "post5_soln" + file_suffix, xpts_shift);
            }

            // post-smooth
            bool rev_colors = true; // rev colors only on post not pre-smooth?
            // bool rev_colors = false;
            mg.grids[i_level].smoothDefect(post_smooth, print, post_smooth - 1, omega, rev_colors); 

            if (write) {
                auto h_defect = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -6, 7.5};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_defect, file_prefix + "post6_postsmooth_defect" + file_suffix, xpts_shift);
            }
        }

        // compute fine grid defect of V-cycle
        T defect_nrm = mg.grids[0].getDefectNorm();
        printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);
        fin_defect_nrm = defect_nrm;

        if (defect_nrm < atol + rtol * init_defect_nrm) {
            printf("V-cycle GMG converged in %d steps\n", i_vcycle + 1);
            n_steps = i_vcycle + 1;
            break;
        }
    }
    printf("done with v-cycle solve, conv %.2e to %.2e ||defect|| in %d steps\n", init_defect_nrm, fin_defect_nrm, n_steps);
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

//   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

//   // TODO : set this in from optimized design (this is from NACA case)
//   // internal struct and skin/OML thicknesses
//   T its_thick = 0.5666 / SR, skin_thick = 0.5666 / SR;
//     // T its_thick = 0.1, skin_thick = 1.0;
//     // T its_thick = 0.01, skin_thick = 0.1;
//     // T its_thick = 0.001, skin_thick = 0.01;

//     bool is_int_struct[32] = {1, 1, 0, 1,   0, 0, 0, 1,   1, 1, 0, 1,   0, 0, 0, 1,
//         1, 0, 0, 1,   0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 1 };
//     T *h_dvs_ptr = new T[32];
//     for (int j = 0; j < 32; j++) {
//         if (is_int_struct[j]) {
//             h_dvs_ptr[j] = its_thick;
//         } else {
//             h_dvs_ptr[j] = skin_thick;
//         }
//     }
//     auto h_dvs = HostVec<T>(32, h_dvs_ptr);
//     auto global_dvs = h_dvs.createDeviceVec();
//     assembler.set_design_variables(global_dvs);

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

  size_t bytes_per_double = sizeof(double);
  double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
  printf("direct LU solve uses memory(MB) %.2e\n", mem_mb);

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
    bool is_debug = false;
    double SR = 50.0;
    int nsmooth = 4;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "debug") == 0) {
            is_debug = true;
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
    if (is_multigrid && !is_debug) {
        solve_linear_multigrid(comm, level, SR, nsmooth);
    } else if (is_multigrid && is_debug) {
        solve_linear_multigrid_debug(comm, level, SR);
    } else {
        solve_linear_direct(comm, level, SR);
    }

    // TBD multigrid solve..

    MPI_Finalize();
    return 0;
};

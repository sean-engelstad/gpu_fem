// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/mitc_shell.h"

// local multigrid imports
// #include "multigrid/grid.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/fea.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include <string>
#include <chrono>

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void test_direct_solver_class(int nxe, double SR = 100.0) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // direct solver class
    using DirectSolver = CusparseMGDirectLU<T, Assembler>;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // make a single grid
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nxe/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q);
    printf("making grid with nxe %d\n", nxe);

    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;

    // make the grid with multicolor ordering
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();

    // bsr_data.AMD_reordering();
    // bsr_data.compute_full_LU_pattern(10.0, false);
    // assembler.moveBsrDataToDevice();
    
    // make loads and assemble kmat
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    int N = res.getSize();

    // assemble the kmat
    auto start1 = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end1 - start1;
    printf("\tassemble kmat time %.2e\n", assembly_time.count());

    // try and do direct solve
    printf("direct solve v0\n");
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_direct0.vtk");
    // return; // temp debug

    // permute the loads to right order (of direct solver)
    int block_dim = bsr_data.block_dim;
    int *d_iperm = kmat.getIPerm();
    loads.permuteData(block_dim, d_iperm);

    // make the direct solver
    printf("make direct solver v1\n");
    auto solver = DirectSolver(assembler, kmat);
    printf("perform direct solver v1\n");
    solver.solve(loads, soln);

    // write the solution file
    int *d_perm = kmat.getPerm();
    soln.permuteData(block_dim, d_perm);
    auto h_soln2 = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln2, "out/plate_direct1.vtk");
}

void test_smoother(int nxe, double SR = 100.0) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // smoother class
    using Smoother = MulticolorGSSmoother_V1<Assembler>;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // make a single grid
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nxe/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q);
    printf("making grid with nxe %d\n", nxe);

    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;

    // make the grid with multicolor ordering
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();

    // bsr_data.AMD_reordering();
    // bsr_data.compute_full_LU_pattern(10.0, false);
    // assembler.moveBsrDataToDevice();
    
    // make loads and assemble kmat
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    int N = res.getSize();

    // assemble the kmat
    auto start1 = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end1 - start1;
    printf("\tassemble kmat time %.2e\n", assembly_time.count());

    // try and do direct solve
    // CUSPARSE::direct_LU_solve(kmat, loads, soln);
    // auto h_soln = soln.createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
    // return; // temp debug

    // make the smoother
    // T omega = 1.5;
    T omega = 0.7;
    auto smoother = Smoother(assembler, kmat, h_color_rowp, omega);
    auto defect = assembler.createVarsVec();

    // copy loads to defect and permute it
    loads.copyValuesTo(defect);
    int block_dim = bsr_data.block_dim, *d_iperm = kmat.getIPerm();
    defect.permuteData(block_dim, d_iperm);

    // now try and do smoothing solve
    int n_iters = 30, print_freq = 1;
    bool print = true;
    smoother.smoothDefect(defect, soln, n_iters, print, print_freq);

    int *d_perm = kmat.getPerm();
    defect.permuteData(block_dim, d_perm);
    auto h_defect = defect.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_defect, "out/plate_smooth.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    int nxe = 256; // default value

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // done reading arts, now run stuff
    // test_smoother(nxe);
    test_direct_solver_class(nxe);

    return 0;

    
}
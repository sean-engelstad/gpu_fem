// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_static_gmg.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_static_gmg.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}


template <typename T, class Assembler>
void direct_plate_solve(int nxe, double SR, int time_mode = 0) {
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using LUsolver = CusparseMGDirectLU<T, Assembler>;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    auto start0 = std::chrono::high_resolution_clock::now();
    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    if (time_mode < 2) {
        bsr_data.AMD_reordering();
        bsr_data.compute_full_LU_pattern(fillin, print);
    } else {
        bsr_data.compute_nofill_pattern();
    }
    
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Basis, Physics>(nxe, nye, Lx, Ly, Q);

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    auto lu_solver = new LUsolver(cublasHandle, cusparseHandle, assembler, kmat);
    lu_solver->factor();

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    // lu_solver->printTriSolveVsMatVecTiming(loads, soln, 3, 
    //     just_tri_solve, !just_tri_solve);

    lu_solver->printTriSolveVsMatVecTiming_host(loads, soln, 3, 
        time_mode == 0, time_mode == 1, time_mode == 2);

        

    // solve the linear system
    // CUSPARSE::direct_LU_solve(kmat, loads, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    // printf("plate direct solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem (MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_mb);


    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    // bool is_multigrid = false;
    int nxe = 10;
    double SR = 1e3;
    std::string elem_type = "MITC4";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        // if (strcmp(arg, "direct") == 0) {
        //     is_multigrid = false;
        // } else if (strcmp(arg, "mg") == 0) {
        //     is_multigrid = true;
        // } else 
        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        // } 
        // else if (strcmp(arg, "--cycle") == 0) {
        //     if (i + 1 < argc) {
        //         cycle_type = argv[++i];
        //     } else {
        //         std::cerr << "Missing value for --level\n";
        //         return 1;
        //     }
        // } else if (strcmp(arg, "--elem") == 0) {
        //     if (i + 1 < argc) {
        //         elem_type = argv[++i];
        //     } else {
        //         std::cerr << "Missing value for --elem\n";
        //         return 1;
        //     }
        // } else if (strcmp(arg, "--nsmooth") == 0) {
        //     if (i + 1 < argc) {
        //         nsmooth = std::atoi(argv[++i]);
        //     } else {
        //         std::cerr << "Missing value for --nsmooth\n";
        //         return 1;
        //     }
        // } else if (strcmp(arg, "--ninnercyc") == 0) {
        //     if (i + 1 < argc) {
        //         ninnercyc = std::atoi(argv[++i]);
        //     } else {
        //         std::cerr << "Missing value for --nsmooth\n";
        //         return 1;
        //     }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
   
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // printf("plate mesh with %s elements, nxe %d and SR %.2e\n------------\n", elem_type.c_str(), nxe, SR);
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    direct_plate_solve<T, Assembler>(nxe, SR, 0);
    direct_plate_solve<T, Assembler>(nxe, SR, 1);
    direct_plate_solve<T, Assembler>(nxe, SR, 2);
    

    return 0;

    
}
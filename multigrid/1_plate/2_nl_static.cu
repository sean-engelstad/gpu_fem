#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "multigrid/utils/fea.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

/* command line args:
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <bool is_nonlinear>
void solve_direct(int nxe) {
    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR factorization
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get plate loads
    double Q = 5e2;
    // double Q = 1e2;
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    // double in_plane_frac = 0.3;
    // T *my_loads = getPlateNonlinearLoads<T, Physics>(nxe, nye, Lx, Ly, Q, in_plane_frac);
    // T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    if constexpr (is_nonlinear) {
        // newton solve
        int num_load_factors = 100, num_newton = 20;
        T min_load_factor = 0.01, max_load_factor = 1.0, abs_tol = 1e-6,
            rel_tol = 1e-6;
        auto solve_func = CUSPARSE::direct_LU_solve<T>;
        std::string outputPrefix = "out/plate_";

        // const bool fast_assembly = true;
        const bool fast_assembly = false;
        newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
            solve_func, kmat, loads, soln, assembler, res, rhs, vars,
            num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
            rel_tol, outputPrefix, print);

        // print some of the data of host residual
        auto h_soln = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate_nl.vtk");

    } else {
        // assembly
        assembler.add_jacobian(res, kmat);
        assembler.apply_bcs(kmat);

        // direct LU solve
        CUSPARSE::direct_LU_solve(kmat, loads, soln);

        // print some of the data of host residual
        auto h_soln = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate_lin.vtk");
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool use_multigrid = false;
    bool full_LU = true;
    int nxe = 10;  // default value
    // int nxe = 64;

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
            std::cerr << "Usage: " << argv[0] << " [linear|nonlinear] [--iterative] [--nxe value]" << std::endl;
            return 1;
        }
    }

    const bool nonlinear = true;
    // const bool nonlinear = false;
    solve_direct<nonlinear>(nxe);
    return 0;
};
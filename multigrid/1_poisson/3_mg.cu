#include "include/mg.h"

/* command line args:
    [DJ/GS] [--nxe int]
    * nxe must be power of 2

    examples:
    ./3_mg.out GS --nxe 2048    to run gauss seidel on 2d poisson of 2048 x 2048 elem grid
    ./3_mg.out DJ --nxe 2048    to run damped jacobi on 2d poisson of 2048 x 2048 elem grid
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T>
void mg_damped_jacobi(int nxe) {
    // natural ordering damped jacobi on GPU
    using MG = PoissonGeometricMultigrid<T, DAMPED_JACOBI>;
    printf("Startup damped jacobi geom multigrid poisson solver for nxe %d fine grid (GPU)\n", nxe);

    // create geom multigrid object
    MG mg = MG(nxe);
    T omega = 2.0 / 3.0;
    mg.setupDampedJacobi(omega);

    // now solve (these settings were carefully tuned..)
    int pre_smooth = 5, post_smooth = 5, n_vcycles = 100, inner_solve_iters = 100;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, inner_solve_iters, atol, rtol);
}

template <typename T>
void mg_gauss_seidel(int nxe) {
    // red black colored gauss seidel on GPU
    using MG = PoissonGeometricMultigrid<T, GAUSS_SEIDEL>;
    printf("Startup red-black gauss seidel geom multigrid poisson solver for nxe %d fine grid (GPU)\n", nxe);

    // create geom multigrid object
    MG mg = MG(nxe);

    // now solve (these settings were carefully tuned..)
    int pre_smooth = 1, post_smooth = 1, n_vcycles = 100, inner_solve_iters = 40;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, inner_solve_iters, atol, rtol);
}

int main(int argc, char **argv) {
    // input ----------
    bool is_gauss_seidel = false;
    int nxe = 256; // default value

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "dj") == 0) {
            is_gauss_seidel = false;
        } else if (strcmp(arg, "gs") == 0) {
            is_gauss_seidel = true;
        } else if (strcmp(arg, "--nxe") == 0) {
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

    // done reading arts, now run stuff
    using T = float;
    if (is_gauss_seidel) {
        mg_gauss_seidel<T>(nxe);
    } else {
        mg_damped_jacobi<T>(nxe);
    }

    return 0;
}
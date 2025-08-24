// demo the DKT kirchoff geometric multigrid solves..
#include "include/plate_assembler.h"
#include "include/_utils.h"
#include <chrono>

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

int main(int argc, char **argv) {
    using T = double;

    bool is_direct = false; // default is Q-order
    // int nxe = 256; // default
    int nxe = 32;

    // 2048 is largest we can solve..

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_direct = false;
        } else if (strcmp(arg, "qorder") == 0) {
            is_direct = true;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--nxe value]" << std::endl;
            return 1;
        }
    }

    T E = 2e7, thick = 0.01, nu = 0.3;
    T load_mag = 1000.0 / nxe / nxe; // so norm among diff nxe sizes..

    auto start0 = std::chrono::high_resolution_clock::now();
    auto assembler = DKTPlateAssembler(nxe, E, thick, nu, load_mag);

    // do orderings and fillin here..
    if (is_direct) {
        // AMD ordering with direct solve
        assembler.h_bsr_data.AMD_reordering();
        assembler.h_bsr_data.compute_full_LU_pattern(10.0, false);
    } else {
        // Q-order and nofill
        double qorder_p = 0.5; // smaller is more random
        assembler.h_bsr_data.qorder_reordering(qorder_p);  
        assembler.h_bsr_data.compute_nofill_pattern();
    }  

    // then call the full initialize which also does assembly
    assembler.assemble();

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    // now try and do direct solve..
    auto start1 = std::chrono::high_resolution_clock::now();
    if (is_direct) {
        assembler.direct_solve();
    } else {
        assembler.gmres_solve(true);
    }
    

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;

    int ndof = assembler.ndof;
    double total = startup_time.count() + solve_time.count();
    printf("plate direct solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // write soln to file or to python?
    T *h_soln = assembler.d_soln.createHostVec().getPtr();
    write_to_csv<T>(assembler.ndof, h_soln, "out/soln.csv");

    T *h_rhs = assembler.d_rhs.createHostVec().getPtr();
    write_to_csv<T>(assembler.ndof, h_rhs, "out/rhs.csv");

    return 0;
}

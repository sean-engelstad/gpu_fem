// demo the DKT kirchoff geometric multigrid solves..
#include "include/plate_assembler.h"
#include "include/_utils.h"

int main() {
    using T = double;

    // int nxe = 256;
    int nxe = 128;
    // int nxe = 32;
    // int nxe = 16;
    // int nxe = 4;

    T E = 2e7, thick = 0.01, nu = 0.3;
    T load_mag = 1000.0 / nxe / nxe; // so norm among diff nxe sizes..

    auto assembler = DKTPlateAssembler(nxe, E, thick, nu, load_mag);

    // now try and do direct solve..
    assembler.direct_solve();

    // write soln to file or to python?
    T *h_soln = assembler.d_soln.createHostVec().getPtr();
    write_to_csv<T>(assembler.ndof, h_soln, "out/soln.csv");

    T *h_rhs = assembler.d_rhs.createHostVec().getPtr();
    write_to_csv<T>(assembler.ndof, h_rhs, "out/rhs.csv");

    return 0;
}

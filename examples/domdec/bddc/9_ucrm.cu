// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"
#include "multigrid/utils/fea.h"
#include <string>
#include <chrono>
#include "multigrid/solvers/direct/cusp_directLU.h"

#include "domdec/bddc_assembler.h"
#include "multigrid/grid.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"

int main(int argc, char **argv) {
    // NOTE : this version uses inner direct solvers

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::string fname = "../../ilu/uCRM/CRM_box_2nd.bdf"; // clamped BCs (since only written for clamped rn)

    // =================

    printf("TODO : for uCRM case, may need METIS with more general subdomains.. that are not as smooth boundaries? cause mesh more unstructured, RETURNing early\n");
    return;

    return 0;
}

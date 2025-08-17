
#include "mesh/TACSMeshLoader.h"

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;


  using T = double;

  // auto start0 = std::chrono::high_resolution_clock::now();
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  bool mesh_print = false;

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../examples/performance/uCRM-135_wingbox_fine.bdf");

  return 0;
}
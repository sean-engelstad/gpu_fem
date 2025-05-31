
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>
#include "utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  bool mesh_print = false;
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../../examples/uCRM/CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("../../../examples/performance/uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = false; // true
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto start1 = std::chrono::high_resolution_clock::now();
  auto& bsr_data = assembler.getBsrData();

  printf("elem_conn:");
  printVec<int>(20, bsr_data.elem_conn);

  auto xpts = assembler.getXpts();
  auto h_xpts = xpts.createHostVec();
  printf("xpts:");
  printVec<T>(10, h_xpts.getPtr());
  
  double fillin = 10.0;  // 10.0
  // bsr_data.AMD_reordering();
  bsr_data.RCM_reordering(1);
  bsr_data.compute_nofill_pattern();
  // bsr_data.compute_ILUk_pattern(7, fillin, print);
  // bsr_data.compute_full_LU_pattern(fillin, print);

  assembler.moveBsrDataToDevice();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_nz_time = end1 - start1;

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();
  // assembler.add_residual(res, print); // warmup call
  assembler.add_residual(res, print);
  assembler.add_jacobian(res, kmat, print);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  auto h_bsr_data = bsr_data.createHostBsrData();
  
    int *rowp = h_bsr_data.rowp;
    int *cols = h_bsr_data.cols;
    auto h_vals = kmat.getVec().createHostVec();
    int nrows = h_bsr_data.nnodes;
    int ct = 0;
    int MAX_CT = 800;
    // int MAX_CT = 50;

    // print out the kmat sparsity pattern
    printf("nnzb = %d\n", rowp[nrows]);
    printf("rowp:");
    printVec<int>(10, rowp);
    printf("cols:");
    printVec<int>(30, cols);
    printf("perm:");
    printVec<int>(30, h_bsr_data.perm);

    // debug stop here for now
    // return;
    
    // print out the kmat values
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i+1]; jp++) {
        int j = cols[jp];
        
        for (int iv = 0; iv < 36; iv++) {
          T val = h_vals[36 * jp + iv];
          int glob_row = 6 * i + iv / 6;
          int glob_col = 6 * j + iv % 6;
          
          if (iv % 6 == 0) {
            if (iv / 6 == 0) {
              printf("Kmat[%d,%d]: ", glob_row, glob_col);
            } else {
              printf("             ");
            }
          }
          
          printf("%.14e,", val);
          ct += 1;
          if (iv % 6 == 5) printf("\n");
          if (ct > MAX_CT) break;
        }

        if (ct > MAX_CT) break;
      }

      if (ct > MAX_CT) break;
    }
    printf("\n");
  

  // free data
  assembler.free();
  kmat.free();
};

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
  // bool mesh_print = false;
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

  // printf("elem_conn:");
  // printVec<int>(20, bsr_data.elem_conn);

  // auto xpts = assembler.getXpts();
  // auto h_xpts = xpts.createHostVec();
  // printf("xpts:");
  // printVec<T>(10, h_xpts.getPtr());
  
  double fillin = 10.0;  // 10.0
  int lev_fill = 1;
  // int lev_fill = 2;
  // int lev_fill = 7;
  // int lev_fill = 11;
  // int lev_fill = 15;

  // bsr_data.AMD_reordering();  
  // bsr_data.compute_full_LU_pattern(fillin, print);

  // bsr_data.RCM_reordering(1);
  // bsr_data.compute_nofill_pattern();

  bsr_data.qorder_reordering(0.25, 1);
  bsr_data.compute_ILUk_pattern(lev_fill, fillin, print);

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

  // // was not very effective unless eta near 1e6
  // // T eta = 0.0;
  // T eta = 1e-4; // K += eta * I diag
  // // T eta = 1e2;
  // // T eta = 1e6;
  // kmat.add_diag_nugget(eta);

  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 3.0 * 23.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // CUSPARSE::direct_LU_solve(kmat, loads, soln, print);

    // int n_iter = 100, max_iter = 100;
    // int n_iter = 200, max_iter = 200;
    int n_iter = 100, max_iter = 200;
    // int n_iter = 300, max_iter = 300;
    constexpr bool right = true;
    T abs_tol = 1e-30, rel_tol = 1e-8; // for left preconditioning
    bool debug = true; // shows timing printouts with thisa
    print = true;
    // T eta_precond = 0.0;
    // T eta_precond = 1e-2;
    // CUSPARSE::GMRES_solve<T, right, true, true>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print, debug);
    CUSPARSE::HGMRES_solve<T, true>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print, debug);

  // free data
  assembler.free();
  kmat.free();
};
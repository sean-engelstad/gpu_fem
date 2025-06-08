
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
  using T = double;

  // problem inputs ----
  bool full_LU = true;
  // -------------------

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  int ndvs = assembler.get_num_dvs();
  printf("ndvs %d\n", ndvs);
  T thick2 = 1e-2;
  HostVec<T> h_dvs(ndvs, thick2);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  T mass = assembler._compute_mass();
  printf("mass %.4e\n", mass);

  // return 0;

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  if (full_LU) {
    bsr_data.AMD_reordering();
    // bsr_data.qorder_reordering(1.0);
    bsr_data.compute_full_LU_pattern(fillin, print);
  } else {
    // bsr_data.RCM_reordering();
    // bsr_data.AMD_reordering();
    bsr_data.qorder_reordering(0.5, 1); // qordering not working well for some reason..
    bsr_data.compute_ILUk_pattern(5, fillin); // 10, 20 (for BiCGStab)
    // bsr_data.compute_full_LU_pattern(fillin, print);
  }
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 10.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // solve the linear system
  if (full_LU) {
      CUSPARSE::direct_LU_solve(kmat, loads, soln);
  } else {
      int n_iter = 200, max_iter = 200;
      T abs_tol = 1e-11, rel_tol = 1e-15;
      bool print = true;
      constexpr bool right = true, modifiedGS = true; // better with modifiedGS true, yeah it is..
      CUSPARSE::GMRES_solve<T, right, modifiedGS>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
      // CUSPARSE::BiCGStab_solve<T>(kmat, loads, soln, n_iter, abs_tol, rel_tol, print);
      // CUSPARSE::PCG_solve<T>(kmat, loads, soln, n_iter, abs_tol, rel_tol, print);
      // CUSPARSE::GMRES_DR_solve<T, right, modifiedGS>(kmat, loads, soln, 4, 2, 8, abs_tol, rel_tol, print, true, 1);
  }

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "uCRM.vtk");

  // check the residual of the system
  // assembler.set_variables(soln);
  // assembler.add_residual(res);  // internal residual
  // // assembler.add_jacobian(res, kmat);
  // auto rhs = assembler.createVarsVec();
  // CUBLAS::axpy(1.0, loads, rhs);
  // CUBLAS::axpy(-1.0, res, rhs);  // rhs = loads - f_int
  // assembler.apply_bcs(rhs);
  // double resid_norm = CUBLAS::get_vec_norm(rhs);
  // printf("resid_norm = %.4e\n", resid_norm);

  // int block_dim = bsr_data.block_dim;
  // int *iperm = bsr_data.iperm;
  // assembler.apply_bcs(res);
  // auto h_res = res.createPermuteVec(block_dim, iperm).createHostVec();
  // auto h_rhs = rhs.createPermuteVec(block_dim, iperm).createHostVec();
  // int NPRINT = 100;
  // printf("add_res\nr(u): ");
  // printVec<T>(NPRINT, h_res.getPtr());
  // printf("r(u)-b: ");
  // printVec<T>(NPRINT, h_rhs.getPtr());

  // // baseline norm (with zero soln, just loads essentially)
  // rhs.zeroValues();
  // CUBLAS::axpy(1.0, loads, rhs);
  // assembler.apply_bcs(rhs);
  // double init_norm = CUBLAS::get_vec_norm(rhs);
  // printf("init_norm = %.4e\n", init_norm);

  // auto h_rhs2 = rhs.createHostVec();
  // printToVTK<Assembler, HostVec<T>>(assembler, h_rhs2, "uCRM-rhs.vtk");

  // // test get residual here
  // assembler.add_jacobian(res, kmat);
  // assembler.apply_bcs(kmat);
  // T resid2 = CUSPARSE::get_resid<T>(kmat, loads, soln);
  // printf("cusparse resid norm = %.4e\n", resid2);

  // // debug: run GMRES again starting from scratch to see initial beta
  // int n_iter = 1, max_iter = 1;
  // T abs_tol = 1e-11, rel_tol = 1e-15;
  // CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);

  // auto h_rhs = rhs.createHostVec();
  // printf("rhs:");
  // printVec<T>(10, h_rhs.getPtr());

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
  // rhs.free();
  // h_rhs.free();
};
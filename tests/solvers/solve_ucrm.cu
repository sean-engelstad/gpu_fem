
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "../test_commons.h"

void test_ucrm(bool full_LU = true, bool print = false, int ILUk = 3) {
    using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("../../examples/uCRM/CRM_box_2nd.bdf");

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

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  if (full_LU) {
    bsr_data.AMD_reordering();
    // bsr_data.qorder_reordering(1.0);
    bsr_data.compute_full_LU_pattern(fillin, print);
  } else {
    // bsr_data.qorder_reordering(1.0, 10);
    bsr_data.AMD_reordering();
    // had to use ILU(10) before with AMD, lower ILU(k) resulted in nan
    bsr_data.compute_ILUk_pattern(ILUk, fillin, print);
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
      // left preconditioning might converge in fewer iterations, but residual is not the same (not physical)

      int n_iter = 200, max_iter = 400;
      constexpr bool right = false;
      T abs_tol = 1e-11, rel_tol = 1e-15; // for left preconditioning
      
      // right preconditioning not working the best right now if multiple restarts
      // int n_iter = 200, max_iter = 200;
      // constexpr bool right = true;
      // T abs_tol = 1e-8, rel_tol = 1e-10; // for right precond
      
      CUSPARSE::GMRES_solve<T, true, right>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
  }

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "uCRM.vtk");

  // check the residual of the system
  assembler.set_variables(soln);
  assembler.add_residual(res);  // internal residual
  // assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  auto rhs = assembler.createVarsVec();
  CUBLAS::axpy(1.0, loads, rhs);
  CUBLAS::axpy(-1.0, res, rhs);  // rhs = loads - f_int
  assembler.apply_bcs(rhs);
  double resid_norm = CUBLAS::get_vec_norm(rhs);
  
  // test get residual here
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);
  T resid2 = CUSPARSE::get_resid<T>(kmat, loads, soln);

  // test result, check r(u) = r_int(u) - f close to K*u-f
  double err = rel_err(resid_norm, resid2);
  std::string solve_str = full_LU ? "LU" : "GMRES";
  bool passed = err < 1e-4;
  bool accept_fail = resid_norm < 1e-5;
  printTestReport("uCRM linear resid equivalence with " + solve_str, passed, err, accept_fail);
  printf("\t|r(u)| %.4e, |Ku-f| %.4e\n", resid_norm, resid2);

  // also debug: check K*u vs res
  auto res2 = assembler.createVarsVec();
  auto tmp = assembler.createVarsVec();
  assembler.add_jacobian(tmp, kmat);
  assembler.apply_bcs(tmp);
  assembler.apply_bcs(kmat);
  CUSPARSE::mat_vec_mult<T>(kmat, soln, res2);
  double res_norm0 = CUBLAS::get_vec_norm(res);
  double res_norm = CUBLAS::get_vec_norm(tmp);
  double res_norm2 = CUBLAS::get_vec_norm(res2);
  // if (print) {
    printf("\tresid norms |r(u)| v1 %.4e, |r(u)| %.4e, |K*u| %.4e\n", res_norm0, res_norm, res_norm2);
  // }

  // use pert vector and compare <p,r_int(u)> vs <p,K*u>
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();
  auto h_pert = HostVec<T>(nvars);
  for (int i = 0; i < nvars; i++) {
    h_pert[i] = static_cast<double>(rand()) / RAND_MAX;
    // printf("h_pert[%d] = %.4e\n", i, h_pert[i]);
  }
  T dot1 = 0.0, dot2 = 0.0;
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * h_pert[i];
    dot2 += h_res2[i] * h_pert[i];
  }
  T err2 = rel_err(dot1, dot2);
  printf("\t<p,r_int(u)> %.4e, <p,K*u> %.4e, err %.4e\n", dot1, dot2, err2);

  // and double check with |r_int(u)-f|
  rhs.zeroValues();
  CUBLAS::axpy(1.0, loads, rhs);
  CUBLAS::axpy(-1.0, res, rhs);  // rhs = loads - f_int
  assembler.apply_bcs(rhs);
  double resid_norm1 = CUBLAS::get_vec_norm(rhs);
  printf("\t|r_int(u)-F| = %.4e\n", resid_norm1);

  // ok thes r(u) and K*u seem to match, does K*u-f manually here with CUBLAS match?
  rhs.zeroValues();
  CUBLAS::axpy(1.0, loads, rhs);
  CUBLAS::axpy(-1.0, res2, rhs);  // rhs = loads - f_int
  assembler.apply_bcs(rhs);
  double resid_norm2 = CUBLAS::get_vec_norm(rhs);
  printf("\t|K*u-F| = %.4e\n", resid_norm2);

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
  rhs.free();
}

int main() {
  bool print = true;
  bool full_LU = true;
  test_ucrm(full_LU);
  full_LU = false;
  // int ILUk = 3;
  // int ILUk = 8;
  int ILUk = 10;
  // int ILUk = 20;
  test_ucrm(full_LU, print, ILUk);
};

#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"
#include <cassert>
#include <string>
#include <list>

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "../test_commons.h"

void test_resid_vs_kmat_solve(bool reordering = false, bool print = false) {
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
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
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

  auto res2 = assembler.createVarsVec();
  auto h_pert = HostVec<T>(nvars);
  for (int i = 0; i < nvars; i++) {
    h_pert[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  // assemble the kmat
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // solve the linear system
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  // check the residual of the system
  assembler.set_variables(soln);
  assembler.add_residual(res);  // internal residual
  assembler.apply_bcs(res);

  // then also compute Kmat*u (re-assemble so don't have LU factorized values)
  auto tmp = assembler.createVarsVec();
  assembler.add_jacobian(tmp, kmat);
  assembler.apply_bcs(kmat);
  CUSPARSE::mat_vec_mult<T>(kmat, soln, res2);
  assembler.apply_bcs(res2); // is this necessary here?

  // now copy to host, compute dot products and compare for test result
  // ------------------------------------------------------------------
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();

  T dot1 = 0.0, dot2 = 0.0;
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * h_pert[i];
    dot2 += h_res2[i] * h_pert[i];
  }

  double res_norm = CUBLAS::get_vec_norm(res);
  double res_norm2 = CUBLAS::get_vec_norm(res2);

  T err = rel_err(dot1, dot2);
  std::string reorder_str = reordering ? "AMD" : "no reorder";
  std::string testName = "linear shell, uCRM assembly test <u*,r(u)> vs <u*,K*u> " + reorder_str;
  printTestReport(testName, err < 1e-5, err);
  if (print) {
    // int NPRINT = nvars;
    // int NPRINT = 50;
    printf("\tresid norms |r(u)| %.4e, |K*u| %.4e, <u*,r(u)> %.4e, <u*,K*u> %.4e\n", res_norm, res_norm2, dot1, dot2);
    // printf("\tr(u):");
    // printVec<T>(NPRINT, h_res.getPtr());
    // printf("\tK*u: ");
    // printVec<T>(NPRINT, h_res2.getPtr());
  }
}

void test_resid_vs_kmat_prod(std::string ordering, std::string fill_type, bool print = false) {
  using T = double;

  int rcm_iters = 5;
  double p_factor = 1.0;
  int k = 3; // for ILU(k)
  double fillin = 10.0;

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
  // double fillin = 10.0;  // 10.0
  if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, print);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

  assembler.moveBsrDataToDevice();

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto zero_vec = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto res2 = assembler.createVarsVec();
  int nvars = assembler.get_num_vars();
  auto h_vars = HostVec<T>(nvars);
  auto h_pvec = HostVec<T>(nvars);
  for (int i = 0; i < nvars; i++) {
    h_vars[i] = static_cast<double>(rand()) / RAND_MAX;
    h_pvec[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  auto vars = h_vars.createDeviceVec();
  assembler.apply_bcs(vars);

  // assemble the kmat
  assembler.set_variables(vars);
  // assembler.add_residual(res);
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // then also compute Kmat*u (re-assemble so don't have LU factorized values)
  auto tmp = assembler.createVarsVec();
  assembler.set_variables(zero_vec);
  assembler.add_jacobian(tmp, kmat);
  assembler.apply_bcs(kmat);
  CUSPARSE::mat_vec_mult<T>(kmat, vars, res2);
  assembler.apply_bcs(res2);

  // now copy to host, compute dot products and compare for test result
  // ------------------------------------------------------------------
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();

  // printf("h_res:");
  // printVec<T>(10, h_res.getPtr());
  // printf("h_res2:");
  // printVec<T>(10, h_res2.getPtr());

  T dot1 = 0.0, dot2 = 0.0;
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * h_pvec[i];
    dot2 += h_res2[i] * h_pvec[i];
  }

  double res_norm = CUBLAS::get_vec_norm(res);
  double res_norm2 = CUBLAS::get_vec_norm(res2);

  T err = rel_err(dot1, dot2);

  // report test result ------------------
  std::string testName = "linear shell, uCRM assembly test <p,r(u)> vs <p,K*u> test, ";

  testName += ordering + ",";  // always include the ordering name

  if (fill_type == "ILUk") {
      testName += " ILU(" + std::to_string(k) + ")";
  } else if (fill_type == "nofill") {
      testName += " nofill";
  } else if (fill_type == "LU") {
      testName += " LU";
  }
  printTestReport(testName, err < 1e-9, err);
  if (print) {
    // int NPRINT = nvars;
    // int NPRINT = 50;
    printf("\tresid norms |r(u)| %.4e, |K*u| %.4e, <p,r(u)> %.4e, <p,K*u> %.4e\n", res_norm, res_norm2, dot1, dot2);
    // printf("\tr(u):");
    // printVec<T>(NPRINT, h_res.getPtr());
    // printf("\tK*u: ");
    // printVec<T>(NPRINT, h_res2.getPtr());
  }
}


int main() {

  // global Kmat tests for uCRM case

  // compare <u*,r(u)> vs <u*,K*u> where u is solution K^-1 f
  test_resid_vs_kmat_solve(true,true);

  bool test_all = true, print = false;
  if (test_all) {
      test_resid_vs_kmat_prod("none", "nofill", print);
      std::list<std::string> list1 = {"RCM", "AMD", "qorder"};
      std::list<std::string> list2 = {"nofill", "ILUk", "LU"};

      for (auto it2 = list2.begin(); it2 != list2.end(); ++it2) {
          for (auto it1 = list1.begin(); it1 != list1.end(); ++it1) {
              if (*it1 == "qorder" && *it2 == "LU") continue; // too high a nz
              test_resid_vs_kmat_prod(*it1, *it2, print);
          }
      }
  } else {
      // debug failing tests
      print = true;
      test_resid_vs_kmat_prod("none", "nofill", print);
      // test_resid_vs_kmat_prod("qorder", "ILUk", print);
  }  
};
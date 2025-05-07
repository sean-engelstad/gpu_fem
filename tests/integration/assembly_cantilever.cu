#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class Assembler>
HostVec<T> getTipLoads(Assembler &assembler, T length, T beam_tip_force) {
  // find nodes within tolerance of x=10.0
  int num_nodes = assembler.get_num_nodes();
  int num_vars = assembler.get_num_vars();
  HostVec<T> h_loads(num_vars);
  DeviceVec<T> d_xpts = assembler.getXpts();
  auto h_xpts = d_xpts.createHostVec();
  int num_tip_nodes = 0;
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      num_tip_nodes++;
    }
  }
  T nodal_force = beam_tip_force / num_tip_nodes;
  // printf("nodal force = %.4e\n", nodal_force);
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      h_loads[6 * inode + 2] = beam_tip_force / num_tip_nodes;
    }
  }
  return h_loads;
}

void test_resid_vs_kmat(bool reordering = false, bool print = false) {
  /* test <p,r(u)> vs <p,kmat*u> for linear shells */

  using T = double;

  std::ios::sync_with_stdio(false);  // always flush print immediately

  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("baseline/Beam.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
//   constexpr bool is_nonlinear = true;
  constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  // material & thick properties
  double E = 1.2e6, nu = 0.0, thick = 0.1;
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  if (reordering) bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute load magnitude of tip force
  double length = 10.0, width = 1.0;
  double Izz = width * thick * thick * thick / 12.0;
  double beam_tip_force = 4.0 * E * Izz / length / length;

  // compute loads
  auto h_loads = getTipLoads<T>(assembler, length, beam_tip_force);
  auto d_loads = h_loads.createDeviceVec();
  assembler.apply_bcs(d_loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto res2 = assembler.createVarsVec();
  int nvars = assembler.get_num_vars();
  auto h_pert = HostVec<T>(nvars);
  for (int i = 0; i < nvars; i++) {
    h_pert[i] = 0.12435 + 0.23254 * i;
  }
  auto d_pert = h_pert.createDeviceVec();

  // assemble no fill kmat
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // linear solve
  CUSPARSE::direct_LU_solve<T>(kmat, d_loads, soln);

  // now compute r(u) vs Kmat*u
  // --------------------------

  // first compute internal residual r(u)
  assembler.set_variables(soln);
  assembler.add_residual(res);
  assembler.apply_bcs(res);
  double res_norm = CUBLAS::get_vec_norm(res);
  if (print) {
    printf("res_norm %.4e\n", res_norm);
    auto h_res_ = res.createHostVec();
    printf("\tr(u):");
    printVec<T>(nvars, h_res_.getPtr());
    auto h_loads = d_loads.createHostVec();
    printf("\tloads:");
    printVec<T>(nvars, h_loads.getPtr());
  }

  // then also compute Kmat*u (re-assemble so don't have LU factorized values)
  auto tmp = assembler.createVarsVec();
  assembler.add_jacobian(tmp, kmat);
  assembler.apply_bcs(kmat);
  CUSPARSE::mat_vec_mult<T>(kmat, soln, res2);

  // now copy to host, compute dot products and compare for test result
  // ------------------------------------------------------------------
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();

  T dot1 = 0.0, dot2 = 0.0;
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * h_pert[i];
    dot2 += h_res2[i] * h_pert[i];
  }

  T err = rel_err(dot1, dot2);
  std::string reorder_str = reordering ? "AMD" : "no reorder";
  std::string testName = "linear shell, cantilever assembly test <p,r(u)> vs <p,K*u> " + reorder_str;
  printTestReport(testName, err < 1e-5, err);
  if (print) {
    printf("\tr(u):");
    printVec<T>(nvars, h_res.getPtr());
    printf("\tK*u: ");
    printVec<T>(nvars, h_res2.getPtr());
  }
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(void) {

  // check <p,r(u)> == <p,Kmat*u> on global problem with linear shells (checks proper assembly)
  test_resid_vs_kmat(false,false);
  test_resid_vs_kmat(true,false);
  return 0;
};
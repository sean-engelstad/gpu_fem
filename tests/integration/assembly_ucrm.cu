
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"

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
std::string testName = "linear shell, uCRM assembly test <p,r(u)> vs <p,K*u> " + reorder_str;
printTestReport(testName, err < 1e-5, err);
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

void test_resid_vs_kmat_prod(bool reordering = false, bool print = false) {
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
  // double fillin = 10.0;  // 10.0
  if (reordering) bsr_data.AMD_reordering();
  // bsr_data.compute_full_LU_pattern(fillin, print);
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

  // assemble the kmat
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // then also compute Kmat*u (re-assemble so don't have LU factorized values)
  auto tmp = assembler.createVarsVec();
  assembler.set_variables(zero_vec);
  assembler.add_jacobian(tmp, kmat);
  assembler.apply_bcs(kmat);
  CUSPARSE::mat_vec_mult<T>(kmat, vars, res2);

  // now copy to host, compute dot products and compare for test result
  // ------------------------------------------------------------------
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();

  T dot1 = 0.0, dot2 = 0.0;
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * h_pvec[i];
    dot2 += h_res2[i] * h_pvec[i];
  }

  double res_norm = CUBLAS::get_vec_norm(res);
  double res_norm2 = CUBLAS::get_vec_norm(res2);

  T err = rel_err(dot1, dot2);
  std::string reorder_str = reordering ? "AMD" : "no reorder";
  std::string testName = "linear shell, uCRM assembly test <p,r(u)> vs <p,K*u> " + reorder_str;
  printTestReport(testName, err < 1e-5, err);
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


void test_resid_vs_kmat_FD(int N_CHECK, double h = 1e-6, bool reordering = false, bool print = false) {
  /* compare <ej,Im{r(u+1j*ei*h)}/h> vs K_ij at global level */

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
  auto h_bsr = bsr_data.createDeviceBsrData().createHostBsrData();
  if (reordering) bsr_data.AMD_reordering();
  bool fillin = false;
  if (fillin) {
    double fillin = 10.0;  // 10.0
    bsr_data.compute_full_LU_pattern(fillin, print);
  }
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto h_vars_pert = assembler.createVarsVec().createHostVec();
  auto res = assembler.createVarsVec();
  auto res_pert = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();
  auto h_pert = HostVec<T>(nvars);
  auto h_res = HostVec<T>(nvars);
  auto h_res_pert = HostVec<T>(nvars);

  // assemble kmat
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  auto h_kmat_vals = kmat.getVec().createHostVec();

  // choose some i,j entries
  int ct = 0;
  double max_err = 0.0;

  // i is global row, j is global col
  for (int i = 0; i < 6 * h_bsr.nnodes; i++) {

    // int brow = i / 6;
    int inn_row = i % 6;
    // compute new residual with finite diff (central diff)
    h_vars_pert[i] = -h; // ei vec
    h_vars_pert.copyValuesTo(vars);
    assembler.set_variables(vars);
    assembler.add_residual(res);
    res.copyValuesTo(h_res); // copy to host

    h_vars_pert[i] = h;
    h_vars_pert.copyValuesTo(vars);
    assembler.set_variables(vars);
    assembler.add_residual(res_pert);
    res_pert.copyValuesTo(h_res_pert); // copy to host

    // reset
    h_vars_pert[i] = 0.0;
    
    // now loop over the sparsity
    for (int jp = h_bsr.rowp[i]; jp < h_bsr.rowp[i+1]; jp++) {
      int bcol = h_bsr.cols[jp];
      for (int inn_col = 0; inn_col < 6; inn_col++) {
        int j = 6 * bcol + inn_col; // j is global col
        int ind = 36 * jp + 6 * inn_row + inn_col;

        // compute K_ij using resid complex step
        double Kij_resid = (h_res_pert[j] - h_res[j]) / 2.0 / h;

        // compare to actual K_ij global
        double Kij_mat = h_kmat_vals[ind];

        // compute rel error and printout
        double err = rel_err(Kij_resid, Kij_mat);
        if (print) printf("K[%d,%d]: by resid FD %.8e, by mat %.8e, rel err %.8e\n", i, j, Kij_resid, Kij_mat, err);
        max_err = std::max(err, max_err);

        // check if we've checked enough values
        ct += 1;
        if (ct >= N_CHECK) {
          break;
        }
      }

      // early exit for loop
      if (ct >= N_CHECK) {
        break;
      }
    }

    // early exit for loop
    if (ct >= N_CHECK) {
      break;
    }
  }

  // now print test report maybe?
  printTestReport("uCRM global res vs Kmat with FD test", max_err < 1e-8, max_err);
}

int main() {

  // global Kmat tests for uCRM case

  // compare <p,r(u)> vs <p,K*u> where u is solution K^-1 f
  test_resid_vs_kmat_solve(true,true);

  // compare <p,r(u)> vs <p,K*u> where u is random vec
  bool reordering = false, print = true;
  test_resid_vs_kmat_prod(reordering,print);
  reordering = true;
  test_resid_vs_kmat_prod(reordering,print);

  // compare <ej,Im{r(u+1j*ei*h)}/h> vs K_ij at global level
  reordering = false;
  print = true; // false
  int N_CHECK = 300;
  double h = 1e-6;
  // double h = 1e-8; // too small for FD
  test_resid_vs_kmat_FD(N_CHECK, h, reordering, print);

  // compare Im{U(u+1j*ei*h)}/h vs <r(u),ei>
};
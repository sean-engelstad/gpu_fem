#include "utils//local_utils.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "solvers/_solvers.h"

void test_res_jac_prod_dense(bool print = false) {

  /** test r(u) = K*u for linear shells */

  using T = double;
  constexpr bool is_nonlinear = false; // linear shells (see above)

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

  int num_bcs = 2;
  auto assembler = createOneElementAssembler<Assembler>(num_bcs);

  // init variables u
  int num_vars = assembler.get_num_vars();
  auto res = assembler.createVarsVec();
  auto h_vars = HostVec<T>(num_vars);
  auto p_vars = HostVec<T>(num_vars);

  // fixed perturbations of the host and pert vars
  for (int ivar = 0; ivar < 24; ivar++) {
    h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    if (is_nonlinear) h_vars[ivar] *= 1e6;
    p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
  }

  auto vars = h_vars.createDeviceVec();
  assembler.set_variables(vars);

  DenseMat<VecType<T>> mat(num_vars);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  assembler.add_jacobian(res, mat);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // print res, jac
  auto h_res = res.createHostVec();
  auto h_mat = mat.createHostVec();

  // res = r(u), now take res * p
  T res_prod = 0.0;
  for (int i = 0; i < 24; i++) {
    res_prod += h_res[i] * p_vars[i];
  }

  // now compute kmat*u first
  T *Kt_u = new T[24];
  T mat_prod = 0.0;
  for (int i = 0; i < 24; i++) {
    Kt_u[i] = 0.0;
    for (int j = 0; j < 24; j++) {
        Kt_u[i] += h_mat[24*i + j] * h_vars[j];
    }
    mat_prod += Kt_u[i] * p_vars[i];
  }
  printf("here4\n");

  if (print) {
    printf("<p,r(u)> vs <p,K*u>\n");
    printf("<p,r(u)> = %.8e\n", res_prod);
    printf("<p,K*u> = %.8e\n", mat_prod);
  }

  // print residual
  // if (print) {
  //   printf("res(u): ");
  //   printVec<double>(num_vars, h_res.getPtr());
  // }

  const double *h_mat_ptr = h_mat.getPtr();
  // if (print) {
  //   printf("K*u: ");
  //   printVec<double>(num_vars, &Kt_u[0]);
  // }  

    double my_rel_err = rel_err(24, Kt_u, h_res.getPtr());
    double my_rel_err2 = rel_err(res_prod, mat_prod);
    bool passed = my_rel_err < 1e-5 && my_rel_err2 < 1e-5;
    T max_rel_err = rel_err(my_rel_err, my_rel_err2, 1e-9);
    printTestReport("shell elem, linear dense, r(u) vs K*u", passed, max_rel_err);
    printf("\tvec err %.4e, <p,vec> err %.4e\n", my_rel_err, my_rel_err2);


  printKernelTiming(duration.count());
}

void test_res_jac_prod_bsr(bool print = false) {
  /** test r(u) = K*u for linear shells */

  using T = double;
  constexpr bool is_nonlinear = false; // linear shells (see above)

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  int num_bcs = 2;
  auto assembler = createOneElementAssembler<Assembler>(num_bcs);

  // init variables u
  int num_vars = assembler.get_num_vars();
  auto res = assembler.createVarsVec();
  auto h_vars = HostVec<T>(num_vars);
  auto p_vars = HostVec<T>(num_vars);
  auto res2 = assembler.createVarsVec();
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

  // fixed perturbations of the host and pert vars
  for (int ivar = 0; ivar < 24; ivar++) {
    h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    if (is_nonlinear) h_vars[ivar] *= 1e6;
    p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
  }
  auto vars = h_vars.createDeviceVec();

  // assemble stiffness matrix (but just one element here)
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  // no bcs here.. (could add some later maybe)

  // K*u
  CUSPARSE::mat_vec_mult<T>(kmat, vars, res2);

  // test resultsauto h_res = res.createHostVec();
  auto h_res = res.createHostVec();
  auto h_res2 = res2.createHostVec();

  T dot1 = 0.0, dot2 = 0.0;
  int nvars = assembler.get_num_vars();
  for (int i = 0; i < nvars; i++) {
    dot1 += h_res[i] * p_vars[i];
    dot2 += h_res2[i] * p_vars[i];
  }

  double res_norm = CUBLAS::get_vec_norm(res);
  double res_norm2 = CUBLAS::get_vec_norm(res2);

  T err = rel_err(dot1, dot2);
  T vec_err = rel_err(h_res, h_res2);
  std::string testName = "linear shell, elem BSR <p,r(u)> vs <p,K*u> ";
  printTestReport(testName, err < 1e-10, err);
  if (print) printf("\terr %.4e, vec_err %.4e\n", err, vec_err);
}

int main(void) {

  // TODO : maybe also test with the BSRMat too?

  bool print = true;
  test_res_jac_prod_dense(print);
  test_res_jac_prod_bsr(print);
  return 0;
};
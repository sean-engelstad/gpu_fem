#include "utils//local_utils.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <bool is_nonlinear, int strain_case = -1>
void test_kelem_gpu() {

  using T = double;
  bool print = true;

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

  // printf("running!\n");
  int num_bcs = 2;
  auto assembler = createOneElementAssembler<Assembler>(num_bcs);

  if constexpr (strain_case == 1) {
    printf("testing drill strains:\n");
  } else if (strain_case == 2) {
    printf("testing bending strains:\n");
  } else if (strain_case == 3) {
    printf("testing tying strains:\n");
  } else {
    printf("testing all strains:\n");
  }

  // init variables u
  int num_vars = assembler.get_num_vars();
  auto res = assembler.createVarsVec();
  auto h_vars = HostVec<T>(num_vars);
  auto p_vars = HostVec<T>(num_vars);
  auto p_vars2 = HostVec<T>(num_vars);

  // fixed perturbations of the host and pert vars
  for (int ivar = 0; ivar < 24; ivar++) {
    h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    if (is_nonlinear) h_vars[ivar] *= 1e6;
    p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
  }

  auto vars = h_vars.createDeviceVec();
  assembler.set_variables(vars);

  // auto mat = createBsrMat<Assembler, VecType<T>>(assembler);
  // auto mat_ref = createBsrMat<Assembler, VecType<T>>(assembler);

  DenseMat<VecType<T>> mat_ref(num_vars);
  DenseMat<VecType<T>> mat(num_vars);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  bool fast_jac = false;
  assembler.template add_jacobian<strain_case>(res, mat_ref, false, false, fast_jac); // ref call
  CHECK_CUDA(cudaDeviceSynchronize());
  fast_jac = true;
  assembler.template add_jacobian<strain_case>(res, mat, false, false, fast_jac);
  CHECK_CUDA(cudaDeviceSynchronize());

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // get host mat too
  auto h_mat = mat.createHostVec();
  auto h_mat_ref = mat_ref.createHostVec();

  // compare mat here..
  double *h_mat_ref_ptr = h_mat_ref.getPtr();
  double *h_mat_ptr = h_mat.getPtr();
  if (print) {
    int nrows = 4; // for printing (could go up to 24)
    for (int i = 0; i < nrows; i++) {  // i < 2
      printf("kmat ref row %d: ", i);
      printVec<double>(num_vars, &h_mat_ref_ptr[num_vars * i]);

      printf("kmat row %d: ", i);
      printVec<double>(num_vars, &h_mat_ptr[num_vars * i]);

      return; // temporary debug
    }
  }  

  double max_ref = max(576, h_mat_ref_ptr);
  double max_abs_err = abs_err(h_mat, h_mat_ref_ptr);
  double my_rel_err = max_abs_err / max_ref;
  bool passed = my_rel_err < 1e-10;
  T max_rel_err = rel_err(h_mat, h_mat_ref_ptr, 1e-9);
  printTestReport("shell elem-jac linear", passed, my_rel_err);
  printf("\tabs err %.4e, max ref %.4e, norm err %.4e, max rel err %.4e\n", max_abs_err, max_ref, my_rel_err, max_rel_err);

  // printKernelTiming(duration.count());
}

int main(void) {

  test_kelem_gpu<false, 1>(); // linear, drill
  // test_kelem_gpu<false, 2>(); // linear, bending
  // test_kelem_gpu<false, 3>(); // linear, tying

  // test_kelem_gpu<true>(); // nonlinear

  return 0;
};

#include "coupled/meld.h"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include "../../examples/uCRM/_crm_utils.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

void test_meld_ucrm(double beta, bool write_vtk = false, bool print = false) {

  using T = double;
  auto start0 = std::chrono::high_resolution_clock::now();

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

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader<T> mesh_loader{false};
  mesh_loader.scanBDFFile("ref/uCRM-135_wingbox_medium.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  auto xs0 = assembler.getXpts();

  // make struct disps
  auto h_us = HostVec<T>(3 * ns);
  double disp_mag = 1.0;
  auto h_xs0 = xs0.createHostVec();
  for (int i = 0; i < ns; i++) {
    h_us[3 * i + 2] = disp_mag;
    // h_us[3 * i + 2] = disp_mag * (std::sin(h_xs0[3*i]/50 * 3.1415 * 4.0));
  }
  auto us = h_us.createDeviceVec();

  // load the coarse mesh for the aero surf
  // --------------------------------------

//   TACSMeshLoader<T> mesh_loader_aero{};
//   mesh_loader_aero.scanBDFFile("uCRM-135_wingbox_coarse.bdf");
//   auto _assembler_aero =
//       Assembler::createFromBDF(mesh_loader_aero, Data(E, nu, thick));
  int nx = 51;  // 101
  auto _assembler_aero = makeAeroSurfMesh<Assembler>(nx, nx, print);
  int na = _assembler_aero.get_num_nodes();
  int na_vars = _assembler_aero.get_num_vars();

  auto xa0 = _assembler_aero.getXpts();
  auto h_xa0 = xa0.createHostVec();

  // make aero loads
  auto h_fa = HostVec<T>(3 * na);
  double load_mag = 1.0;
  for (int i = 0; i < na; i++) {
    h_fa[3 * i + 2] =
        load_mag * (std::sin(h_xa0[3 * i] / 50 * 3.1415 * 4.0) *
                    std::sin(h_xa0[3 * i + 1] / 40 * 3.1415 * 3.0));
  }
  auto fa = h_fa.createDeviceVec();

  // make the MELD transfer scheme
  // -----------------------------

  // T beta = 1e-3, Hreg = 1e-8;
  T Hreg = 0.0; // 1e-4
  int sym = -1, nn = 128;
  static constexpr int NN_PER_BLOCK = 32;
  bool meld_print = false;
  // need exact_givens true for good load transfer, oneshot meld false for higher nn count, linear_meld false for NZ SVD jacobian
  constexpr bool linear_meld = false, oneshot_meld = false, exact_givens = true;
  using TransferScheme = MELD<T, NN_PER_BLOCK, linear_meld, oneshot_meld, exact_givens>;
  auto meld = TransferScheme(xs0, xa0, beta, nn, sym, Hreg, meld_print);
  meld.initialize();

  // disp transfer
  // -------------

  auto ua = meld.transferDisps(us);

  // visualization
  auto h_us_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_us);
  if (write_vtk) printToVTK<Assembler, HostVec<T>>(assembler, h_us_ext, "uCRM_us.vtk");

  // now extend to full 6-length size (MELD only handles length 3)
  auto h_ua = ua.createHostVec();
  auto h_ua_ext = MELD<T>::expandVarsVec<3, 6>(na, h_ua);

  if (write_vtk) printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_ua_ext, "uCRM_ua.vtk");

  // printf("ua:");
  // printVec<T>(100, h_ua.getPtr());

  // load transfer
  // -------------

  auto fs = meld.transferLoads(fa);
  // DeviceVec<T> fs = DeviceVec<T>(3 * na);
  // for (int i = 0; i < 10; i++) {
  //     fs = meld.transferLoads(fa);
  // }

  // extend both fs, fa to length 6 vars_per_node not 3
  auto h_fs = fs.createHostVec();
  auto h_fa_ext = MELD<T>::expandVarsVec<3, 6>(na, h_fa);
  auto h_fs_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_fs);

  // visualization of fa and fs
  if (write_vtk) printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_fa_ext, "uCRM_fa.vtk");
  if (write_vtk) printToVTK<Assembler, HostVec<T>>(assembler, h_fs_ext, "uCRM_fs.vtk");

  // verify conservation of force and work
  // compute total aero forces
  T fA_tot[3], fS_tot[3];
  memset(fA_tot, 0.0, 3 * sizeof(T));
  memset(fS_tot, 0.0, 3 * sizeof(T));

  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          fA_tot[idim] += h_fa[3 * ia + idim];
      }
  }

  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          fS_tot[idim] += h_fs[3 * is + idim];
      }
  }

  if (print) {
    printf("fA_tot:");
    printVec<double>(3, &fA_tot[0]);

    printf("fS_tot:");
    printVec<double>(3, &fS_tot[0]);
  }

  // compute total work done
  T W_A = 0.0, W_S = 0.0;
  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          W_A += h_fa[3 * ia + idim] * h_ua[3 * ia + idim];
      }
  }
  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          W_S += h_fs[3 * is + idim] * h_us[3 * is + idim];
      }
  }

  // compute force rel errors as abs error across vec / norm of ref vec f_S
  T abs_err = 0.0, ref_norm = 0.0;
  for (int i = 0; i < 3; i++) {
    abs_err = std::max(abs_err, abs(fA_tot[i] - fS_tot[i]));
    ref_norm = std::max(ref_norm, fS_tot[i]);
  }
  T F_rel_err = abs_err / ref_norm;

  // compute the relative error
  T W_rel_err = abs(W_A - W_S) / abs(W_S);

  if (print) printf("W_A %.6e, W_S %.6e, rel err %.6e\n", W_A, W_S, W_rel_err);

  // test result 
  // -------------------
  T ovr_rel_err = std::max(W_rel_err, F_rel_err);
  bool passed = F_rel_err < 1e-4 && W_rel_err < 1e-4;
  int temp = (int) 10 * beta;
  std::string testName = "MELD uCRM conservation test, beta " + std::to_string((int)beta) + "." + std::to_string(temp%10);
  bool accept_fail = beta >= 1.0;
  printTestReport<T>(testName, passed, ovr_rel_err, accept_fail);
  printf("\tW rel err %.4e, F rel err %.4e\n", W_rel_err, F_rel_err);


  // free data
  assembler.free();
  meld.free();
  h_xs0.free();
  h_us.free();
  h_xa0.free();
  h_fa.free();
  h_us_ext.free();
  h_fa_ext.free();
  h_fs_ext.free();
}

int main() {
  // test for different beta values
  test_meld_ucrm(10.0);
  test_meld_ucrm(1.0);
  test_meld_ucrm(0.1);
};

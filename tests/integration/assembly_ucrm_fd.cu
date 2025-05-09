
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

template <typename T>
void test_resid_vs_kmat_FD(std::string ordering, std::string fill_type, int N_CHECK, double h = 1e-6, bool print = false, bool debug = false) {
  /* compare <ej,FD{r(u;p)} with h> vs K_ij at global level */

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
  using Geo = typename Basis::Geo;

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

    auto h_bsr = bsr_data.createDeviceBsrData().createHostBsrData();
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
  if (debug) printf("kmat assembly\n");
  assembler.set_variables(vars);
  assembler.add_jacobian(res, kmat);
  res.copyValuesTo(h_res);
  auto h_kmat_vals = kmat.getVec().createHostVec();
  if (debug) printf("\tdone with kmat assembly\n");

  // choose some i,j entries
  int ct = 0;
  double max_err = 0.0;
  double max_abs_err = 0.0;
  double knorm = 0.0;
  int max_abs_err_ind = -1;

  T hp;
  if constexpr (std::is_same_v<T, A2D_complex_t<double>>) {
    hp = T(0.0, h);
  } else {
    hp = h;
  }

  // i is global row, j is global col
  for (int i = 0; i < 6 * h_bsr.nnodes; i++) {

    int brow_old = i / 6;
    int inn_row = i % 6;
    int brow_new = h_bsr.iperm[brow_old];
    int glob_row = 6 * brow_old + inn_row;
    // printf("glob_row %d, i %d\n", glob_row, i);
    bool continue_check = glob_row == 3 || glob_row == 4;
    if (!continue_check) continue; // just want to check Kmat[3,0] and Kmat[4,2] entries

    // compute new residual with finite diff (central diff)
    if constexpr (std::is_same_v<T, double>) {
      // don't need central difference for complex-step case
      if (debug) printf("r(u-e_{%d}h)\n", glob_row);
      h_vars_pert[glob_row] = -hp; // ei vec
      h_vars_pert.copyValuesTo(vars);
      assembler.set_variables(vars);
      assembler.add_residual(res);
      res.copyValuesTo(h_res); // copy to host
    }

    if (debug) printf("r(u+e_{%d}h)\n", glob_row);
    h_vars_pert[glob_row] = hp;
    h_vars_pert.copyValuesTo(vars);
    assembler.set_variables(vars);
    assembler.add_residual(res_pert);
    res_pert.copyValuesTo(h_res_pert); // copy to host

    // reset
    h_vars_pert[glob_row] = 0.0;
    
    // now loop over the sparsity
    for (int jp = h_bsr.rowp[brow_new]; jp < h_bsr.rowp[brow_new+1]; jp++) {
      int bcol_new = h_bsr.cols[jp];
      int bcol_old = h_bsr.perm[bcol_new];

      for (int inn_col = 0; inn_col < 6; inn_col++) {
        int j = 6 * bcol_old + inn_col; // j is global col
        int glob_col = j;
        int ind = 36 * jp + 6 * inn_row + inn_col;

        // compute K_ij using resid complex step
        double Kij_resid;
        if constexpr (std::is_same_v<T, A2D_complex_t<double>>) {
          // complex-step in complex numbers
          Kij_resid = A2D::ImagPart(h_res_pert[j] - h_res[j]) / h;
        } else {
          // central difference in real numbers
          Kij_resid = (h_res_pert[j] - h_res[j]) / 2.0 / h;
        }

        // compare to actual K_ij global
        double Kij_mat = A2D::RealPart(h_kmat_vals[ind]);

        // compute rel error and printout
        double err = rel_err(Kij_resid, Kij_mat);
        bool entry1 = glob_row == 3 && glob_col == 0;
        bool entry2 = glob_row == 4 && glob_col == 2;
        bool has_entry = entry1 || entry2;
        if (print && has_entry) {
        // if (print) {
          printf("K[%d,%d]: by resid FD %.14e, by mat %.14e, rel err %.8e, ind %d\n", glob_row, glob_col, Kij_resid, Kij_mat, err, ind);
          // printf("\tind %d, has_entry %d\n", ind, has_entry);
          if constexpr (std::is_same_v<T, A2D_complex_t<double>>) {
            printf("r(u+ei*hj)[j] = %.4e,%.4e; r(u)[j] = %.4e,%.4e\n", A2D::RealPart(h_res_pert[j]), A2D::ImagPart(h_res_pert[j]), 
            A2D::RealPart(h_res[j]), A2D::ImagPart(h_res[j]) );
          }
        }
        if (abs(Kij_resid - Kij_mat) > max_abs_err) {
          max_abs_err_ind = ind;
        }

        max_err = std::max(err, max_err);
        max_abs_err = std::max(max_abs_err, abs(Kij_resid - Kij_mat));
        knorm = std::max(knorm, Kij_mat);

        // if (print) 

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

  // also check the abs error over the norm
  double err2 = max_abs_err / knorm;

  // now print test report
  std::string testName = "uCRM global res vs Kmat with FD test, ";

  testName += ordering + ",";  // always include the ordering name

  if (fill_type == "ILUk") {
      testName += " ILU(" + std::to_string(k) + ")";
  } else if (fill_type == "nofill") {
      testName += " nofill";
  } else if (fill_type == "LU") {
      testName += " LU";
  }


  printTestReport(testName, err2 < 1e-8, err2);
  if (print) printf("\tmax abs err %.4e, knorm %.4e, max rel err %.4e, |max abs err|/|knorm| = %.4e, max abs err ind = %d\n", max_abs_err, knorm, max_err, err2, max_abs_err_ind);
}


template <typename T>
void test_energy_vs_resid_FD(std::string ordering, std::string fill_type, int N_CHECK, double h = 1e-6, bool print = false, bool debug = false) {
  /* compare FD{U(u;ei)} vs r_i at global level */

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
  using Geo = typename Basis::Geo;

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

  auto h_bsr = bsr_data.createDeviceBsrData().createHostBsrData();
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();

  // initial vecs
  HostVec<T> h_vars(nvars);
  for (int i = 0; i < nvars; i++) {
    h_vars[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  auto vars = h_vars.createDeviceVec();
  auto res = assembler.createVarsVec();
  auto h_pert = h_vars.copyVec();

  // compute baseline residual
  assembler.set_variables(vars);
  assembler.add_residual(res);
  auto h_res = res.createHostVec();

  T hp;
  if constexpr (std::is_same_v<T, A2D_complex_t<double>>) {
    hp = T(0.0, h);
  } else {
    hp = h;
  }

  // now compute energies for some number of N_CHECK entries
  DeviceVec<T> d_Uenergy{1}, d_Uenergy_p{1};
  HostVec<T> h_Uenergy{1}, h_Uenergy_p{1};
  T U_FD;
  double max_err = 0.0;
  for (int i = 0; i < N_CHECK; i++) {
    // printf("h_pert[%d] orig %.8e\n", i, h_pert[i]);

    // compute energy at u-ei*h
    h_pert[i] -= hp;
    h_pert.copyValuesTo(vars);
    assembler.set_variables(vars);
    d_Uenergy.zeroValues();
    assembler.add_energy(d_Uenergy.getPtr());

    // printf("h_pert[%d] @-h %.8e\n", i, h_pert[i]);

    // compute energy at u+ei*h
    h_pert[i] += T(2.0) * hp;
    h_pert.copyValuesTo(vars);
    assembler.set_variables(vars);
    d_Uenergy_p.zeroValues();
    assembler.add_energy(d_Uenergy_p.getPtr());

    // printf("h_pert[%d] @+h %.8e\n", i, h_pert[i]);

    // compute energy derivative
    d_Uenergy.copyValuesTo(h_Uenergy);
    d_Uenergy_p.copyValuesTo(h_Uenergy_p);
    T U1 = (h_Uenergy_p[0] - h_Uenergy[0]) / T(2) / hp;
    double U_FD = A2D::RealPart(U1);

    // compare energy derivative to residual
    double c_resid = A2D::RealPart(h_res[i]);
    double err = rel_err(U_FD, c_resid);
    max_err = std::max(err, max_err);
    if (print) printf("entry %d, energy FD %.14e, resid %.14e, rel err %.8e\n", i, U_FD, c_resid, err);

    // reset h_pert
    h_pert[i] -= hp;
    // printf("h_pert[%d] reset %.8e\n", i, h_pert[i]);
  }

  // now print test report
  std::string testName = "uCRM global energy vs resid with FD test, ";

  testName += ordering + ",";  // always include the ordering name

  if (fill_type == "ILUk") {
      testName += " ILU(" + std::to_string(k) + ")";
  } else if (fill_type == "nofill") {
      testName += " nofill";
  } else if (fill_type == "LU") {
      testName += " LU";
  }


  printTestReport(testName, max_err < 1e-8, max_err);
}

template <typename T>
void run_tests(bool test_all = true, double h = 1e-6) {
  // compare <ej,{r(u+ei*h)-r(u-ei*h)}/2/h> vs K_ij at global level (central diff FD test)

  int N_CHECK;
  bool print = false;

  N_CHECK = (int)1e3;

  // also test energy vs resid
  test_energy_vs_resid_FD<T>("none", "nofill", N_CHECK, h, print);
  // return; // temp debug
  

  N_CHECK = (int)1e6;
  if (test_all) {
    print = false;
    test_resid_vs_kmat_FD<T>("none", "nofill", N_CHECK, h, print);

    std::list<std::string> list1 = {"RCM", "AMD", "qorder"};
    std::list<std::string> list2 = {"nofill", "ILUk", "LU"};

    for (auto it2 = list2.begin(); it2 != list2.end(); ++it2) {
        for (auto it1 = list1.begin(); it1 != list1.end(); ++it1) {
            if (*it1 == "qorder" && *it2 == "LU") continue; // too high a nz
            test_resid_vs_kmat_FD<T>(*it1, *it2, N_CHECK, h, print);
        }
    }
  } else {
      // debug failing tests
      print = true;
      bool debug = true;
      test_resid_vs_kmat_FD<T>("none", "nofill", N_CHECK, h, print, debug);
      // test_resid_vs_kmat_FD<T>("RCM", "nofill", (int)1e6, h, print, debug);
  }  
}

int main() {

  // global Kmat & resid tests for uCRM case

  // can run with complex-step or real central difference
  constexpr bool complex_step = true;
  bool test_all = true;

  // run the tests
  if constexpr (complex_step) {
    double h = 1e-30;
    using T = A2D_complex_t<double>;
    run_tests<T>(test_all, h);
  } else {
    double h = 1e-6;
    using T = double;
    run_tests<T>(test_all, h);
  }
};
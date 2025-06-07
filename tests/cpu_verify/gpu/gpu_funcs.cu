
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class Assembler, class Data>
Assembler makeAssembler(MPI_Comm &comm, bool solve = true) {
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../../examples/uCRM/CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

  double E = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick, rho, ys));

  if (solve) {
    // do linear solve
    // ---------------------------------
    // ---------------------------------
    auto& bsr_data = assembler.getBsrData();
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern();
    assembler.moveBsrDataToDevice();

    // get the loads
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);
    double load_mag = 100.0;
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
    bool print = true;
    assembler.set_variables(vars);
    assembler.add_jacobian(res, kmat, print);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    CUSPARSE::direct_LU_solve<T>(kmat, loads, soln);

    // set variables in for later ksfailure evaluations
    assembler.set_variables(soln);
  }

  return assembler;
}

template <typename T, class Assembler>
void mass_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 2e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // call mass on init design
  T mass = assembler._compute_mass();
  // should match 1.8826e4
  printf("mass init %.4e\n", mass);

  // mass gradient
  DeviceVec<T> dmdx(ndvs);
  assembler._compute_mass_DVsens(dmdx);
  auto h_dmdx = dmdx.createHostVec();

  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dmass_HC = 0.0;
  cublasDdot(cublasHandle, ndvs, dmdx.getPtr(), 1, d_pert.getPtr(), 1, &dmass_HC);

  /* now FD test */
  T massn1, mass1;

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  mass1 = assembler._compute_mass();

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  massn1 = assembler._compute_mass();

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));

  // report unittest result
  T dmass_FD = (mass1 - massn1) / 2 / h;
  printf("dmass_FD %.4e dmass_HC %.4e\n", dmass_FD, dmass_HC);
}

template <typename T, class Assembler>
void ksfailure_DV_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 2e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // call mass on init design
  T rhoKS = 100.0;
  // T rhoKS = 100000.0;
  T ksfail0 = assembler._compute_ks_failure(rhoKS);
  // should match 1.1460e0
  printf("ksfailure init %.4e\n", ksfail0);

  // mass gradient
  DeviceVec<T> dksdx(ndvs);
  assembler._compute_ks_failure_DVsens(rhoKS, dksdx);
  auto h_dksdx = dksdx.createHostVec();
  // printf("h_dkxdx:");
  // printVec<T>(ndvs, h_dksdx.getPtr());

  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dks_HC = 0.0;
  cublasDdot(cublasHandle, ndvs, dksdx.getPtr(), 1, d_pert.getPtr(), 1, &dks_HC);

  /* now FD test */
  T ksn1, ks1;

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  ks1 = assembler._compute_ks_failure(rhoKS);

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  ksn1 = assembler._compute_ks_failure(rhoKS);

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));

  // report unittest result
  T dks_FD = (ks1 - ksn1) / 2 / h;
  printf("ksfail DV sens\n");
  printf("\tdks_FD %.4e dks_HC %.4e\n", dks_FD, dks_HC);
}

template <typename T, class Data>
void ksfail_strain_FD_test(T &h = 1e-6) {
  T rhoKS = 100.0;
  T E[9], pE[9];
  for (int i = 0; i < 9; i++) {
    E[i] = ((double) rand()) / RAND_MAX;
    pE[i] = ((double) rand()) / RAND_MAX;
  }
  double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  Data physData0(E_, nu, thick, rho, ys);
  T fail_index = physData0.evalFailure(rhoKS, E);
  printf("fail_index %.4e\n", fail_index);

  // FD vs HC strain sens
  T dE[9];
  memset(dE, 0.0, 9 * sizeof(T));
  physData0.evalFailureStrainSens(1.0, rhoKS, E, dE);
  T dks_HC = A2D::VecDotCore<T,9>(dE, pE);

  // FD 
  T newE[9];
  memset(newE, 0.0, 9 * sizeof(T));
  A2D::VecAddCore<T,9>(1.0, E, newE);
  A2D::VecAddCore<T,9>(h, pE, newE);
  T fail1 = physData0.evalFailure(rhoKS, newE);
  A2D::VecAddCore<T,9>(-2.0 * h, pE, newE);
  T failn1 = physData0.evalFailure(rhoKS, newE);
  T dks_FD = (fail1 - failn1) / 2.0 / h;

  printf("dks_strain FD %.4e HC %.4e\n", dks_FD, dks_HC);
}


template <typename T, class ElemGroup, class Data>
void ksfail_elem_SV_FD_test(T &h = 1e-6) {
  /* check shell elemgroup state var sens */
  T rhoKS = 1.0;
  double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  Data data(E_, nu, thick, rho, ys);
  T xpts[12], vars[24], p_vars[24], new_vars[24];
  for (int i = 0; i < 12; i++) {
    xpts[i] = ((double)rand()) / RAND_MAX;
  }
  for (int i = 0; i < 24; i++) {
    vars[i] = ((double)rand()) / RAND_MAX;
    p_vars[i] = ((double)rand()) / RAND_MAX;
    vars[i] *= 1e-4;
    p_vars[i] *= 1e-4;
  }

  int iquad = 0;
  T fail0;
  ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, vars, data, rhoKS, fail0);
  printf("fail0 = %.4e\n", fail0);

  // HC sens
  T fail_du_sens[24];
  memset(fail_du_sens, 0.0, 24 * sizeof(T));
  ElemGroup::template compute_element_quadpt_failure_sv_sens<Data>(true, iquad, xpts, vars, data, rhoKS, 1.0, fail_du_sens);
  T dks_HC = A2D::VecDotCore<T,24>(p_vars, fail_du_sens);

  // FD sens
  memset(new_vars, 0.0, 24 * sizeof(T));
  A2D::VecAddCore<T,24>(1.0, vars, new_vars);
  A2D::VecAddCore<T,24>(h, p_vars, new_vars);
  T fail1, failn1;
  ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, new_vars, data, rhoKS, fail1);
  A2D::VecAddCore<T,24>(-2.0 * h, p_vars, new_vars);
  ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, new_vars, data, rhoKS, failn1);
  T dks_FD = (fail1 - failn1) / 2.0 / h;

  printf("dks_strain_elem FD %.4e HC %.4e\n", dks_FD, dks_HC);
}

template <typename T, class Assembler>
void ksfailure_SV_FD_test(Assembler &assembler, T &h2 = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  T h = 1e-4;

  // prelim, set initial vars in
  int nvars = assembler.get_num_vars();
  // perturbation
  HostVec<T> h_vars(nvars);
  for (int i = 0; i < nvars; i++) {
    h_vars[i] = ((double) rand()) / RAND_MAX;
    h_vars[i] *= 1e-4;
  }
  auto d_vars = h_vars.createDeviceVec();
  assembler.set_variables(d_vars);

  // perturbation
  HostVec<T> h_pertvars(nvars);
  for (int i = 0; i < nvars; i++) {
    h_pertvars[i] = ((double) rand()) / RAND_MAX;
    h_pertvars[i] *= 1e-4;
  }
  auto d_pertvars = h_pertvars.createDeviceVec();

  // call KS on initial vars
  T rhoKS = 100.0;
  // T rhoKS = 1e-2;
  T ksfail0 = assembler._compute_ks_failure(rhoKS);

  // ksfailure gradient
  DeviceVec<T> dksdu(nvars);
  assembler._compute_ks_failure_SVsens(rhoKS, dksdu);
  // auto h_dksdu = dksdu.createHostVec();
  // printf("h_dksdu:");
  // printVec<T>(nvars, h_dksdu.getPtr());

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dks_HC = 0.0;
  // nvars = 10; // debug change how many vars tested in deriv at once
  cublasDdot(cublasHandle, nvars, dksdu.getPtr(), 1, d_pertvars.getPtr(), 1, &dks_HC);

  /* now FD test */
  T ksn1, ks1;

  // int ind = 0;
  // int ind = 2;
  // T *h_vars1 = d_vars.createHostVec().getPtr();
  // printf("h_vars0 = %.4e\n", h_vars1[ind]);

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));
  assembler.set_variables(d_vars);
  ks1 = assembler._compute_ks_failure(rhoKS);
  
  // T *h_vars2 = d_vars.createHostVec().getPtr();
  // printf("h_vars0+p_vars*h = %.4e\n", h_vars2[ind]);

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));
  assembler.set_variables(d_vars);
  ksn1 = assembler._compute_ks_failure(rhoKS);

  // T *h_vars3 = d_vars.createHostVec().getPtr();
  // printf("h_vars0-p_vars*h = %.4e\n", h_vars3[ind]);

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));

  printf("ksfailn1 %.4e, ksfail1 %.4e\n", ksn1, ks1);

  // report unittest result
  T dks_FD = (ks1 - ksn1) / 2 / h;
  printf("ksfail SV sens test:\n");
  printf("\tks_FD %.4e dks_HC %.4e\n", dks_FD, dks_HC);
}

template <typename T, class ElemGroup, class Data>
void ksfail_elem_adjres_FD_test(T &h = 1e-6) {
  /* check shell elemgroup state var sens */
  h = 1e-4;
  double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  // debug
  thick = 1.0;
  Data data(E_, nu, thick, rho, ys);
  T xpts[12], vars[24], psi[24];
  for (int i = 0; i < 12; i++) {
    xpts[i] = ((double)rand()) / RAND_MAX;
  }
  for (int i = 0; i < 24; i++) {
    vars[i] = ((double)rand()) / RAND_MAX;
    vars[i] *= 1e-4;
    psi[i] = ((double)rand()) / RAND_MAX;
  }

  // HC sens
  int iquad = 0;
  T adjres_HC;
  ElemGroup::template compute_element_quadpt_adj_res_product<Data>(true, iquad, xpts, vars, data, psi, &adjres_HC);
  // ElemGroup::template compute_element_quadpt_adj_res_product2<Data>(true, iquad, xpts, vars, data, psi, &adjres_HC);

  // FD sens
  T res[24], res_FD[24];
  memset(res_FD, 0.0, 24 * sizeof(T));
  data.thick += h;
  memset(res, 0.0, 24 * sizeof(T));
  ElemGroup::template add_element_quadpt_residual<Data>(true, iquad, xpts, vars, data, res);
  A2D::VecAddCore<T,24>(0.5 / h, res, res_FD);
  data.thick -= 2 * h;
  memset(res, 0.0, 24 * sizeof(T));
  ElemGroup::template add_element_quadpt_residual<Data>(true, iquad, xpts, vars, data, res);
  A2D::VecAddCore<T,24>(-0.5 / h, res, res_FD);
  T adjres_FD = A2D::VecDotCore<T,24>(res_FD, psi);
  printf("res_FD:");
  printVec<T>(6, res_FD);

  printf("adjres elem test: FD %.4e HC %.4e\n", adjres_FD, adjres_HC);
}

template <typename T, class Assembler>
void ksfailure_adj_res_prod_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 1e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // set vars pert and adjoint
  int nvars = assembler.get_num_vars();
  HostVec<T> h_psi(nvars);
  for (int i = 0; i < nvars; i++) {
    h_psi[i] = ((double) rand()) / RAND_MAX;
  }
  auto psi = h_psi.createDeviceVec();
  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  // call mass on init design
  auto f_int = assembler.createVarsVec();
  auto dfint_FD = assembler.createVarsVec();

  // compute the adjoint res product HC
  DeviceVec<T> dRdx(ndvs);
  assembler._compute_adjResProduct(psi, dRdx);

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  // compute the dot product of dRdx with x pert
  T adjres_HC;
  cublasDdot(cublasHandle, ndvs, dRdx.getPtr(), 1, d_pert.getPtr(), 1, &adjres_HC);

  // compute adjoint res product by FD
  // ---------------------------------

  // compute fint(x + p * h, u)
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  assembler.add_residual(f_int);
  // add f_int / 2 / h into fint_FD
  a = 0.5 / h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, f_int.getPtr(), 1, dfint_FD.getPtr(), 1));

  // compute fint(x - p * h, u)
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  assembler.add_residual(f_int);
  // subtract f_int / 2 / h into fint_FD
  a = -0.5 / h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, f_int.getPtr(), 1, dfint_FD.getPtr(), 1));

  // compute the dot product of fint_FD with psi
  T adjres_FD;
  cublasDdot(cublasHandle, nvars, dfint_FD.getPtr(), 1, psi.getPtr(), 1, &adjres_FD);

  
  printf("adjoint res product\n");
  printf("\tadjres_FD %.4e adjres_HC %.4e\n", adjres_FD, adjres_HC);
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

  // type declarations
  using T = double;
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

  // temp test
  T h = 1e-6;
  ksfail_strain_FD_test<T, Data>(h);
  ksfail_elem_SV_FD_test<T, ElemGroup, Data>(h);
  ksfail_elem_adjres_FD_test<T, ElemGroup, Data>(h);
  // return 0;

  bool solve = true;
  auto assembler = makeAssembler<T, Assembler, Data>(comm, solve);

  // unittests
  // ---------
  
  mass_FD_test<T, Assembler>(assembler, h);
  ksfailure_DV_FD_test<T, Assembler>(assembler, h);
  ksfailure_SV_FD_test<T, Assembler>(assembler, h);
  ksfailure_adj_res_prod_FD_test<T, Assembler>(assembler, h);
};
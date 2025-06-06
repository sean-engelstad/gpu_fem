
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
void ksfailure_FD_test(Assembler &assembler, T &h = 1e-6) {
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
  printf("dks_FD %.4e dks_HC %.4e\n", dks_FD, dks_HC);
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

  bool solve = true;
  auto assembler = makeAssembler<T, Assembler, Data>(comm, solve);

  // unittests
  // ---------
  
  T h = 1e-6;
  mass_FD_test<T, Assembler>(assembler, h);
  ksfailure_FD_test<T, Assembler>(assembler, h);
};
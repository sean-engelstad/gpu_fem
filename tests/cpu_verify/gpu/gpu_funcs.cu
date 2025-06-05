
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../../examples/uCRM/CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

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

  double E = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick, rho));

  // mass debug
  // ---------------------------------
  // ---------------------------------

  int ndvs = assembler.get_num_dvs();
  printf("ndvs %d\n", ndvs);
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  T mass = assembler._compute_mass();
  printf("mass %.4e\n", mass);

  // mass gradient
  DeviceVec<T> dmdx(ndvs);
  auto h_dmdx1 = dmdx.createHostVec();
  printf("h_dmdx1[0] = %.4e\n", h_dmdx1[0]);

  assembler._compute_mass_DVsens(dmdx);
  auto h_dmdx = dmdx.createHostVec();

  // compute mass FD deriv of first DV
  T h = 1e-6;
  global_dvs.add_value(0, h);
  assembler.set_design_variables(global_dvs);
  T mass1 = assembler._compute_mass();

  T pert2 = -2 * h;
  global_dvs.add_value(0, pert2);
  assembler.set_design_variables(global_dvs);
  T massn1 = assembler._compute_mass();
  
  T dmdx_FD_1 = (mass1 - massn1) / 2 / h;
  T dmdx_HC_1 = h_dmdx[0];

  // ---------------------------------
  // ---------------------------------
  // end of mass debug

  // return 0; // temp debug (for just mass)

  // do linear solve
  // ---------------------------------
  // ---------------------------------
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern();
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
  CUSPARSE::direct_LU_solve<T>(kmat, loads, soln);

  // ---------------------------------
  // ---------------------------------
  // end of linear solve

  // ksfailure and state dependent func debug
  // ---------------------------------
  // ---------------------------------

  // T rhoKS = 10.0;
  T rhoKS = 10000.0;
  T max_fail = assembler._compute_ks_failure(rhoKS, true);
  printf("max fail = %.4e\n", max_fail);
};
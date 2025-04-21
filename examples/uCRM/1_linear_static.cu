
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

int main() {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
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

  double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  double fillin = 10.0;  // 10.0
  bool print = true;
  assembler.symbolic_factorization(fillin, print);

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
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "uCRM.vtk");

  // check the residual of the system
  assembler.add_residual(res);  // internal residual
  auto rhs = assembler.createVarsVec();
  CUBLAS::axpy(1.0, loads, rhs);
  CUBLAS::axpy(-1.0, res, rhs);  // rhs = loads - f_int
  assembler.apply_bcs(rhs);
  double resid_norm = CUBLAS::get_vec_norm(rhs);
  printf("resid_norm = %.4e\n", resid_norm);

  auto h_rhs = rhs.createHostVec();
  printf("rhs:");
  printVec<T>(10, h_rhs.getPtr());
};
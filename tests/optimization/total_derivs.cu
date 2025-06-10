#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class StructSolver, class Function>
void function_FD_test(StructSolver &struct_solver, DeviceVec<T> &d_loads, Function &func, T &h = 1e-6) {
    /* full struct solver path FD test (testing total derivs) */

    // design vars pert
    int ndvs = struct_solver.get_num_dvs();
    auto h_pert_dvs = HostVec<T>(ndvs);
    for (int i = 0; i < ndvs; i++) {
      h_pert_dvs[i] = ((T)rand()) / RAND_MAX;
    }
    auto d_pert = h_pert_dvs.createDeviceVec();
    HostVec<T> h_dvs(ndvs, 1e-2);
    auto dvs = h_dvs.createDeviceVec();
    struct_solver.set_design_variables(dvs);

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    
    // gradient at initial design and loads
    struct_solver.solve(d_loads);
    T f0 = struct_solver.evalFunction(func);
    // struct_solver.set_design_variables(dvs); // debug (reset here bc directLU?)
    struct_solver.solve_adjoint(func);
    T HC_deriv;
    cublasDdot(cublasHandle, ndvs, func.dv_sens.getPtr(), 1, d_pert.getPtr(), 1, &HC_deriv);

    // FD gradient
    T a = 1.0 * h;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, dvs.getPtr(), 1));
    struct_solver.set_design_variables(dvs);
    if (func.has_adjoint) struct_solver.solve(d_loads);
    T f1 = struct_solver.evalFunction(func);

    // backwards pert
    a = -2.0 * h;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, dvs.getPtr(), 1));
    struct_solver.set_design_variables(dvs);
    if (func.has_adjoint) struct_solver.solve(d_loads);
    T fn1 = struct_solver.evalFunction(func);

    T FD_deriv = (f1 - fn1) * 0.5 / h;

    T rel_err = abs((HC_deriv - FD_deriv) / FD_deriv);
    bool passed = rel_err < 1e-4;
    std::string testName = "total deriv " + func.name + " FD test";
    printTestReport(testName, passed, rel_err);
    printf("\ttotal deriv test: FD %.8e, HC %.8e\n", FD_deriv, HC_deriv);
}

template <typename T, class StructSolver, class Assembler>
void test_run(StructSolver &struct_solver, Assembler &assembler, DeviceVec<T> &d_loads) {
  // init design variables
  int ndvs = assembler.get_num_dvs();
  HostVec<T> h_dvs(ndvs, 1e-2);
  auto dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(dvs);
  auto dfdu = assembler.createVarsVec();

  // functions
  T rhoKS = 100.0;
  auto mass = Mass<T, DeviceVec>();
  auto ksfail = KSFailure<T, DeviceVec>(rhoKS);

  bool mass_has_adjoint = mass.has_adjoint;
  printf("mass has adjoint %d\n", (int)mass_has_adjoint);

  // compute gradient
  struct_solver.solve(d_loads);
  T mass0 = struct_solver.evalFunction(mass);
  T ksfail0 = struct_solver.evalFunction(ksfail);
  printf("init mass = %.4e, ksfail %.4e\n", mass0, ksfail0);
  struct_solver.solve_adjoint(mass);
  struct_solver.solve_adjoint(ksfail);

  // print dfdx
  auto h_dmdx = mass.dv_sens.createHostVec();
  auto h_dkdx = ksfail.dv_sens.createHostVec();
  printf("dm/dx:");
  printVec<T>(10, h_dmdx.getPtr());
  printf("dk/dx:");
  printVec<T>(10, h_dkdx.getPtr());

  struct_solver.writeSoln("out/uCRM_coupled_us.vtk");
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
  using T = double;

  // important user settings
  // -----------------------
  constexpr bool nonlinear_strain = false; // true
  double load_mag = 100.0;

  // type definitions
  // ----------------

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, nonlinear_strain>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;
  using StructSolver = TacsLinearStatic<T, Assembler>;

  // build the Tacs prelim objects
  // -----------------------------

  double E = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../examples/uCRM/CRM_box_2nd.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick, rho, ys));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = false;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto d_loads = h_loads.createDeviceVec();
  assembler.apply_bcs(d_loads);

  // setup kmat
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto linear_solve = CUSPARSE::direct_LU_solve<T>;

  // optimization
  // ------------

  bool struct_print = false;
  auto struct_solver = StructSolver(
        assembler, kmat, linear_solve, struct_print);

  // prelim debugging
  // test_run<T, StructSolver, Assembler>(struct_solver, assembler, d_loads);

  using MyFunction1 = Mass<T, DeviceVec>;
  auto mass = MyFunction1();
  T h = 1e-6;
  function_FD_test<T, StructSolver, MyFunction1>(struct_solver, d_loads, mass, h);

  // ksfail deriv test
  T rhoKS = 100.0;
  using MyFunction2 = KSFailure<T, DeviceVec>;
  auto ksfail = KSFailure<T, DeviceVec>(rhoKS);
  h = 1e-6;
  function_FD_test<T, StructSolver, MyFunction2>(struct_solver, d_loads, ksfail, h);

  // free
  struct_solver.free();
  assembler.free();
  d_loads.free();

  return 0;
};
#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class Assembler>
HostVec<T> getTipLoads(Assembler &assembler, T length, T beam_tip_force) {
  // find nodes within tolerance of x=10.0
  int num_nodes = assembler.get_num_nodes();
  int num_vars = assembler.get_num_vars();
  HostVec<T> h_loads(num_vars);
  DeviceVec<T> d_xpts = assembler.getXpts();
  auto h_xpts = d_xpts.createHostVec();
  int num_tip_nodes = 0;
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      num_tip_nodes++;
    }
  }
  T nodal_force = beam_tip_force / num_tip_nodes;
  // printf("nodal force = %.4e\n", nodal_force);
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      h_loads[6 * inode + 2] = beam_tip_force / num_tip_nodes;
    }
  }
  return h_loads;
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(void) {
  using T = double;
  bool print = false;

  std::ios::sync_with_stdio(false);  // always flush print immediately

  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("baseline/Beam.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  // constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  // material & thick properties
  double E = 1.2e6, nu = 0.0, thick = 0.1;
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  // bsr_data.AMD_ordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute load magnitude of tip force
  double length = 10.0, width = 1.0;
  double Izz = width * thick * thick * thick / 12.0;
  double beam_tip_force = 4.0 * E * Izz / length / length;

  // compute loads
  auto h_loads = getTipLoads<T>(assembler, length, beam_tip_force);
  auto d_loads = h_loads.createDeviceVec();
  assembler.apply_bcs(d_loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto rhs = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // newton solve
  int num_load_factors = 20, num_newton = 30;
  T min_load_factor = 0.05, max_load_factor = 1.00, abs_tol = 1e-8,
    rel_tol = 1e-7;
  auto solve_func = CUSPARSE::direct_LU_solve<T>;
  std::string outputPrefix = "out/beam_";
  newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler>(
      solve_func, kmat, d_loads, soln, assembler, res, rhs, vars,
      num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
      rel_tol, outputPrefix, print);

  // get tip displacements
  int num_nodes = assembler.get_num_nodes();
  auto xpts = assembler.getXpts();
  auto h_xpts = xpts.createHostVec();
  auto h_soln = vars.createHostVec();
  T end_disp[3];
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      for (int j = 0; j < 3; j++) {
        end_disp[j] = h_soln[6 * inode + j];
      }
      if (print) {
        printf("End disp (u,v,w):");
        printVec<T>(3, &h_soln.getPtr()[6*inode]);
      }
    }
  }

//   thetaTip=1.125661077149547 XdispNorm=0.3315629142866051 ZdispNorm=0.6720677231202588
  T ref_end_disp[] = {-3.315629142866051, 0.0, 6.720677231202588};

  T max_rel_err = rel_err(3, end_disp, ref_end_disp, 1e-8);
  bool passed = max_rel_err < 1e-2;
  printTestReport("nonlinear cantilever end disp", passed, max_rel_err);
  printf("\tin-plane disp\tGPU %.4e, ref %.4e\n", end_disp[0], ref_end_disp[0]);
  printf("\ttip disp\tGPU %.4e, ref %.4e\n", end_disp[2], ref_end_disp[2]);

  return 0;
};
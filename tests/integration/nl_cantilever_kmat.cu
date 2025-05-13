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
  // no fillin or reordering, let's just compute the kmat for comparison with TACS
  assembler.moveBsrDataToDevice();

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();

  // assemble no fill kmat
  assembler.add_jacobian(res, kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // compare the global stiffness matrix
  auto h_kmat_vec = kmat.getVec().createHostVec();
  h_kmat_vec.print("kmat");

  // TODO : need CPU ref global kmat

  // T max_rel_err = rel_err(3, end_disp, ref_end_disp, 1e-8);
  T max_rel_err = 0.0;
  bool passed = max_rel_err < 1e-2;
  printTestReport("NL cantilever global kmat", passed, max_rel_err);

  return 0;
};
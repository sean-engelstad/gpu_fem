#include "_local_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
  using T = double;

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  int num_bcs = 10;

  // int num_elements = 1e3;
  // int num_geo_nodes = 1e2;
  // int num_vars_nodes = 1e2;

    int num_geo_nodes = 10;
    int num_vars_nodes = num_geo_nodes;
    int num_elements = 4;

//   int num_geo_nodes = 100;
//   int num_vars_nodes = num_geo_nodes;
//   int num_elements = 1e3;

  auto assembler = createFakeAssembler<Assembler>(
      num_bcs, num_elements, num_geo_nodes, num_vars_nodes);

  // init variables u
  bool randomize = true;
  auto vars = assembler.createVarsVec(randomize);
  assembler.set_variables(vars);

  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto bsrData = kmat.getBsrData();

  // original fillin
  printf("rowPtr:\n");
  printVec<int32_t>(10, bsrData.rowPtr);
  printf("colPtr:\n");
  printVec<int32_t>(bsrData.nnzb, bsrData.colPtr);



  return 0;
};
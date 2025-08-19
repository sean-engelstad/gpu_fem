
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>
#include "utils.h"
#include <fstream>

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T>
void apply_load_bcs(MPI_Comm comm, DeviceVec<T> d_loads, int *&d_perm, int *&d_iperm) {
  // make assembler just for loads
  auto start0 = std::chrono::high_resolution_clock::now();
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  bool mesh_print = false;
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../../examples/uCRM/CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("../../../examples/performance/uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  // using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = false; // true
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  auto& bsr_data = assembler.getBsrData();
  bsr_data.AMD_reordering();
  bsr_data.compute_ILUk_pattern(7, 10.0, print);
  assembler.moveBsrDataToDevice();

  assembler.apply_bcs(d_loads);
  d_perm = bsr_data.perm;
  d_iperm = bsr_data.iperm;
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

  using T = double;

  // load kmat in ILU(7) format pre-factorization from CPU
  std::ifstream fin("../cpu/cpu_prefactor_ILU7.bcsr", std::ios::binary);

  int block_dim, n_block_rows, nnzb;
  fin.read((char*)&block_dim, sizeof(int));
  fin.read((char*)&n_block_rows, sizeof(int));
  fin.read((char*)&nnzb, sizeof(int));

  std::vector<int> rowp(n_block_rows + 1);
  std::vector<int> cols(nnzb);
  std::vector<double> values(nnzb * block_dim * block_dim);

  fin.read((char*)rowp.data(), sizeof(int) * (n_block_rows + 1));
  fin.read((char*)cols.data(), sizeof(int) * nnzb);
  fin.read((char*)values.data(), sizeof(double) * nnzb * block_dim * block_dim);

  fin.close();

  printf("block_dim %d\n", block_dim);
  int nrows = n_block_rows;
  printf("nnzb = %d\n", rowp[nrows]);
  printf("rowp:");
  printVec<int>(10, rowp.data());
  printf("cols:");
  printVec<int>(30, cols.data());

  // make kmat, etc.
  // ---------------------------------

  // deep copy
  int mb = n_block_rows;
  int *h_rowp = new int[mb + 1];
  int *h_cols = new int[nnzb];
  for (int i = 0; i < mb + 1; i++) h_rowp[i] = rowp[i];
  for (int i = 0; i < nnzb; i++) h_cols[i] = cols[i];
  int nnz = nnzb * block_dim * block_dim;
  T *h_vals = new T[nnz];
  for (int i = 0; i < nnz; i++) h_vals[i] = values[i];

  // move BSR data to the device
  index_t *d_rowp = HostVec<int>(mb + 1, h_rowp).createDeviceVec().getPtr();
  index_t *d_cols = HostVec<int>(nnzb, h_cols).createDeviceVec().getPtr();
  // int *iperm = bsr_data2.iperm;
  auto d_vals = HostVec<T>(nnzb * block_dim * block_dim, h_vals).createDeviceVec();
  int ct = 0;
  int MAX_CT = 800;

  int *h_rowp2 = DeviceVec<int>(mb + 1, d_rowp).createHostVec().getPtr();
    printf("h_rowp2:");
    printVec<int>(10, h_rowp2);

  // make RHS
  int N = mb * 6;
  auto h_loads = HostVec<T>(N);
  double load_mag = 3.0 * 23.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < mb; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  int *d_perm, *d_iperm;
  apply_load_bcs<T>(comm, loads, d_perm, d_iperm);
  auto soln = DeviceVec<T>(N);

  // T *h_rhs = DeviceVec<T>(mb * 6, rhs_perm.getPtr()).getPtr();
  //   printf("h_rhs:");
  //   printVec<T>(10, h_rhs);

  // make BsrData (device version)
  BsrData bsr_data = BsrData(mb, block_dim, nnzb, d_rowp, d_cols, d_perm, d_iperm, false);

  // make BsrMat
  BsrMat<DeviceVec<T>> kmat = BsrMat<DeviceVec<T>>(bsr_data, d_vals);

  
  
  // do GMRES solve
  // --------------
  bool print = true;
  int n_iter = 300, max_iter = 300;
  constexpr bool right = true;
  T abs_tol = 1e-6, rel_tol = 1e-6; // for left preconditioning
  CUSPARSE::GMRES_solve<T, true, right>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);

};
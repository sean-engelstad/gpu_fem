
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

struct COOEntry {
    int row;
    int col;
    double value;
};

void load_mtx(const std::string& filename, int& nrows, int& ncols, std::vector<COOEntry>& entries) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream iss(line);
    int nnz;
    iss >> nrows >> ncols >> nnz;

    entries.reserve(nnz);
    int r, c;
    double val;
    for (int i = 0; i < nnz; ++i) {
        infile >> r >> c >> val;
        entries.push_back({r - 1, c - 1, val});  // Convert to 0-based indexing
    }
}

void solve_linear(MPI_Comm &comm, bool full_LU = true, double qorder_p = 0.5, int fill_level = 1) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // read BSRData from the mtx file
  int nrows, ncols;
  std::vector<COOEntry> entries;
  load_mtx("stiffness_ucrm13p5.mtx", nrows, ncols, entries);

  printf("nrows %d, ncols %d\n", nrows, ncols);
  for (int i = 0; i < 3; i++) {
    COOEntry entry = entries[i];
    printf("CooEntry row %d, col %d, entry %.4e\n", entry.row, entry.col, entry.value);
  }

  // make BsrData now
  int nnz = entries.size();
  int nnzb = nnz / 36;
  printf("nnz %d, nnzb %d\n", nnz, nnzb);
  int nbrows = nrows/6;
  int *h_rowp = new int[nbrows+1];
  int *h_cols = new int[nnzb];
  T *h_vals = new T[36 * nnzb];

  int nnodes = nrows / 6;
  bool *bcs = new bool[nnodes];
  memset(bcs, false, nnodes * sizeof(bool));

  int nnzb_ct = 0;
  for (int i = 0; i < nnz; i++) {
    COOEntry entry = entries[i];
    int row = entry.row;
    int col = entry.col;
    T val = entry.value;

    if (row == col && val == 1.0) {
      bcs[row/6] = true;
    }

    int brow = row / 6;
    int bcol = col / 6;

    if (row % 6 == 0 && col % 6 == 0) {
      // printf("row %d, col %d, brow %d, bcol %d, nnzb_ct %d\n", row, col, brow, bcol, nnzb_ct);
      // update block rowp, cols here
      h_cols[nnzb_ct] = bcol;
      nnzb_ct++;
      h_rowp[brow+1] = nnzb_ct;
    }

    // figure out where to put the value
    for (int jp = h_rowp[brow]; jp < h_rowp[brow+1]; jp++) {
      int j = h_cols[jp];
      if (j == bcol) {
        int inrow = row % 6;
        int incol = col % 6;
        int inind = 6 * inrow + incol;
        h_vals[36 * jp + inind] = val;
        break;
      }
    }
  }

  printf("h_rowp:");
  // printVec<int>(nnodes+1, h_rowp);
  printVec<int>(30, h_rowp);
  printf("h_cols:");
  printVec<int>(100, h_cols);
  // printVec<int>(nnzb, h_cols);
  // printf("h_vals:");
  // printVec<T>(100, h_vals);

  int *orig_rowp = new int[nbrows+1];
  int *orig_cols = new int[nnzb];
  memcpy(orig_rowp, h_rowp, (nbrows+1) * sizeof(int));
  memcpy(orig_cols, h_cols, nnzb * sizeof(int));

  int mb = nrows / 6;
  int block_dim = 6;

  auto h_bsr_data = BsrData(mb, block_dim, nnzb, h_rowp, h_cols, nullptr, nullptr, true);
  
  bool print = true;
  printf("1\n");
  if (full_LU) {
    h_bsr_data.AMD_reordering();
    h_bsr_data.compute_full_LU_pattern(/*fillin*/ 10.0, print);
  } else {
    h_bsr_data.qorder_reordering(qorder_p, 1);
    h_bsr_data.compute_ILUk_pattern(fill_level, 10.0);
  }

  // get chain lengths and write that out
  double chain_lengths[h_bsr_data.nnodes];
  h_bsr_data.get_chain_lengths(chain_lengths);
  write_to_csv<double>(h_bsr_data.nnodes, chain_lengths, "../csv/chain_lengths.csv");

  printf("2\n");
  auto bsr_data = h_bsr_data.createDeviceBsrData();
  printf("3\n");

  T *h_fill_vals = new T[h_bsr_data.nnzb * 36];
  memset(h_fill_vals, 0.0, 36 * h_bsr_data.nnzb * sizeof(T));

  // copy values to new spots (after perm and reordering..)
  for (int i = 0; i < nrows; i++) {
    int brow_old = i/6;
    int inn_row = i % 6;
    int brow_new = h_bsr_data.iperm[brow_old];
    // int glob_row = 6 * brow_new + inn_row;

    // loop over the old sparsity
    for (int jp = orig_rowp[brow_old]; jp < orig_rowp[brow_old+1]; jp++) {
      int bcol_old = orig_cols[jp];
      int bcol_new = h_bsr_data.iperm[bcol_old];
      
      // get corresponding bcol in new permuted
      bool found_bcol = false;
      int _jp2 = -1;
      for (int jp2 = h_bsr_data.rowp[brow_new]; jp2 < h_bsr_data.rowp[brow_new+1]; jp2++) {
        int bcol2 = h_bsr_data.cols[jp2];
        if (bcol2 == bcol_new) {
          found_bcol = true;
          _jp2 = jp2;
        }
      }
      assert(found_bcol);

      for (int jj = 0; jj < 6; jj++) {
        int ii = 6 * inn_row + jj;
        h_fill_vals[36 * _jp2 + ii] = h_vals[36 * jp + ii];
      }
    }
  }

  // put bsr values on the device
  auto h_vals_vec = HostVec<T>(h_bsr_data.nnzb * 36, h_fill_vals);
  auto d_vals = h_vals_vec.createDeviceVec();
  auto kmat = BsrMat<DeviceVec<T>>(bsr_data, d_vals);

  // create the loads to apply to it
  HostVec<T> h_loads(nrows);
  double load_mag = 10.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = bcs[inode] ? 0.0 : load_mag;
  }
  auto loads = h_loads.createDeviceVec();

  // printf("h_loads:");
  // printVec<T>(30, h_loads_ptr);

  auto soln = DeviceVec<T>(nrows);

  // solve system
  if (full_LU) {
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
  } else {
      int n_iter = 200, max_iter = 200;
      T abs_tol = 1e-8, rel_tol = 1e-8;
      // T abs_tol = 1e-13, rel_tol = 1e-13;
      bool print = true;
      // constexpr bool right = false, modifiedGS = true; // better with modifiedGS true, yeah it is..
      constexpr bool right = true, modifiedGS = true; // better with modifiedGS true, yeah it is..
      CUSPARSE::GMRES_solve<T, right, modifiedGS>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
  }
}

int main(int argc, char **argv) {
    /* command line args:
       ./1_static.out linear      to run linear
       ./1_static.out nonlinear   to run nonlinear
       add the option --iterative to make it switch from full_LU (only for linear)
    */

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    // solve_linear(comm, /*full_LU*/ true, /*qorder_p=*/ 0.5, /*fill_level=*/ 1);
    // solve_linear(comm, /*full_LU*/ false, /*qorder_p=*/ 0.5, /*fill_level=*/ 0);
    solve_linear(comm, /*full_LU*/ false, /*qorder_p=*/ 0.5, /*fill_level=*/ 1);
    // solve_linear(comm, /*full_LU*/ false, /*qorder_p=*/ 0.5, /*fill_level=*/ 9);

    MPI_Finalize();
    return 0;
};

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

int main(int argc, char **argv) {
    
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

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

  // do LU factorization like in GMRES
  // ---------------------------------

  // move BSR data to the device
  int mb = n_block_rows;
  index_t *d_rowp = HostVec<int>(n_block_rows + 1, rowp.data()).createDeviceVec().getPtr();
  index_t *d_cols = HostVec<int>(nnzb, cols.data()).createDeviceVec().getPtr();
  // int *iperm = bsr_data2.iperm;
  T *d_vals_ILU0 = HostVec<T>(nnzb * block_dim * block_dim, values.data()).createDeviceVec().getPtr();
  int ct = 0;
  int MAX_CT = 800;

  bool print_pre_factor = true;
  if (print_pre_factor) {  
    // auto h_ILU_vals_pre = DeviceVec<T>(nnzb * block_dim * block_dim, d_vals_ILU0).createHostVec();

    // do the factorization here
    // print out the kmat values
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i+1]; jp++) {
        int j = cols[jp];
        
        for (int iv = 0; iv < 36; iv++) {
          T val = values[36 * jp + iv];
          int glob_row = 6 * i + iv / 6;
          int glob_col = 6 * j + iv % 6;
          
          if (iv % 6 == 0) {
            if (iv / 6 == 0) {
              printf("Kmat[%d,%d]: ", glob_row, glob_col);
            } else {
              printf("             ");
            }
          }
          
          printf("%.14e,", val);
          ct += 1;
          if (iv % 6 == 5) printf("\n");
          if (ct > MAX_CT) break;
        }

        if (ct > MAX_CT) break;
      }

      if (ct > MAX_CT) break;
    }
    printf("\n");
  }

  // create initial cusparse and cublas handles --------------

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* Create CUSPARSE context */
  cusparseHandle_t cusparseHandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

  // create ILU(0) preconditioner
  // -----------------------------
  // [equiv to ILU(k) precondioner if ILU(k) sparsity pattern used in BsrData object]

  // init objects for LU factorization and LU solve
  cusparseMatDescr_t descr_L = 0, descr_U = 0;
  bsrsv2Info_t info_L = 0, info_U = 0;
  void *pBuffer = 0;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                              policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  // tried changing both policy L and U to be USE_LEVEL not really a change
  // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
  // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                            trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

  // T a = 1.0, b = 0.0;

  // perform the symbolic and numeric factorization of LU on given sparsity pattern
  CUSPARSE::perform_LU_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                      mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                      trans_U, policy_L, policy_U, dir);
  
  // copy back h_vals_ILU0 to host
  auto h_ILU_vals = DeviceVec<T>(nnzb * block_dim * block_dim, d_vals_ILU0).createHostVec();

  printf("after factorization\n");
  ct = 0;
    // do the factorization here
    // print out the kmat values
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i+1]; jp++) {
        int j = cols[jp];
        
        for (int iv = 0; iv < 36; iv++) {
          T val = h_ILU_vals[36 * jp + iv];
          int glob_row = 6 * i + iv / 6;
          int glob_col = 6 * j + iv % 6;
          
          if (iv % 6 == 0) {
            if (iv / 6 == 0) {
              printf("ILU[%d,%d]: ", glob_row, glob_col);
            } else {
              printf("             ");
            }
          }
          
          printf("%.14e,", val);
          ct += 1;
          if (iv % 6 == 5) printf("\n");
          if (ct > MAX_CT) break;
        }

        if (ct > MAX_CT) break;
      }

      if (ct > MAX_CT) break;
    }
    printf("\n");
};
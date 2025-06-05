
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>
#include "utils.h"

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
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // bool mesh_print = false;
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("../../../examples/uCRM/CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("../../../examples/performance/uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = false; // true
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto start1 = std::chrono::high_resolution_clock::now();
  auto& bsr_data = assembler.getBsrData();

  printf("elem_conn:");
  printVec<int>(20, bsr_data.elem_conn);

  auto xpts = assembler.getXpts();
  auto h_xpts = xpts.createHostVec();
  printf("xpts:");
  printVec<T>(10, h_xpts.getPtr());
  
  double fillin = 10.0;  // 10.0
  bsr_data.AMD_reordering();
  // bsr_data.compute_nofill_pattern();
  bsr_data.compute_ILUk_pattern(7, fillin, print);
  // bsr_data.compute_full_LU_pattern(fillin, print);

  assembler.moveBsrDataToDevice();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_nz_time = end1 - start1;

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();
  // assembler.add_residual(res, print); // warmup call
  assembler.add_residual(res, print);
  assembler.add_jacobian(res, kmat, print);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  auto h_bsr_data = bsr_data.createHostBsrData();
  
    int *rowp = h_bsr_data.rowp;
    int *cols = h_bsr_data.cols;
    int nrows = h_bsr_data.nnodes;
    int ct = 0;
    int MAX_CT = 800;
    // int MAX_CT = 50;

    // print out the kmat sparsity pattern
    printf("nnzb = %d\n", rowp[nrows]);
    printf("rowp:");
    printVec<int>(10, rowp);
    printf("cols:");
    printVec<int>(30, cols);
    printf("perm:");
    printVec<int>(30, h_bsr_data.perm);

  // do LU factorization like in GMRES
  // ---------------------------------

  // copy important inputs for Bsr structure out of BsrMat
  // TODO : was trying to make some of these const but didn't accept it in
  // final solve
  BsrData bsr_data2 = kmat.getBsrData();
  int mb = bsr_data2.nnodes;
  int nnzb = bsr_data2.nnzb;
  int block_dim = bsr_data2.block_dim;
  index_t *d_rowp = bsr_data2.rowp;
  index_t *d_cols = bsr_data2.cols;
  // int *iperm = bsr_data2.iperm;
  int N = soln.getSize();

  // note this changes the mat data to be LU (but that's the whole point
  // of LU solve is for repeated linear solves we now just do triangular
  // solves)
  T *d_vals = kmat.getPtr();

  // also make a temporary array for the preconditioner values
  T *d_vals_ILU0 = DeviceVec<T>(kmat.get_nnz()).getPtr();
  // ILU0 equiv to ILU(k) if sparsity pattern has ILU(k)
  // CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, kmat.get_nnz() * sizeof(T))); // isn't this redundant, try commenting out later
  CHECK_CUDA(
      cudaMemcpy(d_vals_ILU0, d_vals, kmat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

  bool print_pre_factor = 0;
  if (print_pre_factor) {  
    auto h_ILU_vals_pre = DeviceVec<T>(kmat.get_nnz(), d_vals_ILU0).createHostVec();

    // do the factorization here
    // print out the kmat values
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i+1]; jp++) {
        int j = cols[jp];
        
        for (int iv = 0; iv < 36; iv++) {
          T val = h_ILU_vals_pre[36 * jp + iv];
          int glob_row = 6 * i + iv / 6;
          int glob_col = 6 * j + iv % 6;
          
          if (iv % 6 == 0) {
            if (iv / 6 == 0) {
              printf("ILU_pre[%d,%d]: ", glob_row, glob_col);
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
  auto h_ILU_vals = DeviceVec<T>(kmat.get_nnz(), d_vals_ILU0).createHostVec();

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

  // free data
  assembler.free();
  kmat.free();
};

#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"
#include "utils/_mat_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "../test_commons.h"

template <typename T>
void bsr_cusparse_precond(std::string ordering, std::string fill_type, int k = 3, bool print = false) {
    // temp add this back
    auto start0 = std::chrono::high_resolution_clock::now();

    int rcm_iters = 5;
    double p_factor = 1.0;
    double fillin = 10.0;

    // uCRM mesh files can be found at:
    // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("../../examples/uCRM/CRM_box_2nd.bdf");

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = typename Basis::Geo;

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
    auto& bsr_data = assembler.getBsrData();
    
    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, print);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

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
    auto rhs = h_loads.createDeviceVec();
    assembler.apply_bcs(rhs);

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

    // permute the rhs
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(kmat, rhs);

    // get the important data
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = rhs.getSize();
    T *d_rhs = rhs_perm.getPtr();
    T *d_vals = kmat.getPtr(); // keep ILU(0) in d_vals not another array

    T *d_tmp = DeviceVec<T>(rhs.getSize()).getPtr();
    T *d_resid = DeviceVec<T>(rhs.getSize()).getPtr();

    // compute M the LU preconditioner
    // create initial cusparse and cublas handles --------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create ILU(0) preconditioner
    // -----------------------------

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_LU_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                       mb, nnzb, block_dim, d_vals, d_rowp, d_cols, trans_L,
                                       trans_U, policy_L, policy_U, dir);

    // now compute |M^-1 b| with cusparse ILU(0)
    // L^-1 * d_rhs => d_tmp
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                            descr_L, d_vals, d_rowp, d_cols, block_dim,
                                            info_L, d_rhs, d_tmp, policy_L, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // U^-1 * d_tmp => into d_resid
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                            descr_U, d_vals, d_rowp, d_cols, block_dim,
                                            info_U, d_tmp, d_resid, policy_U, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // now compute the init resid norm |M^-1 b|
    T nrm_b;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_rhs, 1, &nrm_b));
    T beta;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("BSR cusparse precond |b| = %.4e, |M^-1 b| = %.8e\n", nrm_b, beta);

    // then free
    kmat.free();
    rhs.free();

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements

    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_tmp));
}


template <typename T>
void csr_cusparse_precond(std::string ordering, std::string fill_type, int k = 3, bool print = false) {
    // temp add this back
    auto start0 = std::chrono::high_resolution_clock::now();

    int rcm_iters = 5;
    double p_factor = 1.0;
    // int nxe = 5;
    double fillin = 10.0;

    // uCRM mesh files can be found at:
    // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("../../examples/uCRM/CRM_box_2nd.bdf");

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = typename Basis::Geo;

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
    auto& bsr_data = assembler.getBsrData();
    
    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, print);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

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
    auto rhs = h_loads.createDeviceVec();
    assembler.apply_bcs(rhs);

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

    // permute the rhs
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(kmat, rhs);

    // get the bsr data on the host
    auto h_bsr_data = bsr_data.createHostBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *h_bsr_rowp = h_bsr_data.rowp;
    index_t *h_bsr_cols = h_bsr_data.cols;
    int N = rhs.getSize();
    T *h_bsr_vals = kmat.getVec().createHostVec().getPtr();

    // get important vector data
    T *d_rhs = rhs_perm.getPtr();
    T *d_tmp = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();   
    T *d_w = DeviceVec<T>(N).getPtr();   

    // convert bsr to csr data
    int nz;
    int *h_csr_rowp, *h_csr_cols;
    T *h_csr_vals;
    BSRtoCSR<T>(block_dim, N, nnzb, h_bsr_rowp, h_bsr_cols, h_bsr_vals, &h_csr_rowp, &h_csr_cols, &h_csr_vals, &nz);

    // now move the CSR data back to the device
    HostVec<int> h_rowp(N+1, h_csr_rowp), h_cols(nz, h_csr_cols);
    HostVec<T> h_vals(nz, h_csr_vals);
    int *d_rowp = h_rowp.createDeviceVec().getPtr();
    int *d_cols = h_cols.createDeviceVec().getPtr();
    T *d_vals = h_vals.createDeviceVec().getPtr();

    // compute M the LU preconditioner
    // create initial cusparse and cublas objects
    // ------------------------------------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descr = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // wrap dense vectors into cusparse dense vector objects
    // -----------------------------------------------------

    cusparseDnVecDescr_t vec_rhs, vec_tmp, vec_w, vec_resid;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_rhs, N, d_rhs, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_tmp, N, d_tmp, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w, N, d_w, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_resid, N, d_resid, CUDA_R_64F));
    
    // create the matrix CSR objects
    // -----------------------------

    cusparseSpMatDescr_t matM_lower, matM_upper;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    //Lower Part 
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_lower, N, N, nz, d_rowp, d_cols, d_vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

                                      CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
        CUSPARSE_SPMAT_FILL_MODE,
        &fill_lower, sizeof(fill_lower)) );
        CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
        CUSPARSE_SPMAT_DIAG_TYPE,
        &diag_unit, sizeof(diag_unit)) );

    // M_upper
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_upper, N, N, nz, d_rowp, d_cols, d_vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
                                      CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
            CUSPARSE_SPMAT_FILL_MODE,
            &fill_upper, sizeof(fill_upper)) );
            CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
            CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_non_unit,
            sizeof(diag_non_unit)) );

    // create ILU(0) preconditioner
    // ----------------------------


    int                 bufferSizeLU = 0;
    size_t              bufferSizeL, bufferSizeU;
    void*               d_bufferLU, *d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseMatDescr_t   matLU;
    csrilu02Info_t      infoILU = NULL;
    const T floatone = 1.0;
    const T floatzero = 0.0;

    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&infoILU));
    CHECK_CUSPARSE( cusparseCreateMatDescr(&matLU) );
    CHECK_CUSPARSE( cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CHECK_CUSPARSE( cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO) );

    /* Allocate workspace for cuSPARSE */
    CHECK_CUSPARSE(cusparseDcsrilu02_bufferSize(
    cusparseHandle, N, nz, matLU, d_vals, d_rowp, d_cols, infoILU, &bufferSizeLU));
    CHECK_CUDA( cudaMalloc(&d_bufferLU, bufferSizeLU) );

    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrL) );
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
    cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower, vec_tmp, vec_w, CUDA_R_64F,
    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    CHECK_CUDA( cudaMalloc(&d_bufferL, bufferSizeL) );

    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrU) );
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper, vec_tmp, vec_w, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
    CHECK_CUDA( cudaMalloc(&d_bufferU, bufferSizeU) );

    // GMRES solve now with CSR matrix
    // -------------------------------

    /* Perform analysis for ILU(0) */
    CHECK_CUSPARSE(cusparseDcsrilu02_analysis(
        cusparseHandle, N, nz, descr, d_vals, d_rowp, d_cols, infoILU,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    int structural_zero;
    CHECK_CUSPARSE(cusparseXcsrilu02_zeroPivot(cusparseHandle, infoILU, &structural_zero));
    // print or assert if needed
    if (structural_zero != -1) printf("structural zero = %d\n", structural_zero);

    /* generate the ILU(0) factors */
    CHECK_CUSPARSE(cusparseDcsrilu02(
        cusparseHandle, N, nz, matLU, d_vals, d_rowp, d_cols, infoILU,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    int numerical_zero;
    CHECK_CUSPARSE(cusparseXcsrilu02_zeroPivot(cusparseHandle, infoILU, &numerical_zero));
    // again, print/check these for zero pivots
    if (numerical_zero != -1) printf("numerical_zero = %d\n", numerical_zero);

    /* perform triangular solve analysis */
    CHECK_CUSPARSE(cusparseSpSV_analysis(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_lower, vec_tmp, vec_w, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL));

    CHECK_CUSPARSE(cusparseSpSV_analysis(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_upper, vec_tmp, vec_w, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU));

        
    // preconditioner application: d_zm1 = U^-1 L^-1 d_r
    CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_lower, vec_rhs, vec_tmp, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        spsvDescrL) );
        
    CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper,
        vec_tmp, vec_resid,
        CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT,
        spsvDescrU));

    // GMRES initial residual
    // assumes here d_X is 0 initially => so r0 = b - Ax = b
    T nrm_b;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_rhs, 1, &nrm_b));
    T beta;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));

    printf("CSR cusparse precond |b| = %.4e, |M^-1 b| = %.8e\n", nrm_b, beta);

    // then free
    kmat.free();
    rhs.free();

    // free resources
    // cudaFree(pBuffer);
    // cusparseDestroyMatDescr(descr_L);
    // cusparseDestroyMatDescr(descr_U);
    // cusparseDestroyBsrsv2Info(info_L);
    // cusparseDestroyBsrsv2Info(info_U);
    // cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements

    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_tmp));
}

template <typename T>
void bsr_cpu_precond() {
    // get the kmat and rhs
    BsrMat<DeviceVec<T>> *kmat;
    DeviceVec<T> *rhs;
    get_ucrm_linear_system(kmat, rhs);

    // compare to TACS GMRES..

    // TODO : model how TACS does ILU(k) here..

    // pass the BSR mat back to the host

    // compute M the LU preconditioner in some CPU library


    // now compute |M^-1 b| with cusparse ILU(0)


    // report the init resid norm (print it)
    

    // then free
    kmat->free();
    rhs->free();
}

int main() {
    // test init |M^-1 b| on ILU preconditioner of uCRM linear shells matrix

    std::string ordering = "AMD"; // qorder
    std::string fill_type = "ILUk";
    int k = 3;
    bool print = false;

    bsr_cusparse_precond<double>(ordering, fill_type, k, print);
    csr_cusparse_precond<double>(ordering, fill_type, k, print);
};
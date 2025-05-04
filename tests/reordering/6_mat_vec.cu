#include "../../examples/plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "../test_commons.h"
#include <cassert>
#include <string>
#include <random>

// cusparse and cublas
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

int main(int argc, char* argv[]) {
    // prelim command line inputs
    // --------------------------

    // bool print = false;
    std::string ordering = argv[1];   // "none", "RCM", or "qorder"
    std::string fill_type = argv[2];  // "nofill", "ILUk", or "LU"

    if (argc < 3) {
        std::cerr << "Usage: ./program <ordering> <fill>\n";
        std::cerr << "Example: ./program qorder ILUk\n";
        return 1;
    }

    int rcm_iters = 5;
    double p_factor = 1.0;
    int k = 3;
    int nxe = 10;
    double fillin = 10.0;

    // ----------------------------------

    using T = double;   

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
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // first compute Kmat with no permutation and no fillin
    // ----------------------------------------------------

    // get bsr data and make copy of original for later
    auto& bsr_data0 = assembler.getBsrData();
    auto h_bsr_data0 = bsr_data0.createDeviceBsrData().createHostBsrData();
    auto bsr_data = bsr_data0.createDeviceBsrData().createHostBsrData();

    // Apply fill pattern
    if (fill_type == "nofill") {
        bsr_data0.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data0.compute_ILUk_pattern(k);
    } else if (fill_type == "LU") {
        bsr_data0.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return 1;
    }

    assembler.moveBsrDataToDevice();

    // assemble unpermuted kmat
    auto kmat0 = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    assembler.add_jacobian(res, kmat0);
    
    // then compute Kmat = P * Kmat * P^Twith permutation and no fillin
    // ----------------------------------------------------------------

    // reorder the bsr data
    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return 1;
    }

    // compute nofill and set new bsr data into it on the device
    // Apply fill pattern
    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return 1;
    }
    auto d_bsr_data = bsr_data.createDeviceBsrData();
    assembler.setBsrData(d_bsr_data);

    // assemble permuted kmat (nofill)
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    assembler.add_jacobian(res, kmat);

    // build unreordered and reordered vec
    // -----------------------------------

    // unreordered random vector
    int nvars = assembler.get_num_vars();
    std::random_device rd;  // seed source
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<T> test_vec(nvars);
    for (auto& val : test_vec) {
        val = dis(gen);
    }
    HostVec<T> h_test_vec(nvars, test_vec.data());
    auto d_test_vec = h_test_vec.createDeviceVec();

    // reordered random vector, iperm: old to new vals
    auto d_test_vec_perm = h_test_vec.createDeviceVec();
    d_test_vec_perm.permuteData(bsr_data.block_dim, d_bsr_data.iperm);

    // compare kmat*u before and after reordering
    // ------------------------------------------

    // create temp vecs for the products
    auto temp0 = assembler.createVarsVec();
    auto temp_perm = assembler.createVarsVec();

    // initial data needed for cusparse matrices
    int mb = bsr_data.nnodes, block_dim = bsr_data.block_dim;
    T *d_vals0 = kmat0.getPtr();
    T *d_vals_perm = kmat.getPtr();
    int *d_rowp0 = bsr_data0.rowp;
    int *d_cols0 = bsr_data0.cols;
    int *d_rowp_perm = d_bsr_data.rowp;
    int *d_cols_perm = d_bsr_data.cols;
    T *d_temp0 = temp0.getPtr();
    T *d_temp_perm = temp_perm.getPtr();
    T *d_u0 = d_test_vec.getPtr(); // final device pointer
    T *d_u_perm = d_test_vec_perm.getPtr();

    // create inital cublas and cusparse handles --------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // perform Kmat*test_vec unreordered
    T a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle, 
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        mb, mb, bsr_data0.nnzb,
        &a, descrA,
        d_vals0, d_rowp0, d_cols0,
        block_dim,
        d_u0,
        &b,
        d_temp0
    ));

    // perform Kmat*test_vec reordered
    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle, 
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        mb, mb, bsr_data.nnzb,
        &a, descrA,
        d_vals_perm, d_rowp_perm, d_cols_perm,
        block_dim,
        d_u_perm,
        &b,
        d_temp_perm
    ));

    // use perm : new rows => old rows on d_temp_perm
    temp_perm.permuteData(bsr_data.block_dim, d_bsr_data.perm);

    // offload from the device -----------------------

    /* copy device vecs to the host */
    auto h_temp0 = temp0.createHostVec();
    auto h_temp_perm = temp_perm.createHostVec();

    // final test result -----------------------------

    // test name
    std::string testName = "Kmat*u reordering consistency test, with ";

    testName += ordering;  // always include the ordering name

    if (fill_type == "ILUk") {
        testName += " ILU(" + std::to_string(k) + ")";
    } else if (fill_type == "nofill") {
        testName += " nofill";
    } else if (fill_type == "LU") {
        testName += " LU";
    }

    // now print out test report
    double mat_vec_err = rel_err(h_temp0, h_temp_perm);
    bool pass = mat_vec_err < 1e-5;
    printTestReport(testName, pass, mat_vec_err);
};
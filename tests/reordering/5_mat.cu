#include "../../examples/plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "../test_commons.h"
#include <cassert>
#include <string>
#include <list>

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

void test_mat_reordering(std::string ordering, std::string fill_type, bool print = false) {
    // bool print = false;
    // std::string ordering = argv[1];   // "none", "RCM", or "qorder"
    // std::string fill_type = argv[2];  // "nofill", "ILUk", or "LU"

    int rcm_iters = 5;
    double p_factor = 1.0;
    int k = 1; // for ILU(k)
    int nxe = 3;
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
    auto bsr_data = bsr_data0.createDeviceBsrData().createHostBsrData();

    if (print) {
        printf("unreordered pre fillin nnzb %d\n", bsr_data0.nnzb);
        printf("\t rowp: ");
        printVec<int>(bsr_data0.nnodes+1, bsr_data0.rowp);
        printf("\t cols: ");
        printVec<int>(bsr_data0.nnzb, bsr_data0.cols);
    }

    // Apply fill pattern
    if (fill_type == "nofill") {
        bsr_data0.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data0.compute_ILUk_pattern(k, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data0.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    if (print) {
        printf("unreordered nnzb %d\n", bsr_data0.nnzb);
        printf("\t rowp: ");
        printVec<int>(bsr_data0.nnodes+1, bsr_data0.rowp);
        printf("\t cols: ");
        printVec<int>(bsr_data0.nnzb, bsr_data0.cols);
    }

    // get bsr data on host after fillin for later kmat comprison check
    auto h_bsr_data_orig = bsr_data0.createDeviceBsrData().createHostBsrData();
    assembler.moveBsrDataToDevice();

    // assemble unpermuted kmat
    auto kmat0 = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    assembler.add_jacobian(res, kmat0);

    // get values off the device
    auto kmat0_vals = kmat0.getVec().createHostVec();

    if (print) printf("\n\n\n--------------------------------\ndone with first assembly\n--------------------------------\n\n\n");

    // then compute Kmat = P * Kmat * P^Twith permutation and no fillin
    // ----------------------------------------------------------------

    // reorder the bsr data
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

    if (print) {
        printf("reordered pre fillin nnzb %d\n", bsr_data.nnzb);
        printf("\t iperm: ");
        printVec<int>(bsr_data.nnodes, bsr_data.iperm);
        printf("\t rowp: ");
        printVec<int>(bsr_data.nnodes+1, bsr_data.rowp);
        printf("\t cols: ");
        printVec<int>(bsr_data.nnzb, bsr_data.cols);
    }

    // compute nofill and set new bsr data into it on the device
    // Apply fill pattern
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

    if (print) {
        printf("reordered nnzb %d\n", bsr_data.nnzb);
        printf("\t iperm: ");
        printVec<int>(bsr_data.nnodes, bsr_data.iperm);
        printf("\t rowp: ");
        printVec<int>(bsr_data.nnodes+1, bsr_data.rowp);
        printf("\t cols: ");
        printVec<int>(bsr_data.nnzb, bsr_data.cols);
    }

    auto d_bsr_data = bsr_data.createDeviceBsrData();
    assembler.setBsrData(d_bsr_data);

    // assemble permuted kmat (nofill)
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    assembler.add_jacobian(res, kmat);

    // get values off the device
    auto kmat_vals = kmat.getVec().createHostVec();

    // compare the kmat vals at each sparsity location
    // -----------------------------------------------

    // since iperm: old vals to new vals
    //       perm : new vals to old vals

    // first loop through original sparsity
    bool sparsity_pass = true; // sparsity pass should only be checked for nofill
    double K_abs_err = 0.0;
    double K0_norm = 0.0;
    for (int i = 0; i < kmat0_vals.getSize(); i++) {
        K0_norm = max(K0_norm, abs(kmat0_vals[i]));
    }
    int block_dim = bsr_data.block_dim;
    int block_dim2 = block_dim * block_dim;

    if (print) {
        printf("unreordered nnzb again %d\n", bsr_data0.nnzb);
        printf("\t rowp: ");
        printVec<int>(h_bsr_data_orig.nnodes+1, h_bsr_data_orig.rowp);
        printf("\t cols: ");
        printVec<int>(h_bsr_data_orig.nnzb, h_bsr_data_orig.cols);
    }

    for (int i = 0; i < bsr_data.nnodes; i++) {
        int i2 = bsr_data.iperm[i]; // old row => new row with iperm

        // loop through original cols
        for (int jp = h_bsr_data_orig.rowp[i]; jp < h_bsr_data_orig.rowp[i+1]; jp++) {
            int j = h_bsr_data_orig.cols[jp];
            

            // loop through new cols to find match
            int jp2 = -1, j2 = -1;
            for (int _jp2 = bsr_data.rowp[i2]; _jp2 < bsr_data.rowp[i2+1]; _jp2++) {
                int _j2 = bsr_data.cols[_jp2]; // new cols
                if (bsr_data.iperm[j] == _j2) { // iperm[old col] == new col
                    jp2 = _jp2;
                    j2 = _j2;
                    break;
                }
            }
            // check found matching column, sparsity in agreement
            if (jp2 == -1 || j2 == -1) {
                sparsity_pass = false;
                if (print) printf("orig brow, bcol = %d, %d don't have matching sparsity\n", i, j);
            } else {
                // compute abs err among K values now, looping through the block
                for (int ii = 0; ii < block_dim2; ii++) {
                    double val0 = kmat0_vals[block_dim2 * jp + ii];
                    double valp = kmat_vals[block_dim2 * jp2 + ii];
                    // int ind1 = block_dim2 * jp + ii;
                    // int ind2 = block_dim2 * jp2 + ii;
                    // if (ind1 != ind2) {
                    //     printf("in K_abs_err, this ind not equal %d, %d\n", ind1, ind2);
                    // }
                    double c_abs_err = abs_err(val0, valp);
                    K_abs_err = max(K_abs_err, c_abs_err);
                }
            }
        }
    }

    // test results --------------------------
    
    // printout host kmat comparisons for debugging
    if (print && nxe <= 5) {

        printf("nnzb compare: %d to %d\n", h_bsr_data_orig.nnzb, bsr_data.nnzb);
        printf("cols compare\n");
        printf("\tunreordered: ");
        printVec<int>(h_bsr_data_orig.nnzb, h_bsr_data_orig.cols);
        printf("\treordered: ");
        printVec<int>(bsr_data.nnzb, bsr_data.cols);


        // very useful printout
        // printf("kmat unperm vs perm\n");
        T max_rerr = 0.0;
        int max_rerr_ind = -1;
        for (int i = 0; i < kmat0_vals.getSize(); i++) {
            T val1 = kmat0_vals[i];
            T val2 = kmat_vals[i];
            T m_rel_err = abs_err(val1, val2);
            // printf("%.4e %.4e, rel err %.4e\n", val1, val2, m_rel_err);
            if (m_rel_err > max_rerr) {
                max_rerr = m_rel_err;
                max_rerr_ind = i;
            }
        }

        // get the brow, bcol and element #

        printf("max rel err %.4e at ind %d\n", max_rerr, max_rerr_ind);
        // check the location in brow, bcol and in element stiffness matrices..
        for (int i = 0; i < bsr_data.nnodes; i++) {
            int brow = i;
            for (int j = bsr_data.rowp[i]; j < bsr_data.rowp[i+1]; j++) {
                int bcol = bsr_data.cols[j];
                for (int ii = 0; ii < 36; ii++) {
                    int ind = 36 * j + ii;
                    int irow = ii / 6;
                    int icol = ii % 6;
                    if (ind == max_rerr_ind) {
                        printf("max rel err ind %d occurs at brow %d, bcol %d, inner_row %d, inner_col %d, with vals %.4e %.4e\n", max_rerr_ind, 
                            brow, bcol, irow, icol, kmat0_vals[ind], kmat_vals[ind]);
                    }
                }
            }
        }
    }

    // test name
    std::string testName = "K vs PKP^T consistency test, with ";

    testName += ordering;  // always include the ordering name

    if (fill_type == "ILUk") {
        testName += " ILU(" + std::to_string(k) + ")";
    } else if (fill_type == "nofill") {
        testName += " nofill";
    } else if (fill_type == "LU") {
        testName += " LU";
    }

    // now print out test report
    // rel err is very sensitive to small entries in matrix
    double Kmat_err = K_abs_err / K0_norm;
    bool vals_pass = Kmat_err < 1e-5;
    // sparsity may be different between LU and ILUk, but nofill should match
    sparsity_pass = sparsity_pass || fill_type != "nofill";
    bool pass = vals_pass && sparsity_pass;
    printTestReport(testName, pass, Kmat_err);
    if (print) printf("\tmax abs err on K %.4e, K norm %.4e\n", K_abs_err, K0_norm);
}

int main() {
    // turn off test all for debugging
    bool test_all = true; // test all since tests should work

    if (test_all) {
        std::list<std::string> list1 = {"none", "RCM", "AMD", "qorder"};
        std::list<std::string> list2 = {"nofill", "ILUk", "LU"};

        for (auto it2 = list2.begin(); it2 != list2.end(); ++it2) {
            for (auto it1 = list1.begin(); it1 != list1.end(); ++it1) {
                test_mat_vec_product(*it1, *it2);
            }
        }
    } else {
        // test single failing test
        test_mat_vec_product("RCM", "nofill", true);
    }  
};
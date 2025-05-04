#include "../../examples/plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "../test_commons.h"
#include <cassert>
#include <string>


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

int main(int argc, char* argv[]) {
    // prelim command line inputs
    // --------------------------

    bool print = false;
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

    // get values off the device
    auto kmat0_vals = kmat0.getVec().createHostVec();

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

    // get values off the device
    auto kmat_vals = kmat.getVec().createHostVec();

    // compare the kmat vals at each sparsity location
    // -----------------------------------------------

    // since iperm: old vals to new vals
    //       perm : new vals to old vals

    // first loop through original sparsity
    bool sparsity_pass = true;
    double K_abs_err = 0.0;
    int block_dim = bsr_data.block_dim;
    int block_dim2 = block_dim * block_dim;
    for (int i = 0; i < bsr_data.nnodes; i++) {
        int i2 = bsr_data.iperm[i]; // old row => new row with iperm
        // loop through original cols
        for (int jp = h_bsr_data0.rowp[i]; jp < h_bsr_data0.rowp[i+1]; jp++) {
            int j = h_bsr_data0.cols[jp];
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
                    double c_abs_err = abs(valp - val0);
                    K_abs_err = max(K_abs_err, c_abs_err);
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
    bool vals_pass = K_abs_err < 1e0;
    bool pass = vals_pass && sparsity_pass;
    printTestReport(testName, pass, K_abs_err);
};
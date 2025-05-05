#include "coupled/meld.h"
#include "../../examples/plate/_plate_utils.h"
#include "../test_commons.h"

int main() {

    using T = double;
    bool write_vtk = false;
    bool print = false;
    
    // setup 
    T Lx = 1.0, Ly = 1.0;
    int nnx_a = 31, nny_a = 31;
    int nnx_s = 37, nny_s = 37;

    // auto xs0 = makeGridMesh<T>(nnx_s, nny_s, Lx, Ly, -0.2);
    // auto xa0 = makeGridMesh<T>(nnx_a, nny_a, Lx, Ly,-0.15);

    auto xs0 = makeGridMesh<T>(nnx_s, nny_s, Lx, Ly, 0.01);
    auto xa0 = makeGridMesh<T>(nnx_a, nny_a, Lx, Ly, 0.01);

    // prescribed displacements
    auto h_us = makeInPlaneShearDisp<T>(xs0, 20.0);
    // auto us = makeCustomDisp<T>(xs0, 0.1);
    // auto ua = HostVec<T>(xa0.getSize());
    // printVec<T>(N, )

    // convert to device vecs
    auto d_xa0 = xa0.createDeviceVec();
    auto d_xs0 = xs0.createDeviceVec();
    auto d_us = h_us.createDeviceVec();

    // auto h_xs0 = d_xs0.createHostVec();
    // printVec<double>(10, h_xs0.getPtr());
    // return 0;
    
    // create MELD
    T beta = 0.1;
    // T beta = 3.0;
    static constexpr int NN_MAX = 32; // choose a multiple of 32 if you can
    int sym = -1; // no symmetry yet I believe
    int nn = 128; // 32, 64, 256
    double Hreg = 1e-4;
    static constexpr bool oneshot = false;
    constexpr bool exact_givens = true; // important to be True for good load transfer
    auto meld = MELD<T, NN_MAX, false, oneshot, exact_givens>(d_xs0, d_xa0, beta, nn, sym, Hreg);
    meld.initialize();

    // transfer disps
    auto d_ua = meld.transferDisps(d_us);
    auto h_ua = d_ua.createHostVec();

    // printf("us:");
    // printVec<T>(10, h_us.getPtr());
    // printf("ua:");
    // printVec<T>(2700, h_ua.getPtr());

    // visualize one of the meshes in paraview
    if (write_vtk) printGridToVTK<T>(nnx_s, nny_s, xs0, h_us, "out/xs.vtk");
    if (write_vtk) printGridToVTK<T>(nnx_a, nny_a, xa0, h_ua, "out/xa.vtk");

    // now setup loads and do load transfer
    auto h_fa = makeInPlaneShearDisp<T>(xa0, 10.0);
    auto d_fa = h_fa.createDeviceVec();

    auto d_fs = meld.transferLoads(d_fa);
    auto h_fs = d_fs.createHostVec();
    
    // printf("h_fs:");
    // printVec<double>(h_fs.getSize(), h_fs.getPtr());

    if (write_vtk) printGridToVTK<T>(nnx_a, nny_a, xa0, h_fa, "out/fa.vtk");
    if (write_vtk) printGridToVTK<T>(nnx_s, nny_s, xs0, h_fs, "out/fs.vtk");

    // compute total aero forces
    T fA_tot[3], fS_tot[3];
    int na = h_fa.getSize() / 3;
    int ns = h_fs.getSize() / 3;
    memset(fA_tot, 0.0, 3 * sizeof(T));
    memset(fS_tot, 0.0, 3 * sizeof(T));

    for (int ia = 0; ia < na; ia++) {
        for (int idim = 0; idim < 3; idim++) {
            fA_tot[idim] += h_fa[3 * ia + idim];
        }
    }

    for (int is = 0; is < ns; is++) {
        for (int idim = 0; idim < 3; idim++) {
            fS_tot[idim] += h_fs[3 * is + idim];
        }
    }

    if (print) {
        printf("fA_tot:");
        printVec<double>(3, &fA_tot[0]);

        printf("fS_tot:");
        printVec<double>(3, &fS_tot[0]);
    }

    // compute total work done
    T W_A = 0.0, W_S = 0.0;
    for (int ia = 0; ia < na; ia++) {
        for (int idim = 0; idim < 3; idim++) {
            W_A += h_fa[3 * ia + idim] * h_ua[3 * ia + idim];
        }
    }
    for (int is = 0; is < ns; is++) {
        for (int idim = 0; idim < 3; idim++) {
            W_S += h_fs[3 * is + idim] * h_us[3 * is + idim];
        }
    }

    // compute force rel errors as abs error across vec / norm of ref vec f_S
    T abs_err = 0.0, ref_norm = 0.0;
    for (int i = 0; i < 3; i++) {
        abs_err = std::max(abs_err, abs(fA_tot[i] - fS_tot[i]));
        ref_norm = std::max(ref_norm, fS_tot[i]);
    }
    T F_rel_err = abs_err / ref_norm;

    // compute the relative error
    T W_rel_err = abs(W_A - W_S) / abs(W_S);

    if (print) printf("W_A %.6e, W_S %.6e, rel err %.6e\n", W_A, W_S, W_rel_err);

    // test result 
    // -------------------
    T ovr_rel_err = std::max(W_rel_err, F_rel_err);
    // need slightly more accurate SVD jacobian to get more work and force conservation precision
    bool passed = F_rel_err < 1e-3 && W_rel_err < 1e-3;
    printTestReport<T>("MELD plate conservation test", passed, ovr_rel_err);
    printf("\tW rel err %.4e, F rel err %.4e\n", W_rel_err, F_rel_err);

    return 0;
};
#include "coupled/meld.h"
#include "_loc_utils.h"

int main() {

    using T = double;
    
    // setup 
    T Lx = 1.0, Ly = 1.0;
    int nnx_a = 30, nny_a = 30;
    int nnx_s = 17, nny_s = 17;

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
    T beta = 10.0;
    // T beta = 3.0;
    int nn = 32;
    int sym = -1; // no symmetry yet I believe
    // double Hreg = 1e-1; // regularization for H
    double Hreg = 0.0;
    auto meld = MELD<T>(d_xs0, d_xa0, beta, nn, sym, Hreg);
    meld.initialize();

    // transfer disps
    auto d_ua = meld.transferDisps(d_us);
    auto h_ua = d_ua.createHostVec();

    printf("us:");
    printVec<T>(10, h_us.getPtr());
    // printf("ua:");
    // printVec<T>(2700, h_ua.getPtr());

    // visualize one of the meshes in paraview
    printGridToVTK<T>(nnx_s, nny_s, xs0, h_us, "out/xs.vtk");
    printGridToVTK<T>(nnx_a, nny_a, xa0, h_ua, "out/xa.vtk");

    // now setup loads and do load transfer
    auto h_fa = makeInPlaneShearDisp<T>(xa0, 10.0);
    auto d_fa = h_fa.createDeviceVec();

    auto d_fs = meld.transferLoads(d_fa);
    auto h_fs = d_fs.createHostVec();
    
    // printf("h_fs:");
    // printVec<double>(h_fs.getSize(), h_fs.getPtr());

    printGridToVTK<T>(nnx_a, nny_a, xa0, h_fa, "out/fa.vtk");
    printGridToVTK<T>(nnx_s, nny_s, xs0, h_fs, "out/fs.vtk");

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

    printf("fA_tot:");
    printVec<double>(3, &fA_tot[0]);

    printf("fS_tot:");
    printVec<double>(3, &fS_tot[0]);

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

    printf("W_A %.4e, W_S %.4e\n", W_A, W_S);

    return 0;
};
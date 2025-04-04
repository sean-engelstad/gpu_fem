#include "transfer/lin_meld.h"
#include "loc_utils.h"


int main() {

    using T = double;
    
    // setup 
    T Lx = 1.0, Ly = 1.0;
    int nnx_a = 30, nny_a = 30;
    int nnx_s = 17, nny_s = 17;

    auto xs0 = makeGridMesh<T>(nnx_s, nny_s, Lx, Ly, -0.2);
    auto xa0 = makeGridMesh<T>(nnx_a, nny_a, Lx, Ly, -0.1);

    // prescribed displacements
    auto us = makeInPlaneShearDisp<T>(xs0, 20.0);
    // auto us = makeCustomDisp<T>(xs0, 0.1);
    // auto ua = HostVec<T>(xa0.getSize());
    // printVec<T>(N, )

    // convert to device vecs
    auto d_xa0 = xa0.createDeviceVec();
    auto d_xs0 = xs0.createDeviceVec();
    auto d_us = us.createDeviceVec();

    // auto h_xs0 = d_xs0.createHostVec();
    // printVec<double>(10, h_xs0.getPtr());
    // return 0;
    
    // create MELD
    T beta = 20.0;
    int nn = 32;
    int sym = 0;
    auto meld = LinearizedMELD<T>(d_xs0, d_xa0, beta, nn, sym);
    meld.initialize();
    // return 0;

    // perform disp transfer
    // return 0;
    auto d_ua = meld.transferDisps(d_us);
    // return 0;

    // copy out of device
    auto h_ua = d_ua.createHostVec();

    // visualize one of the meshes in paraview
    printGridToVTK<T>(nnx_s, nny_s, xs0, us, "xs.vtk");
    printGridToVTK<T>(nnx_a, nny_a, xa0, h_ua, "xa.vtk");

    // auto xs = meld.getStructDeformed();
    // auto h_xs = xs.createHostVec();
    // printGridToVTK<T>(nnx_s, nny_s, h_xs, us, "xs_def.vtk");

    // now setup loads and we'll do a load transfer
    auto h_fa = makeInPlaneShearDisp<T>(xa0, 10.0);
    auto d_fa = h_fa.createDeviceVec();

    auto d_fs = meld.transferLoads(d_fa);
    auto h_fs = d_fs.createHostVec();

    printGridToVTK<T>(nnx_a, nny_a, xa0, h_fa, "fa.vtk");
    printGridToVTK<T>(nnx_s, nny_s, xs0, h_fs, "fs.vtk");

    return 0;
};
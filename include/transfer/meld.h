
#pragma once
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "locate_point.h"
#ifdef USE_GPU
#include "meld.cuh"
#endif

// intended for the GPU here, later could generalize it if you want
// main MELD paper https://arc.aiaa.org/doi/epdf/10.2514/1.J057318
// this is the nonlinear version of MELD, there's also a linearized form
// the source code is in the FUNtoFEM github,
// https://github.com/smdogroup/funtofem/blob/master/src/MELD.cpp

template <typename T>
class MELDTransfer {
   public:
    MELDTransfer(DeviceVec<T> &xs0, DeviceVec<T> &xa0, T beta, int num_nearest, int sym)
        : xs0(xs0), xa0(xa0), beta(beta), nn(num_nearest), sym(sym) {
        // assumes 3D so that xs0, xa0 are 3*nS, 3*nA sizes
        ns = xs0.getSize() / 3;
        na = xa0.getSize() / 3;

        xs = DeviceVec<T>(3 * ns);
        us = DeviceVec<T>(3 * ns);

        xa = DeviceVec<T>(3 * na);
        ua = DeviceVec<T>(3 * na);

        // auto h_xs0 = xs0.createHostVec();
        // printf("h_xs0 constructor:");
        // printVec<double>(10, h_xs0.getPtr());
    }

    __HOST__ DeviceVec<T> &getStructDisps() { return us; }
    __HOST__ DeviceVec<T> &getAeroDisps() { return ua; }
    __HOST__ DeviceVec<T> &getStructDeformed() { return xs; }

    __HOST__ void initialize() {
        printf("inside initialize\n");

        // auto h_xs0 = xs0.createHostVec();
        // printf("h_xs0:");
        // printVec<double>(10, h_xs0.getPtr());
        // return;

        // was going to maybe compute aero struct connectivity (nearest neighbors)
        // using octree, but instead just going to reuse CPU code for this from F2F MELD
        computeAeroStructConn();
        printf("\tfinished aero struct conn\n");

        // compute aero struct connectivity (DEPRECATED here)
        // aerostruct_conn = DeviceVec<int>(nn * na);
        // dim3 block(128); // number of struct nodes considered at a time
        // int nblocks = (ns + block.x) / block.x;
        // dim3 grid(na);
        // compute_aerostruct_conn<double><<<grid, block>>>(nn, xa0, xs0, aerostruct_conn);

        // compute weights (assumes fixed here even) => reinitialize under shape change
        weights = DeviceVec<double>(nn * na);
        dim3 block(32);
        dim3 grid(na);

        // debug with 1 thread
        // dim3 block(32);
        // dim3 grid(1);

        // auto h_conn = aerostruct_conn.createHostVec();
        // printf("h_conn");
        // printVec<int>(h_conn.getSize(), h_conn.getPtr());

        compute_weights_kernel<T><<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, weights);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished weights kernel\n");

        // auto h_weights = weights.createHostVec();
        // printVec<double>(h_weights.getSize(), h_weights.getPtr());
    }

    __HOST__ void computeAeroStructConn() {
        HostVec<int> conn(nn * na);
        // printf("inside aero struct conn\n");
        auto h_xa0 = xa0.createHostVec();

        // TODO : later do sym case
        auto h_xs = xs0.createHostVec();
        // printVec<double>(10, h_xs.getPtr());
        // return;

        int min_bin_size = 10;
        int npts = h_xs.getSize() / 3;
        auto *locator = new LocatePoint<T>(h_xs.getPtr(), npts, min_bin_size);
        // printf("created locate point\n");

        int *indx = new int[nn];
        T *dist = new T[nn];

        // For each aerodynamic node, copy the indices of the nearest n structural
        // nodes into the conn array
        for (int i = 0; i < na; i++) {
            T loc_xa0[3];
            memcpy(loc_xa0, &h_xa0[3 * i], 3 * sizeof(T));

            locator->locateKClosest(nn, indx, dist, loc_xa0);

            for (int k = 0; k < nn; k++) {
                // not doing reflections for now
                // if (indx[k] >= ns) {
                //     int kk = indx[k] - ns;
                //     conn[nn * i + k] = locate_to_reflected_index[kk];
                // } else {
                //     conn[nn * i + k] = indx[k];
                // }
                conn[nn * i + k] = indx[k];
            }

            // if (i < 3) {
            // printf("node %d conn:", i);
            // printVec<int>(nn, &conn[nn * i]);
            // }
        }

        // cleanup
        // if (Xs_dup) {
        //     delete[] Xs_dup;
        // }
        if (indx) {
            delete[] indx;
        }
        if (dist) {
            delete[] dist;
        }
        delete locator;

        // printf("conn:");
        // printVec<int>(nn * na, conn.getPtr());

        // then copy onto the device in
        aerostruct_conn = conn.createDeviceVec();

        // auto h_conn = aerostruct_conn.createHostVec();
        // printf("h_conn:");
        // printVec<int>(nn * na, h_conn.getPtr());
    }

    __HOST__ DeviceVec<T> &transferDisps(DeviceVec<T> &new_us) {
        printf("inside transferDisps\n");
        new_us.copyValuesTo(us);
        xs.zeroValues();

        // auto h_us = us.createHostVec();
        // printf("h_us:");
        // printVec<double>(h_us.getSize(), h_us.getPtr());

        // add us to xs0
        // use cublas or a custom kernel here for axpy?
        dim3 block1(32);
        int nblocks = (3 * ns + block1.x - 1) / block1.x;
        dim3 grid1(nblocks);
        vec_add_kernel<T><<<grid1, block1>>>(us, xs0, xs);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished vec_add_kernel\n");

        // auto h_xs = xs.createHostVec();
        // printf("h_xs:");
        // printVec<double>(h_xs.getSize(), h_xs.getPtr());

        // zero out xa, ua before we change them
        xa.zeroValues();
        ua.zeroValues();

        // transfer disps on the kernel for each aero node
        // dim3 block2(128);  // power of 2 larger than the # nearest neighbors?
        // dim3 grid2(na);

        dim3 block2(32);
        dim3 grid2(na);

        // debug launch
        // dim3 block2(1);
        // dim3 grid2(1);

        // return ua;

        transfer_disps_kernel<T>
            <<<grid2, block2>>>(nn, aerostruct_conn, weights, xs0, xs, xa0, xa, ua);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished transfer_disps_kernel\n");

        auto h_ua = ua.createHostVec();
        // printf("h_ua:");
        // printVec<double>(h_ua.getSize(), h_ua.getPtr());

        printf("h_ua at node 899:");
        printVec<double>(3, &h_ua[3 * 899]);

        return ua;
    }

    __HOST__ DeviceVec<T> transferLoads(DeviceVec<T> new_fa) {
        new_fa.copyValuesTo(fa);
        fs.zeroValues();

        dim3 block(128);
        dim3 grid(na);
        transfer_loads_kernel<T>
            <<<grid, block>>>(nn, aerostruct_conn, weights, xs0, xs, xa0, xa, fa, fs);

        return fs;
    }

   private:
    DeviceVec<T> xs0, xa0, weights;
    DeviceVec<T> xs, xa;
    DeviceVec<T> us, ua;
    DeviceVec<T> fs, fa;
    DeviceVec<int> aerostruct_conn;
    double beta;
    int sym, nn;
    int na, ns;
};
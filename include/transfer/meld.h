
#pragma once
#include "cuda_utils.h"
#include "linalg/vec.h"
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
    MELDTransfer(DeviceVec<T> xs0, DeviceVec<T> xa0, T beta, int num_nearest, int sym)
        : xs0(xs0), xa0(xa0), beta(beta), nn(num_nearest), sym(sym) {
        // assumes 3D so that xs0, xa0 are 3*nS, 3*nA sizes
        ns = xs0.getSize() / 3;
        na = xa0.getSize() / 3;
    }

    __HOST__ void initialize() {
        // was going to maybe compute aero struct connectivity (nearest neighbors)
        // using octree, but instead just going to reuse CPU code for this from F2F MELD
        computeAeroStructConn();

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

        compute_weights_kernel<T><<<grid, block>>>(..., weights);
    }

    __HOST__ void computeAeroStructConn() {
        HostVec<int> conn(nn * na);

        // TODO : later do sym case
        T *Xs_dup = NULL;
        int num_locate_nodes = 0;

        int min_bin_size = 10;
        LocatePoint *locator = new LocatePoint<T>(Xs_dup, num_locate_nodes, min_bin_size);

        int *indx = new int[nn];
        T *dist = new T[nn];

        // For each aerodynamic node, copy the indices of the nearest n structural
        // nodes into the conn array
        for (int i = 0; i < na; i++) {
            T xa0[3];
            memcpy(xa0, &Xa[3 * i], 3 * sizeof(T));

            locator->locateKClosest(nn, indx, dist, xa0);

            for (int k = 0; k < nn; k++) {
                if (indx[k] >= ns) {
                    int kk = indx[k] - ns;
                    conn[nn * i + k] = locate_to_reflected_index[kk];
                } else {
                    conn[nn * i + k] = indx[k];
                }
            }
        }

        // cleanup
        if (Xs_dup) {
            delete[] Xs_dup;
        }
        if (indx) {
            delete[] indx;
        }
        if (dist) {
            delete[] dist;
        }
        delete locator;

        // then copy onto the device in
        aerostruct_conn = conn.createDeviceVec();
    }

    __HOST__ void transferDisps(DeviceVec<T> new_us) {
        new_us.copyValuesTo(us);

        // add us to xs0
        // use cublas or a custom kernel here for axpy?
        dim3 block1(256);
        int nblocks = (ns + block1.x - 1) / block1.x;
        dim3 grid1(nblocks);
        vec_add_kernel<T><<<grid1, block1>>>(us, xs0, xs);

        // zero out xa, ua before we change them
        xa.zeroValues();
        ua.zeroValues();

        // transfer disps on the kernel for each aero node
        dim3 block2(128);  // power of 2 larger than the # nearest neighbors?
        dim3 grid2(na);
        transfer_disps_kernel<T>
            <<<grid2, block2>>>(nn, aerostruct_conn, weights, xs0, xs, us, xa0, xa, ua);
    }

    __HOST__ void transferLoads(DeviceVec<T> new_fa) {
        new_fa.copyValuesTo(fa);
        fs.zeroValues();

        dim3 block(128);
        dim3 grid(na);
        transfer_loads_kernel<T>
            <<<grid, block>>>(nn, aerostruct_conn, weights, xs0, xs, xa0, xa, fa, fs);
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
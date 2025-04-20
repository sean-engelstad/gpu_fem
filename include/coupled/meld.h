
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

template <typename T, int NN_MAX_ = 256, bool linear_ = false>
class MELD {
    static constexpr int NN_MAX = NN_MAX_;  // want NN_MAX to be a multiple of 32
    static constexpr bool linear = linear_;

   public:
    MELD(DeviceVec<T> &xs0, DeviceVec<T> &xa0, T beta, int num_nearest, int sym, T H_reg,
         bool print = false)
        : xs0(xs0), xa0(xa0), beta(beta), nn(num_nearest), sym(sym) {
        // assumes 3D so that xs0, xa0 are 3*nS, 3*nA sizes
        ns = xs0.getSize() / 3;
        na = xa0.getSize() / 3;

        xs = DeviceVec<T>(3 * ns);
        us = DeviceVec<T>(3 * ns);

        xa = DeviceVec<T>(3 * na);
        ua = DeviceVec<T>(3 * na);

        fa = DeviceVec<T>(3 * na);
        fs = DeviceVec<T>(3 * ns);

        H_reg = H_reg;
        this->print = print;

        // auto h_xs0 = xs0.createHostVec();
        // printf("h_xs0 constructor:");
        // printVec<double>(10, h_xs0.getPtr());
    }

    template <int N0 = 6, int NF = 3>
    static HostVec<T> extractVarsVec(int nnodes, HostVec<T> vec) {
        HostVec<T> vec2(NF * nnodes);
        for (int inode = 0; inode < nnodes; inode++) {
            for (int j = 0; j < NF; j++) {
                vec2[NF * inode + j] = vec[N0 * inode + j];
            }
        }
        return vec2;
    }

    template <int N0 = 3, int NF = 6>
    static HostVec<T> expandVarsVec(int nnodes, HostVec<T> vec) {
        HostVec<T> vec2(NF * nnodes);
        for (int inode = 0; inode < nnodes; inode++) {
            for (int j = 0; j < N0; j++) {
                vec2[NF * inode + j] = vec[N0 * inode + j];
            }
        }
        return vec2;
    }

    __HOST__ DeviceVec<T> &getStructDisps() { return us; }
    __HOST__ DeviceVec<T> &getAeroDisps() { return ua; }
    __HOST__ DeviceVec<T> &getStructDeformed() { return xs; }

    __HOST__ void initialize() {
        if (print) printf("inside initialize\n");

        // was going to maybe compute aero struct connectivity (nearest neighbors)
        // using octree, but instead just going to reuse CPU code for this from F2F MELD
        computeAeroStructConn();
        if (print) printf("\tfinished aero struct conn\n");

        // compute weights (assumes fixed here even) => reinitialize under shape change
        weights = DeviceVec<double>(nn * na);
        dim3 block(NN_MAX);
        dim3 grid(na);

        compute_weights_kernel<T, NN_MAX>
            <<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, weights);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tfinished weights kernel\n");

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
        if (print) printf("inside transferDisps\n");
        new_us.copyValuesTo(us);
        xs.zeroValues();

        // add us to xs0
        // use cublas or a custom kernel here for axpy?
        dim3 block1(32);
        int nblocks = (3 * ns + block1.x - 1) / block1.x;
        dim3 grid1(nblocks);
        vec_add_kernel<T><<<grid1, block1>>>(us, xs0, xs);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tfinished vec_add_kernel\n");

        // zero out xa, ua before we change them
        xa.zeroValues();
        ua.zeroValues();

        dim3 block2(NN_MAX);
        dim3 grid2(na);
        // dim3 block2(1);
        // dim3 grid2(1);

        transfer_disps_kernel<T, NN_MAX>
            <<<grid2, block2>>>(nn, H_reg, aerostruct_conn, weights, xs0, xs, xa0, ua);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tfinished transfer_disps_kernel\n");

        dim3 block4(32);
        nblocks = (3 * na + block4.x - 1) / block4.x;
        dim3 grid4(nblocks);
        vec_add_kernel<T><<<grid4, block4>>>(ua, xa0, xa);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tfinished vec_add_kernel for ua\n");

        return ua;
    }

    __HOST__ DeviceVec<T> transferLoads(DeviceVec<T> new_fa) {
        if (print) printf("inside transferLoads\n");
        new_fa.copyValuesTo(fa);
        fs.zeroValues();

        // auto h_fa = fa.createHostVec();
        // printf("fa:");
        // printVec<double>(h_fa.getSize(), h_fa.getPtr());

        dim3 block(NN_MAX, 3);  // one warp for parallelization by spatial_dim
        dim3 grid(na);

        // dim3 block(1, 1);
        // dim3 grid(1);

        if (print) printf("launch transfer_loads_kernel\n");
        transfer_loads_kernel<T, NN_MAX, linear>
            <<<grid, block>>>(nn, H_reg, aerostruct_conn, weights, xs0, xs, xa0, xa, fa, fs);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tdone with transfer_loads_kernel\n");

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
    T H_reg;
    bool print;
};
#pragma once
#include "linalg/svd_utils.h"
#include "a2dcore.h"

template <typename T>
__GLOBAL__ void compute_weights_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    // printf("xs0:");
    // printVec<double>(xs0.getSize(), xs0.getPtr());

    // printf("inside host weights kernel\n");

    #define NN 32 // need nn < 64 then
    // may want to check this

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T sum_distsq[1];
    __SHARED__ T sum[1];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    __syncthreads();

    // printf("loc_conn\n");
    // printVec<int>(nn, &loc_conn[0]);
    // printf("loc_xs0\n");
    // printVec<double>(3 * nn, &loc_xs0[0]);
    // printf("loc_xa0\n");
    // printVec<double>(3, &loc_xa0[0]);

    // compute avg dist_sq for normalization
    memset(sum_distsq, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        atomicAdd(&sum_distsq[0], distsq);
    }
    __syncthreads();
    T avg_distsq = sum_distsq[0] / (1.0 * nn);
    avg_distsq += 1e-7;
    // printf("avg_distsq = %.4e\n", avg_distsq);

    // compute weights (un-normalized first)
    memset(loc_w, 0.0, nn * sizeof(T));
    memset(sum, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq / avg_distsq);
        loc_w[inode] += new_weight;
        atomicAdd(&sum[0], new_weight);
    }
    __syncthreads();

    // normalize the weights so it becomes a partition of unity
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        loc_w[inode] /= sum[0];
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("aero_ind %d\n", aero_ind);
    //     printf("loc_w\n");
    //     printVec<double>(nn, &loc_w[0]);
    // }

    int global_start = aero_ind * nn;
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        atomicAdd(&weights[global_start+inode], loc_w[inode]);
        // weights[global_start + inode] = loc_w[inode];
    }
}

template <typename T>
__GLOBAL__ void transfer_disps_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> xs, DeviceVec<T> xa0,
                                      DeviceVec<T> ua) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    #define NN 32 // need nn < 64 then
    // may want to check this
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xs[3 * NN];
    __SHARED__ T loc_xa0[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T xs_bar[3];
    __SHARED__ T H[9];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs[0]);
    __syncthreads();

    // all of the following will be reduction-like operations
    // until H is computed (each thread helps in the work)

    // compute the centroids of the nearest neighbors
    // xs0_bar
    memset(&xs0_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * nn; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs0_bar[idim], loc_w[inode] * loc_xs0[i]);
    }

    // xs_bar
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * nn; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        // if (threadIdx.x == 0) {
        //     printf("loc_w[%d] = %.4e\n", inode, loc_w[inode]);
        //     printf("loc_xs[%d] = %.4e\n", i, loc_xs[i]);
        // }
        atomicAdd(&xs_bar[idim], loc_w[inode] * loc_xs[i]);
    }
    __syncthreads();

    // printf("xs0_bar:");
    // printVec<double>(3, &xs0_bar[0]);
    // printf("xs_bar:");
    // printVec<double>(3, &xs_bar[0]);

    // compute covariance H (reduction step across threads)
    memset(&H[0], 0.0, 9 * sizeof(T));
    for (int i = threadIdx.x; i < 9 * nn; i += blockDim.x) {
        int inode = i / 9;
        int i9 = i % 9;
        int idim = i9 / 3;
        int jdim = i9 % 3;

        // H += w_{inode} * p_{inode,idim} * q_{inode,jdim}
        // where p = xS - xSbar for each node and dim, sim q = xS0 - xS0bar
        // could do warp shuffling here
        T h_val = loc_w[inode] * (loc_xs[3 * inode + idim] - xs_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // diagonalization of u for stability
    // change later to use trace(H)
    H[0] += H_reg;
    H[4] += H_reg;
    H[8] += H_reg;

    if ((aero_ind == 522 || aero_ind == 521) && threadIdx.x == 0) {
        printf("H[%d]: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", aero_ind, H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
        // printVec<T>(3, loc_ua);
    }

    // after computing H, each thread is going to do the same thing
    // and we'll just average the result (same answer), but computations are cheap

    // get SVD of H (call device function in svd_utils.h)
    T R[9];
    // bool print = (aero_ind == 522 || aero_ind == 521) && threadIdx.x == 0;
    bool print = (aero_ind == 521) && threadIdx.x == 0;
    // bool print = false;
    computeRotation<T>(H, R, print);

    // if (print) {
    // printf("R[%d]: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", aero_ind, R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
        // printVec<T>(3, loc_ua);
    // }

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, r);

    // perform rotation and translations
    // rho = R * r + t
    T rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T, 3, 3>(R, r, rho);

    T loc_xa[3];
    A2D::VecSumCore<T, 3>(xs_bar, rho, loc_xa);

    T loc_ua[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa, -1.0, loc_xa0, loc_ua);

    // debug states
    T loc_us[3];
    A2D::VecSumCore<T, 3>(1.0, xs_bar, -1.0, xs0_bar, loc_us);

    // if ((aero_ind == 522 || aero_ind == 521) && threadIdx.x == 0) {
    //     printf("loc_ua[%d]: %.4e %.4e %.4e\n", aero_ind, loc_ua[0], loc_ua[1], loc_ua[2]);
    //     // printVec<T>(3, loc_ua);
    // }

    // update xa and u0 globally with add reduction by the blockDim.x
    // int nb = blockDim.x;
    // should probably do warp shuffle here among the threads to speed it up
    // before atomic add
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++) {
            // atomicAdd(&xa[3 * aero_ind+i], loc_xa[i]);
            atomicAdd(&ua[3 * aero_ind+i], loc_ua[i]);
        }
    }
}

template <typename T>
__GLOBAL__ void transfer_loads_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> us, DeviceVec<T> xa0,
                                      DeviceVec<T> xa, DeviceVec<T> fa, DeviceVec<T> fs) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    using T2 = A2D::ADScalar<T, 1>;

    #define NN 32 // need nn < 64 then
    // may want to check this and throw error if nn > NN          

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_us[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_fa[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T xs_bar[3];
    __SHARED__ T loc_d[3];
    __SHARED__ T H[9];

    int xy_thread_start = threadIdx.y * blockDim.x + threadIdx.x;
    int xy_thread_stride = blockDim.x * blockDim.y;

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + xy_thread_start) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, xy_thread_start, nn, xy_thread_stride, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, xy_thread_start, nn, xy_thread_stride, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xa0[0]);
    fa.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_fa[0]);
    xs0.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    us.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_us[0]);
    __syncthreads();

    // first need to get xs0_bar, xs_bar, no forward AD yet
    memset(&xs0_bar[0], 0.0, 3 * sizeof(T));
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = xy_thread_start; i < 3 * nn; i += xy_thread_stride) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs0_bar[idim], loc_w[inode] * loc_xs0[i]);
    }
    __syncthreads();
    // return;

    // xs_bar
    for (int i = xy_thread_start; i < 3 * nn; i += xy_thread_stride) {
        int inode = i / 3;
        int idim = i % 3;
        // double a = loc_w[inode];
        // double b = loc_us[i];
        // double c = loc_xs0[i];
        // for some reason this print statement fixes race condition in kernel
        // as long as compiler expects it to maybe be here of course -1 will never be called
        // moved memset up above there..
        // if (aero_ind == 653) {  
        //     printf("xs_bar step %d: a %.4e, b %.4e, c %.4e\n", i, a, b, c);
        // }
        atomicAdd(&xs_bar[idim], loc_w[inode] * (loc_us[i] + loc_xs0[i]));
    }
    __syncthreads();

    memset(&H[0], 0.0, 9 * sizeof(T));
    for (int i = xy_thread_start; i < 9 * nn; i += xy_thread_stride) {
        int inode = i / 9;
        int i9 = i % 9;
        int idim = i9 / 3;
        int jdim = i9 % 3;

        // H += w_{inode} * p_{inode,idim} * q_{inode,jdim}
        // where p = xS - xSbar for each node and dim, sim q = xS0 - xS0bar
        // could do warp shuffling here
        T h_val = loc_w[inode] * (loc_us[3 * inode + idim] + loc_xs0[3 * inode + idim] - xs_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // diagonalization of u for stability
    H[0] += H_reg;
    H[4] += H_reg;
    H[8] += H_reg;

    // return;

    // forward AD section on uA(uS,n) for single nn node n and x,y,z direc on each thread
    // ----------------------------------------------------------------------------------

    // GPU parallelization settings, each thread is doing one nearest neighbor and one spatial dim
    int inode = threadIdx.x; // nearest neighbor node
    int idim = threadIdx.y;
    int global_struct_node = loc_conn[inode];

    // setup dot(uS,n) for forward AD on this single node, single direc
    // T2 is the forward AD type
    T2 this_us[3];
    for (int i = 0; i < 3; i++) {
        this_us[i].value = loc_us[3 * inode + i];
        this_us[i].deriv[0] = 0.0;
    }
    this_us[idim].deriv[0] = 1.0; // only one spatial dim deriv per thread


    // now dot(uS) => dot(xS_bar), only single node nonzero dot so easy computation
    // atomicAdd(&xs_bar[idim], loc_w[inode] * loc_xs[i]);
    T2 xs_bar2[3];
    for (int i = 0; i < 3; i++) {
        xs_bar2[i].value = xs_bar[i]; // copy forward analysis       
        xs_bar2[i].deriv[0] = 0.0; // don't know why but apparently I need this here
    }
    xs_bar2[idim].deriv[0] = loc_w[inode];

    // now compute dot(H) forward prop
    T2 H2[9];
    // T h_val = loc_w[inode] * (loc_xs[3 * inode + idim] - xs_bar[idim]) *
    //               (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
    for (int i = 0; i < 9; i++) {
        // copy forward analysis
        H2[i].value = H[i];
    }
    for (int jdim = 0; jdim < 3; jdim++) {
        // compute forward AD part
        T2 temp = loc_w[inode] * (loc_xs0[3 * inode + idim] + this_us[idim] - xs_bar2[idim]) * 
        (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        H2[3 * idim + jdim].deriv[0] = temp.deriv[0] + (idim == jdim) ? H_reg : 0.0;
    }

    // now forward AD types through the SVD and final uA disp calculation

    // get SVD of H (call device function in svd_utils.h)
    T2 R[9];
    // constexpr bool print = aero_ind == 279 and global_struct_node == 71 and threadIdx.y == 0;
    // const bool print = global_struct_node == 217 and threadIdx.x == 0;
    const bool print = false;
    computeRotation<T2>(H2, R, print);    

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, r);
    T2 r2[3];
    for (int i = 0; i < 3; i++) {
        r2[i] = r[i]; // passive AD type
    }

    // perform rotation and translations
    // rho = R * r + t
    T2 rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T2, 3, 3>(R, r2, rho);

    T2 loc_xa[3];
    A2D::VecSumCore<T2, 3>(xs_bar2, rho, loc_xa);

    T2 loc_ua[3];
    for (int i = 0; i < 3; i++) {
        loc_ua[i] = loc_xa[i] - loc_xa0[i];
    }

    // now compute the dot product dot(uA) * fA where dot(cdot) is forward AD state
    // and then we'll add this into the fS as our calculation of (duA/duS,n)^T fA
    // for single node, single direction
    T2 aero_dot = loc_ua[0] * loc_fa[0] + loc_ua[1] * loc_fa[1] + loc_ua[2] * loc_fa[2];
    T fS_contribution = aero_dot.deriv[0];

    int my_ind = 3 * global_struct_node + idim;

    // top corner node
    int print_dim = 2; // z force
    if (aero_ind == 899 && global_struct_node == 288 && threadIdx.y == print_dim) {
        printf("this_us:");
        printVec<T2>(3, this_us);
        
        printf("xs_bar2:");
        printVec<T2>(3, xs_bar2);

        printf("R in d%d:", threadIdx.y);
        printVec<T2>(9, R);

        printf("rho:");
        printVec<T2>(3, rho);

        printf("loc_xa:");
        printVec<T2>(3, loc_xa);

        printf("loc_ua:");
        printVec<T2>(3, loc_ua);

        printf("loc_fa:");
        printVec<T>(3, loc_fa);
    }

    atomicAdd(&fs[my_ind], fS_contribution);
    __syncthreads();

    // if (global_struct_node == 71 and aero_ind == 279 and threadIdx.y == 0) {
    //     printf("fs[%d] = %.4e\n", my_ind, fs[my_ind]);
    // }

}
#include "svd_utils.h"

template <typename T>
__GLOBAL__ void compute_aerostruct_conn(DeviceVec<int> nn, DeviceVec<T> xa0, DeviceVec<T> xs0,
                                        DeviceVec<int> &aerostruct_conn) {
    int ns_block = blockDim.x;
    int thread_starting_idxs = threadIdx.x;

    // let's use an octree on our mesh here
}

template <typename T>
__GLOBAL__ void compute_centroid_kernel(DeviceVec<T> x, DeviceVec<T> weights, DeviceVec<T> x_bar) {
    // assumes x and weights are size 3 * n
    // x_bar is size 3

    int local_ind = blockDim.x * blockIdx.x + threadIdx.x;
    int ns = x.getSize() / 3;

    const T *loc_x = &x[3 * local_ind];
    const T *loc_w = &weights[3 * local_ind];

    __SHARED__ T loc_xbar[3];
    memset(&loc_xbar[0], 0.0, 3 * sizeof(T));

    if (local_ind < ns) {
        for (int idim = 0; idim < 3; idim++) {
            loc_xbar[idim] += loc_x[idim] * loc_w[idim];
        }
    }

    // then atomicAdd back from loc_xbar into xbar
    for (int idim = 0; idim < 3; idim++) {
        atomicAdd(&xbar[idim], loc_xbar[idim]);
    }
}

template <typename T>
__GLOBAL__ void compute_weights_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ loc_conn[nn];
    __SHARED__ loc_w[nn];
    __SHARED__ loc_xs0[3 * nn];
    __SHARED__ loc_xa0[3];

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

    // compute weights (un-normalized first)
    memset(loc_w, 0.0, nn * sizeof(T));
    T sum = 0.0;
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[3 * inode + idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq);
        loc_w[inode] += new_weight;
        sum += new_weight;
    }

    // normalize the weights so it becomes a partition of unity
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        loc_w[inode] /= sum;
    }
    __syncthreads();

    int global_start = aero_ind;
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // no atomic add here, unique
        weights[global_start + inode] = loc_w[inode];
    }
}

template <typename T>
__GLOBAL__ void transfer_disps_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> xs, DeviceVec<T> xa0,
                                      DeviceVec<T> xa, DeviceVec<T> ua) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ loc_conn[nn];
    __SHARED__ loc_w[nn];
    __SHARED__ loc_xs0[3 * nn];
    __SHARED__ loc_xs[3 * nn];
    __SHARED__ loc_xa0[3];

    __SHARED__ xs0_bar[3];
    __SHARED__ xs_bar[3];
    __SHARED__ H[9];

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
        xs0_bar[idim] += loc_w[i] * loc_xs0[i];
    }

    // xs_bar
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = 0; i < 3 * nn; i++) {
        int inode = i / 3;
        int idim = i % 3;
        xs_bar[idim] += loc_w[i] * loc_xs[i];
    }
    __syncthreads();

    // compute covariance H (reduction step across threads)
    for (int i = threadIdx.x; i < 9 * nn; i += blockDim.x) {
        int inode = i / 9;
        int i9 = inode % 9;
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

    // after computing H, each thread is going to do the same thing
    // and we'll just average the result (same answer), but computations are cheap

    // get SVD of H (call device function in svd_utils.h)
    T R[9];
    computeRotation<T>(H, R);

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, r);

    // perform rotation and translations
    // rho = R * r + t
    T rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T, 3, 3>(R, r, rho);

    T loc_xa[3];
    A2D::VecAddCore<T, 3>(xs0, rho, loc_xa);

    T loc_ua[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa, -1.0, loc_xa0, loc_ua);

    // update xa and u0 globally with add reduction by the blockDim.x
    int nb = blockDim.x;
    // should probably do warp shuffle here among the threads to speed it up
    // before atomic add
    for (int i = 0; i < 3; i++) {
        atomicAdd(&xa[3 * aero_ind], 1.0 / nb * loc_xa[i]);
        atomicAdd(&ua[3 * aero_ind], 1.0 / nb * loc_ua[i]);
    }
}

template <typename T>
__GLOBAL__ void transfer_loads_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> xs, DeviceVec<T> xa0,
                                      DeviceVec<T> xa, DeviceVec<T> fa, DeviceVec<T> fs) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ loc_conn[nn];
    __SHARED__ loc_w[nn];
    __SHARED__ loc_xs0[3 * nn];
    __SHARED__ loc_xs[3 * nn];
    __SHARED__ loc_xa0[3];
    __SHARED__ loc_fa[3];

    __SHARED__ xs0_bar[3];
    __SHARED__ xs_bar[3];
    __SHARED__ H[9];
    __SHARED__ M[225], adjoint[15], rhs[15];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    fa.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_fa[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs[0]);
    __syncthreads();

    // TODO : this is the same code as with transfer_disps_kernel ? can we make a method that calls
    // some of this stuff? to avoid repeated code?

    // all of the following will be reduction-like operations
    // until H is computed (each thread helps in the work)

    // compute the centroids of the nearest neighbors
    // xs0_bar
    memset(&xs0_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * nn; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        xs0_bar[idim] += loc_w[i] * loc_xs0[i];
    }

    // xs_bar
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = 0; i < 3 * nn; i++) {
        int inode = i / 3;
        int idim = i % 3;
        xs_bar[idim] += loc_w[i] * loc_xs[i];
    }
    __syncthreads();

    // compute covariance H (reduction step across threads)
    for (int i = threadIdx.x; i < 9 * nn; i += blockDim.x) {
        int inode = i / 9;
        int i9 = inode % 9;
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

    // after computing H, each thread is going to do the same thing
    // and we'll just average the result (same answer), but computations are cheap

    // get SVD of H (call device function in svd_utils.h)
    T R[9], S[9];
    computeRotation<T>(H, R, S);

    // note in the CPU version of MELD we store R, S for each aero node
    // I could do that, but I've chosen to just re-compute it for now (maybe this is incorrect)
    // now we have H, R, S so we can begin assembling and solving the 15x15 linear system
    svd_15x15_adjoint(M, adjoint, rhs, loc_fa, d, R, S);

    // copy adjoint solutions out for each thread
    T X[9], Y[6];
    for (int i = 0; i < 9; i++) {
        X[i] = adjoint[i];
    }
    for (int j = 0; j < 6; j++) {
        Y[j] = adjoint[9 + j];
    }

    // now compute struct loads
    // fs_{n} += w_n * fA + wn * X^T * qn
    // and add into global struct loads directly
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        T fs_tmp[3];
        // w_n * fA part
        for (int idim = 0; idim < 3; idim++) {
            fs_tmp[idim] = loc_w[inode] * loc_fa[idim];
        }

        // compute qn = xS0,n - xs0_bar
        T qn[3];
        A2D::VecSumCore<T, 3>(1.0, loc_xs[3 * inode], -1.0, xs0_bar, qn);

        // += wn * X^T * qn part
        bool additive = true;
        A2D::MatVecCoreScale<T, 3, 3, A2D::MatOp::TRANSPOSE, additive>(loc_w[inode], X, qn, fs_tmp);

        // could do warp shuffle here?
        int global_node = aero_struct_conn[inode];
        for (int idim = 0; idim < 3; idim++) {
            atomicAdd(&fs[3 * global_node + idim], fs_tmp[idim]);
        }
    }
}
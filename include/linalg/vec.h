#pragma once
#include "../base/utils.h"
#include "../cuda_utils.h"
#include "chrono"
#include "stdlib.h"
#include <complex>
#include <cstring>

#ifdef USE_GPU
#include "../cuda_utils.h"
#include "vec.cuh"
#endif

// forward referencing
template <typename T> class DeviceVec;
template <typename T> class HostVec;

template <typename T> class BaseVec {
  public:
    using type = T;

    BaseVec() = default; // default constructor
    __HOST_DEVICE__ BaseVec(int N, T *data)
        : N(N), data(data), perm(nullptr), iperm(nullptr), block_dim(1) {}
    __HOST_DEVICE__ BaseVec(int N, T *data, int block_dim)
        : N(N), data(data), perm(nullptr), iperm(nullptr),
          block_dim(block_dim) {}
    __HOST_DEVICE__ BaseVec(int N, T *myData, int block_dim, int *newPerm,
                            int *newIPerm) N(N),
        data(nullptr), block_dim(block_dim), perm(newPerm), iperm(newIPerm) {
        // permute data to new permutation
        permuteData(myData, this->perm);
        this->data = myData;
    }
    __HOST_DEVICE__ BaseVec(const BaseVec &vec) {
        // copy constructor
        this->N = vec.N;
        this->data = vec.data;
        this->block_dim = vec.block_dim;
        this->perm = vec.perm;
        this->iperm = vec.iperm;
    }
    __HOST_DEVICE__ void getData(T *myData) { myData = data; }
    __HOST_DEVICE__ T *getPtr() { return data; }
    __HOST_DEVICE__ const T *getPtr() const { return data; }
    __HOST_DEVICE__ int getSize() const { return N; }
    __HOST_DEVICE__ int *getPerm() { return perm; }
    __HOST_DEVICE__ void setPerm(int *newPerm) { perm = newPerm; }
    __HOST_DEVICE__ int *getIperm() { return iperm; }
    __HOST_DEVICE__ void setIPerm(int *newIPerm) { iperm = newIPerm; }
    __HOST_DEVICE__ int getBlockDim() { return block_dim; }
    __HOST_DEVICE__ int getNumNodes() { return N / block_dim; }

    __HOST_DEVICE__ static int permuteDof(int _idof, int *myPerm,
                                          int myBlockDim) {
        int _inode = _idof / myBlockDim;
        int inner_dof = _idof % myBlockDim;
        int inode = perm[_inode];
        int idof = inode * myBlockDim + inner_dof;
        return idof;
    }

    __HOST__ static void permuteData(T *myData, int *permuteVec) {
        // same baseline permute routine used for the forward and inverse
        // permute
        T *temp = new T[N];
        int nnodes = getNumNodes();
        // store permutation of myData in temp
        for (int inode = 0; inode < nnodes; inode++) {
            int new_inode = permuteVec[inode];
            for (int inner = 0; inner < block_dim; inner++) {
                temp[new_inode * block_dim + inner] =
                    myData[inode * blockDim + inner];
            }
        }
        // copy data back from temp to myData
        for (int inode = 0; inode < nnodes; inode++) {
            for (int inner = 0; inner < block_dim; inner++) {
                myData[inode * block_dim + inner] =
                    temp[inode * blockDim + inner];
            }
        }
        delete[] temp;
    }

    // __HOST__ void zeroValues() {
    //     memset(this->data, 0.0, this->N * sizeof(T));
    // }

    __HOST__ void scale(T my_scale) {
        for (int i = 0; i < N; i++) {
            this->data[i] *= my_scale;
        }
    }

    __HOST_DEVICE__ void getElementValues(const int dof_per_node,
                                          const int nodes_per_elem,
                                          const int32_t *elem_conn,
                                          T *elem_data) const {
        for (int inode = 0; inode < nodes_per_elem; inode++) {
            int32_t _global_inode = elem_conn[inode];
            int32_t global_inode = perm[_global_inode];
            for (int idof = 0; idof < dof_per_node; idof++) {
                int local_ind = inode * dof_per_node + idof;
                int global_ind = global_inode * dof_per_node + idof;
                elem_data[inode * dof_per_node + idof] =
                    data[global_inode * dof_per_node + idof];
            }
        }
    }

    __HOST_DEVICE__ void apply_bcs(const BaseVec<int> bcs) {
        int nbcs = bcs.getSize();
        for (int ibc = 0; ibc < nbcs; ibc++) {
            int _idof = bcs[ibc];
            int idof = this->permuteDof(_idof, this->perm, this->block_dim);
            data[idof] = 0.0; // if non-dirichlet bcs handle later.. in TODO
        }
    }

    __HOST_DEVICE__
    void addElementValues(const T scale, const int dof_per_node,
                          const int nodes_per_elem, const int32_t *elem_conn,
                          const T *elem_data) {
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
            int elem_inode = idof / dof_per_node;
            int _inode = elem_conn[elem_inode];
            int inode = perm[_inode];
            int iglobal = inode * dof_per_node + (idof % dof_per_node);

            // printf("add value %.8e into location %d with idof %d\n",
            //        scale * elem_data[idof], iglobal, idof);
            data[iglobal] += scale * elem_data[idof];
        }
    }

    template <typename I> __HOST_DEVICE__ T &operator[](const I i) {
        return data[i];
    }
    template <typename I> __HOST_DEVICE__ const T &operator[](const I i) const {
        return data[i];
    }

  protected:
    int N, block_dim;
    T *data;
    int *perm, *iperm;
};

template <typename T> class DeviceVec : public BaseVec<T> {
  public:
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
#endif // USE_GPU

    DeviceVec() = default; // default constructor
    __HOST__ DeviceVec(int N, int block_dim, int *perm, int *iperm,
                       bool memset = true)
        : BaseVec<T>(N, nullptr, block_dim, perm, iperm) {
#ifdef USE_GPU
        cudaMalloc((void **)&this->data, N * sizeof(T));
        if (memset) {
            cudaMemset(this->data, 0.0, N * sizeof(T));
        }
#endif
    }
    __HOST__ DeviceVec(int N, T *data) : BaseVec<T>(N, data) {}
    __HOST__ void zeroValues() {
        cudaMemset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST__ void apply_bcs(DeviceVec<int> bcs) {
#ifdef USE_GPU
        dim3 block = bcs_block;
        int nbcs = bcs.getSize();
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);
        // printf("in deviceVec apply_bcs\n");

        apply_vec_bcs_kernel<T, DeviceVec>
            <<<grid, block>>>(bcs, this->data, block_dim, perm);

        CHECK_CUDA(cudaDeviceSynchronize());
#else // NOT USE_GPU
        BaseVec<T>::apply_bcs(bcs);
#endif
    }
    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST__ HostVec<T> createHostVec() {
        HostVec<T> vec(this->N, block_dim);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vec.getPerm(), this->perm, this->N * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vec.getIPerm(), this->iperm, this->N * sizeof(int),
                   cudaMemcpyDeviceToHost);
#endif
        return vec;
    }

    __HOST__ HostVec<T> createOutputVec() {
        // output vec for printing, displaying to visualization, etc.
        // inverts the permutation so we can view the data the original nodal
        // ordering of the mesh

        HostVec<T> h_vec = this->createHostVec();
        return h_vec.createOutputVec();
    }

    __HOST__ DeviceVec<T> copyVec() {
        // copy the device vec since Kmat gets modified during LU solve
        DeviceVec<T> vec(this->N, block_dim);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(vec.getPerm(), this->perm, this->N * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(vec.getIPerm(), this->iperm, this->N * sizeof(int),
                   cudaMemcpyDeviceToDevice);
#endif
        return vec;
    }

    __HOST__ DeviceVec<T> createDeviceVec() { return *this; }

#ifdef USE_GPU
    __DEVICE__ void copyValuesToShared(const bool active_thread,
                                       const int start, const int stride,
                                       const int dof_per_node,
                                       const int nodes_per_elem,
                                       const int32_t *elem_conn,
                                       T *shared_data) const {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread) {
            return;
        }

        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride) {
            int elem_inode = idof / dof_per_node;
            int _inode = elem_conn[elem_inode];
            int inode =
                BaseVec<T>::permuteDof(_inode, this->perm, this->block_dim);
            int iglobal = inode * dof_per_node + (idof % dof_per_node);
            shared_data[idof] = this->data[iglobal];
        }
        // make sure you call __syncthreads() at some point after this
    }

    __DEVICE__ void copyValuesToShared_BCs(const bool active_thread,
                                           const int start, const int stride,
                                           const int dof_per_node,
                                           const int nodes_per_elem,
                                           const int32_t *elem_conn,
                                           T *shared_data) const {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread) {
            return;
        }

        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride) {
            int elem_inode = idof / dof_per_node;
            int _inode = elem_conn[elem_inode];
            int inode =
                BaseVec<T>::permuteDof(_inode, this->perm, this->block_dim);
            int iglobal = inode * dof_per_node + (idof % dof_per_node);
            shared_data[idof] = this->data[iglobal];
        }
        // make sure you call __syncthreads() at some point after this
    }

    __DEVICE__ static void copyLocalToShared(const bool active_thread,
                                             const T scale, const int N,
                                             const T *local, T *shared) {
        for (int i = 0; i < N; i++) {
            atomicAdd(&shared[i], scale * local[i]);
        }
        __syncthreads();
    }

    __DEVICE__ void
    addElementValuesFromShared(const bool active_thread, const int start,
                               const int stride, const int dof_per_node,
                               const int nodes_per_elem,
                               const int32_t *elem_conn, const T *shared_data) {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
            return;
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride) {
            int elem_inode = idof / dof_per_node;
            int _inode = elem_conn[elem_inode];
            int inode =
                BaseVec<T>::permuteDof(_inode, this->perm, this->block_dim);
            int iglobal = inode * dof_per_node + (idof % dof_per_node);

            atomicAdd(&this->data[iglobal], shared_data[idof]);
        }
    }
#endif // USE_GPU
};

template <typename T> class HostVec : public BaseVec<T> {
  public:
    HostVec() = default; // default constructor
    __HOST_DEVICE__ HostVec(int N, int block_dim)
        : BaseVec<T>(N, nullptr, int block_dim) {
        this->data = new T[N];
        memset(this->data, 0.0, N * sizeof(T));
    }
    __HOST_DEVICE__ HostVec(int N, T *data, int block_dim, int *perm,
                            int *iperm)
        : BaseVec<T>(N, data, block_dim, perm, iperm) {}

    __HOST_DEVICE__ HostVec(int N, T one_data) : HostVec(N, nullptr) {
        // initialize each entry with same value
        for (int i = 0; i < N; i++) {
            this->data[i] = one_data;
        }
    }

    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST_DEVICE__ void zeroValues() {
        memset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST__ void randomize(T maxVal = 1) {
        // create random initial values (for debugging)
        if constexpr (std::is_same<T, double>::value) {
            for (int i = 0; i < this->N; i++) {
                this->data[i] = maxVal * static_cast<double>(rand()) / RAND_MAX;
            }
        } else if constexpr (std::is_same<T, std::complex<double>>::value) {
            for (int i = 0; i < this->N; i++) {
                double my_double =
                    maxVal.real() * static_cast<double>(rand()) / RAND_MAX;
                this->data[i] = T(my_double, 0.0);
            }
        } else { // int's
            for (int i = 0; i < this->N; i++) {
                this->data[i] = rand() % maxVal;
            }
        }
    }

    __HOST__ HostVec<T> copyVec() {
        HostVec<T> vec(this->N, nullptr, this->block_dim, nullptr, nullptr);
        memcpy(vec.getPtr(), this->data, this->N * sizeof(T));
        memcpy(vec.getPerm(), this->perm, this->N * sizeof(int));
        memcpy(vec.getIPerm(), this->iperm, this->N * sizeof(int));
        return vec;
    }

    __HOST__ HostVec<T> createOutputVec() {
        // output vec for printing, displaying to visualization, etc.
        // inverts the permutation so we can view the data the original nodal
        // ordering of the mesh

        this->permuteData(this->data, this->iperm);
        return this->copyVec();
    }

    __HOST__ DeviceVec<T> createDeviceVec(bool memset = true,
                                          bool can_print = false) {
        // creates a device vector and copies this host data to the device
        DeviceVec<T> vec =
            DeviceVec<T>(this->N, memset, this->block_dim, nullptr, nullptr);
        auto start = std::chrono::high_resolution_clock::now();
#ifdef USE_GPU
        if (can_print) {
            printf("copy host to device vec %d entries\n", this->N);
        }
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(vec.getPerm(), this->perm, this->N * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(vec.getIPerm(), this->iperm, this->N * sizeof(int),
                   cudaMemcpyHostToDevice);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        if (can_print) {
            printf("\tcopy host to device vec in %d microseconds\n",
                   (int)duration.count());
        }
#endif
        return vec;
    }

    __HOST__ HostVec<T> createHostVec() { return *this; }
};

/*
convertVec : converts vec to host or device vec depending on whether
CUDA compiles are being used (maybe could have better name)
*/
#ifdef USE_GPU
template <typename T> DeviceVec<T> convertVecType(HostVec<T> &vec) {
    return vec.createDeviceVec();
}
#else
template <typename T> HostVec<T> convertVecType(HostVec<T> &vec) { return vec; }
#endif

#ifdef USE_GPU
template <typename T> using VecType = DeviceVec<T>;
#else
template <typename T> using VecType = HostVec<T>;
#endif

template <typename T>
T getVecRelError(HostVec<T> vec1, HostVec<T> vec2,
                 const double threshold = 1e-10, const bool print = true) {
    T rel_err = 0.0;
    for (int i = 0; i < vec1.getSize(); i++) {
        T loc_rel_err = abs((vec1[i] - vec2[i]) / (vec2[i] + 1e-14));
        if (loc_rel_err > rel_err &&
            abs(vec2[i]) >
                threshold) { // don't use rel error if vec2 is
                             // near floating point zero (then it's meaningless)
            rel_err = loc_rel_err;
        }
    }

    if (print) {
        printf("rel_err = %.8e\n", rel_err);
    }
    return rel_err;
}
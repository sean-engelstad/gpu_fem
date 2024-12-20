#pragma once
#include "../cuda_utils.h"
#include "stdlib.h"

#ifdef USE_GPU
#include "vec.cuh"
#endif

// forward referencing
template <typename T> class DeviceVec;
template <typename T> class HostVec;

template <typename T> class BaseVec {
  public:
    using type = T;

    BaseVec() = default; // default constructor
    __HOST_DEVICE__ BaseVec(int N, T *data) : N(N), data(data) {}
    __HOST_DEVICE__ BaseVec(const BaseVec &vec) {
        // copy constructor
        this->N = vec.N;
        this->data = vec.data;
    }
    __HOST_DEVICE__ void getData(T *myData) { myData = data; }
    __HOST_DEVICE__ T *getPtr() { return data; }
    __HOST_DEVICE__ const T *getPtr() const { return data; }
    __HOST_DEVICE__ int getSize() const { return N; }
    // __HOST__ virtual DeviceVec<T> createDeviceVec() const = 0;
    // __HOST__ virtual HostVec<T> createHostVec() const = 0;

    __HOST_DEVICE__ void getElementValues(const int dof_per_node,
                                          const int nodes_per_elem,
                                          const int32_t *elem_conn,
                                          T *elem_data) const {
        for (int inode = 0; inode < nodes_per_elem; inode++) {
            int32_t global_inode = elem_conn[inode];
            for (int idof = 0; idof < dof_per_node; idof++) {
                elem_data[inode * dof_per_node + idof] =
                    data[global_inode * dof_per_node + idof];
            }
        }
    }

    __HOST_DEVICE__ void apply_bcs(const BaseVec<int> bcs) {
        int nbcs = bcs.getSize();
        for (int ibc = 0; ibc < nbcs; ibc++) {
            int idof = bcs[ibc];
            data[idof] = 0.0; // if non-dirichlet bcs handle later.. in TODO
        }
    }

    __HOST_DEVICE__
    void addElementValues(const T scale, const int dof_per_node,
                          const int nodes_per_elem, const int32_t *elem_conn,
                          const T *elem_data) {
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);
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
    int N;
    T *data;
};

template <typename T> class DeviceVec : public BaseVec<T> {
  public:
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
#endif // USE_GPU

    DeviceVec() = default; // default constructor
    __HOST__ DeviceVec(int N, bool memset = true) : BaseVec<T>(N, nullptr) {
#ifdef USE_GPU
        cudaMalloc((void **)&this->data, N * sizeof(T));
        if (memset) {
            cudaMemset(this->data, 0.0, N * sizeof(T));
        }
#endif
    }
    __HOST_DEVICE__ void zeroValues() {
        cudaMemset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST__ void apply_bcs(DeviceVec<int> bcs) {
#ifdef USE_GPU
        dim3 block = bcs_block;
        int nbcs = bcs.getSize();
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);
        printf("in deviceVec apply_bcs\n");

        apply_bcs_kernel<T, DeviceVec><<<grid, block>>>(bcs, this->data);
#else // NOT USE_GPU
        BaseVec<T>::apply_bcs(bcs);
#endif
    }
    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST__ HostVec<T> createHostVec() {
        HostVec<T> vec(this->N);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToHost);
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
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);
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
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);

            atomicAdd(&this->data[iglobal], shared_data[idof]);
        }
    }
#endif // USE_GPU
};

template <typename T> class HostVec : public BaseVec<T> {
  public:
    HostVec() = default; // default constructor
    __HOST_DEVICE__ HostVec(int N) : BaseVec<T>(N, nullptr) {
        this->data = new T[N];
        memset(this->data, 0.0, N * sizeof(T));
    }
    __HOST_DEVICE__ HostVec(int N, T *data) : BaseVec<T>(N, data) {}
    __HOST_DEVICE__ HostVec(int N, T one_data) : HostVec(N) {
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
        } else { // int's
            for (int i = 0; i < this->N; i++) {
                this->data[i] = rand() % maxVal;
            }
        }
    }

    __HOST__ DeviceVec<T> createDeviceVec(bool memset = true) {
        // creates a device vector and copies this host data to the device
        DeviceVec<T> vec(this->N, memset);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyHostToDevice);
#endif
        return vec;
    }

    __HOST__ HostVec<T> createHostVec() { return *this; }
};

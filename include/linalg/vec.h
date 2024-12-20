#pragma once
#include "../cuda_utils.h"
#include "stdlib.h"

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
    __HOST_DEVICE__ int getSize() { return N; }
    __HOST__ virtual DeviceVec<T> createDeviceVec() const = 0;
    __HOST__ virtual HostVec<T> createHostVec() const = 0;

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
    DeviceVec() = default; // default constructor
    __HOST__ DeviceVec(int N) : BaseVec<T>(N, nullptr) {
#ifdef USE_GPU
        cudaMalloc((void **)&this->data, N * sizeof(T));
        cudaMemset(this->data, 0.0, N * sizeof(T));
#endif
    }
    __HOST_DEVICE__ void zeroValues() {
        cudaMemset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST__ virtual HostVec<T> createHostVec() const override {
        HostVec<T> vec(this->N);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToHost);
#endif
        return vec;
    }
    __HOST__ virtual DeviceVec<T> createDeviceVec() const override {
        return *this;
    }

#ifdef USE_GPU
    __DEVICE__ void copyValuesToShared(const bool active_thread,
                                       const int start, const int stride,
                                       const int dof_per_node,
                                       const int nodes_per_elem,
                                       const int32_t *elem_conn,
                                       T *shared_data) const {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
            return;

        // shared_data[0] = 0.0;
        // printf("shared_data[0] = %.8e\n", shared_data[0]);

        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);
            iglobal = 0; // debug check
            // shared_data[idof] = this->data[iglobal];
            shared_data[idof] = 0.0;
            printf("shared[%d] = %.8e\n", idof, shared_data[idof]);
            // printf("shared[%d] = %.8e, global[%d] = %.8e\n", idof,
            //        shared_data[idof], iglobal, this->data[iglobal]);
        }
        // __syncthreads();
    }

    __DEVICE__ static void copyLocalToShared(const bool active_thread,
                                             const int N, const T *local,
                                             T *shared) {
        for (int i = 0; i < N; i++) {
            atomicAdd(&shared[i], local[i]);
        }
    }

    __DEVICE__ void addValuesFromShared(const bool active_thread,
                                        const int start, const int stride,
                                        const int dof_per_node,
                                        const int nodes_per_elem,
                                        const int32_t *elem_conn,
                                        const T *shared_data) {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
            return;
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
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

    __HOST__ virtual DeviceVec<T> createDeviceVec() const override {
        // creates a device vector and copies this host data to the device
        DeviceVec<T> vec(this->N);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyHostToDevice);
#endif
        return vec;
    }

    __HOST__ virtual HostVec<T> createHostVec() const override { return *this; }
};

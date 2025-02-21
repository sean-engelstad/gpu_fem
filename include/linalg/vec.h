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
template <typename T>
class DeviceVec;
template <typename T>
class HostVec;

template <typename T>
class BaseVec
{
public:
    using type = T;

    BaseVec() = default; // default constructor
    __HOST_DEVICE__ BaseVec(int N, T *data) : N(N), data(data) {}
    __HOST_DEVICE__ BaseVec(const BaseVec &vec)
    {
        // copy constructor
        this->N = vec.N;
        this->data = vec.data;
    }
    __HOST_DEVICE__ void getData(T *myData) { myData = data; }
    __HOST_DEVICE__ T *getPtr() { return data; }
    __HOST_DEVICE__ const T *getPtr() const { return data; }
    __HOST_DEVICE__ int getSize() const { return N; }
    // __HOST__ void zeroValues() {
    //     memset(this->data, 0.0, this->N * sizeof(T));
    // }

    __HOST__ void scale(T my_scale)
    {
        for (int i = 0; i < N; i++)
        {
            this->data[i] *= my_scale;
        }
    }

    __HOST_DEVICE__ void getElementValues(const int dof_per_node,
                                          const int nodes_per_elem,
                                          const int32_t *elem_conn,
                                          T *elem_data) const
    {
        for (int inode = 0; inode < nodes_per_elem; inode++)
        {
            int32_t global_inode = elem_conn[inode];
            for (int idof = 0; idof < dof_per_node; idof++)
            {
                int local_ind = inode * dof_per_node + idof;
                int global_ind = global_inode * dof_per_node + idof;
                elem_data[inode * dof_per_node + idof] =
                    data[global_inode * dof_per_node + idof];
            }
        }
    }

    __HOST_DEVICE__ void apply_bcs(const BaseVec<int> bcs)
    {
        int nbcs = bcs.getSize();
        for (int ibc = 0; ibc < nbcs; ibc++)
        {
            int idof = bcs[ibc];
            data[idof] = 0.0; // if non-dirichlet bcs handle later.. in TODO
        }
    }

    __HOST_DEVICE__ static int permuteDof(int _idof, int *myPerm,
                                          int myBlockDim)
    {
        int _inode = _idof / myBlockDim;
        int inner_dof = _idof % myBlockDim;
        int inode = myPerm[_inode];
        int idof = inode * myBlockDim + inner_dof;
        return idof;
    }

    __HOST__ void permuteData(int block_dim, int *perm)
    {
        // apply this permutation to our data
        T *temp;
#ifndef USE_GPU
        temp = new T[N];
        int nnodes = N / block_dim;
        // store permutation of myData in temp
        for (int inode = 0; inode < nnodes; inode++)
        {
            int new_inode = perm[inode];
            for (int inner = 0; inner < block_dim; inner++)
            {
                temp[new_inode * block_dim + inner] =
                    data[inode * block_dim + inner];
            }
        }
        // copy data back from temp to myData
        memcpy(this->data, temp, this->N * sizeof(T));
        delete[] temp;
#endif
#ifdef USE_GPU
        cudaMalloc((void **)&temp, N * sizeof(T));
        // launch kernel to permute vec
        // create temp still? or just use shared memory to do it?
        dim3 block(128, 1, 1);
        int num_nodes = N / block_dim;
        int nblocks = (num_nodes + block.x - 1) / block.x;
        dim3 grid(nblocks);

        permute_vec_kernel<T, DeviceVec>
            <<<grid, block>>>(num_nodes, this->data, temp, block_dim, perm);

        CHECK_CUDA(cudaDeviceSynchronize());

        // then use cudaMemcpy back from temp to data
        cudaMemcpy(this->data, temp, this->N * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        cudaFree(temp);
#endif
    }

    __HOST_DEVICE__
    void addElementValues(const T scale, const int dof_per_node,
                          const int nodes_per_elem, const int32_t *elem_conn,
                          const T *elem_data)
    {
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++)
        {
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);

            // printf("add value %.8e into location %d with idof %d\n",
            //        scale * elem_data[idof], iglobal, idof);
            data[iglobal] += scale * elem_data[idof];
        }
    }

    __HOST__ void setData(T *myData, int *perm, int block_dim)
    {
        data = myData;
        // deep copy and permute data at same time
        permuteData(block_dim, perm);
    }

    template <typename I>
    __HOST_DEVICE__ T &operator[](const I i)
    {
        return data[i];
    }
    template <typename I>
    __HOST_DEVICE__ const T &operator[](const I i) const
    {
        return data[i];
    }

protected:
    int N;
    T *data;
};

template <typename T>
class DeviceVec : public BaseVec<T>
{
public:
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
#endif // USE_GPU

    DeviceVec() = default; // default constructor
    __HOST__ DeviceVec(int N, bool memset = true) : BaseVec<T>(N, nullptr)
    {
#ifdef USE_GPU
        cudaMalloc((void **)&this->data, N * sizeof(T));
        if (memset)
        {
            cudaMemset(this->data, 0.0, N * sizeof(T));
        }
#endif
    }
    __HOST__ DeviceVec(int N, T *data) : BaseVec<T>(N, data) {}
    __HOST__ void zeroValues()
    {
        cudaMemset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST__ void apply_bcs(DeviceVec<int> bcs)
    {
#ifdef USE_GPU
        dim3 block = bcs_block;
        int nbcs = bcs.getSize();
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);
        // printf("in deviceVec apply_bcs\n");

        apply_vec_bcs_kernel<T, DeviceVec><<<grid, block>>>(bcs, this->data);

        CHECK_CUDA(cudaDeviceSynchronize());
#else // NOT USE_GPU
        BaseVec<T>::apply_bcs(bcs);
#endif
    }

    __HOST__ void ~DeviceVec()
    {
        if (this->data)
        {
            cudaFree(this->data);
        }
    }

    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST__ HostVec<T> createHostVec()
    {
        HostVec<T> vec(this->N);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToHost);
#endif
        return vec;
    }

    __HOST__ DeviceVec<T> copyVec()
    {
        // copy the device vec since Kmat gets modified during LU solve
        DeviceVec<T> vec(this->N);
#ifdef USE_GPU
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToDevice);
#endif
        return vec;
    }

    __HOST__ DeviceVec<T> createPermuteVec(int block_dim, int *perm)
    {
        auto new_vec = copyVec();
        new_vec.permuteData(block_dim, perm);
        return new_vec;
    }

    __HOST__ void copyValuesTo(DeviceVec<T> dest)
    {
        cudaMemcpy(dest.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    }

    __HOST__ DeviceVec<T> createDeviceVec() { return *this; }

#ifdef USE_GPU
    __DEVICE__ void copyElemValuesToShared(const bool active_thread,
                                           const int start, const int stride,
                                           const int dof_per_node,
                                           const int nodes_per_elem,
                                           const int32_t *elem_conn,
                                           T *shared_data) const
    {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
        {
            return;
        }

        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride)
        {
            int local_inode = idof / dof_per_node;
            int global_inode = elem_conn[local_inode]; // no perm here since
                                                       // xpts isn't permuted
            int iglobal = global_inode * dof_per_node + (idof % dof_per_node);
            shared_data[idof] = this->data[iglobal];
        }
        // make sure you call __syncthreads() at some point after this
    }

    __DEVICE__ void copyValuesToShared(const bool active_thread,
                                       int start, int end, int stride,
                                       int global_start, T *shared_data)
        const
    {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
        {
            return;
        }

        for (int i = start; i < end; i += stride)
        {
            shared_data[i] = this->data[global_start + i];
        }
        // make sure you call __syncthreads() at some point after this
    }

    __DEVICE__ void copyValuesToShared_BCs(const bool active_thread,
                                           const int start, const int stride,
                                           const int dof_per_node,
                                           const int nodes_per_elem,
                                           const int32_t *elem_conn,
                                           T *shared_data) const
    {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
        {
            return;
        }

        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride)
        {
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);
            shared_data[idof] = this->data[iglobal];
        }
        // make sure you call __syncthreads() at some point after this
    }

    __DEVICE__ static void copyLocalToShared(const bool active_thread,
                                             const T scale, const int N,
                                             const T *local, T *shared)
    {
        for (int i = 0; i < N; i++)
        {
            atomicAdd(&shared[i], scale * local[i]);
        }
        __syncthreads();
    }

    __DEVICE__ void
    addElementValuesFromShared(const bool active_thread, const int start,
                               const int stride, const int dof_per_node,
                               const int nodes_per_elem, const int32_t *perm,
                               const int32_t *elem_conn, const T *shared_data)
    {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread)
            return;
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride)
        {
            int local_inode = idof / dof_per_node;
            int _global_inode = elem_conn[local_inode];
            int global_inode = perm[_global_inode];
            int iglobal = global_inode * dof_per_node + (idof % dof_per_node);

            atomicAdd(&this->data[iglobal], shared_data[idof]);
        }
    }
#endif // USE_GPU
};

template <typename T>
class HostVec : public BaseVec<T>
{
public:
    HostVec() = default; // default constructor
    __HOST_DEVICE__ HostVec(int N) : BaseVec<T>(N, nullptr)
    {
        this->data = new T[N];
        memset(this->data, 0.0, N * sizeof(T));
    }
    __HOST_DEVICE__ HostVec(int N, T *data) : BaseVec<T>(N, data) {}
    __HOST_DEVICE__ HostVec(int N, T one_data) : HostVec(N)
    {
        // initialize each entry with same value
        for (int i = 0; i < N; i++)
        {
            this->data[i] = one_data;
        }
    }

    __HOST_DEVICE__ void getData(T *&myData) { myData = this->data; }
    __HOST_DEVICE__ void zeroValues()
    {
        memset(this->data, 0.0, this->N * sizeof(T));
    }
    __HOST__ void randomize(T maxVal = 1)
    {
        // create random initial values (for debugging)
        if constexpr (std::is_same<T, double>::value)
        {
            for (int i = 0; i < this->N; i++)
            {
                this->data[i] = maxVal * static_cast<double>(rand()) / RAND_MAX;
            }
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            for (int i = 0; i < this->N; i++)
            {
                double my_double =
                    maxVal.real() * static_cast<double>(rand()) / RAND_MAX;
                this->data[i] = T(my_double, 0.0);
            }
        }
        else
        { // int's
            for (int i = 0; i < this->N; i++)
            {
                this->data[i] = rand() % maxVal;
            }
        }
    }

    __HOST__ HostVec<T> copyVec()
    {
        HostVec<T> vec(this->N);
        memcpy(vec.getPtr(), this->data, this->N * sizeof(T));
        return vec;
    }

    __HOST__ HostVec<T> createPermuteVec(int block_dim, int *perm)
    {
        auto new_vec = copyVec();
        new_vec.permuteData(block_dim, perm);
        return new_vec;
    }

    __HOST_DEVICE__ void ~HostVec()
    {
        if (this->data)
        {
            delete[] this->data;
        }
    }

    __HOST__ DeviceVec<T> createDeviceVec(bool memset = true,
                                          bool can_print = false)
    {
        // creates a device vector and copies this host data to the device
        DeviceVec<T> vec = DeviceVec<T>(this->N, memset);
        auto start = std::chrono::high_resolution_clock::now();
#ifdef USE_GPU
        if (can_print)
        {
            printf("copy host to device vec %d entries\n", this->N);
        }
        cudaMemcpy(vec.getPtr(), this->data, this->N * sizeof(T),
                   cudaMemcpyHostToDevice);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        if (can_print)
        {
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
template <typename T>
DeviceVec<T> convertVecType(HostVec<T> &vec)
{
    return vec.createDeviceVec();
}
#else
template <typename T>
HostVec<T> convertVecType(HostVec<T> &vec) { return vec; }
#endif

#ifdef USE_GPU
template <typename T>
using VecType = DeviceVec<T>;
#else
template <typename T>
using VecType = HostVec<T>;
#endif

template <typename T>
T getVecRelError(HostVec<T> vec1, HostVec<T> vec2,
                 const double threshold = 1e-10, const bool print = true)
{
    T rel_err = 0.0;
    for (int i = 0; i < vec1.getSize(); i++)
    {
        T loc_rel_err = abs((vec1[i] - vec2[i]) / (vec2[i] + 1e-14));
        if (loc_rel_err > rel_err &&
            abs(vec2[i]) >
                threshold)
        { // don't use rel error if vec2 is
          // near floating point zero (then it's meaningless)
            rel_err = loc_rel_err;
        }
    }

    if (print)
    {
        printf("rel_err = %.8e\n", rel_err);
    }
    return rel_err;
}
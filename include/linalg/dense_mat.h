// dense matrix for debugging
#pragma once
#include "../cuda_utils.h"

// square N x N dense matrix
template <class Vec> class DenseMat {
  public:
    using T = Vec::type;

    __HOST_DEVICE__ DenseMat(int N) : N(N) { data = Vec(N); }

    __HOST_DEVICE__ void getSize() { return N; }
    __HOST_DEVICE__ void zeroValues() { data.zeroValues(); }
    __HOST_DEVICE__ Vec getVec() { return data; }
    __HOST_DEVICE__ T *getPtr() { return data.getPtr(); }

    __HOST_DEVICE__ addElementMatrixValues(const T scale, const int ielem,
                                           const int dof_per_node,
                                           const int nodes_per_elem,
                                           const int32_t *elem_conn,
                                           const T *elem_mat) {
        // similar method to vec.h or Vec.addElementValues but here for matrix
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
            int local_inode = idof / dof_per_node;
            int iglobal =
                elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);

            for (int jdof = 0; jdof < dof_per_elem; jdof++) {
                int local_jnode = jdof / dof_per_node;
                int iglobal = elem_conn[local_jnode] * dof_per_node +
                              (jdof % dof_per_node);
                data[N * iglobal + jglobal] +=
                    scale * elem_mat[dof_per_elem * idof + jdof];
            }
        }
    }

    template <typename I> __HOST_DEVICE__ T &operator[](const I i) {
        return data[i];
    }
    template <typename I> __HOST_DEVICE__ const T &operator[](const I i) const {
        return data[i];
    }

  private:
    int N;
    Vec data;
};

template <class Assembler, class Vec>
__HOST_DEVICE__ DenseMat<Vec> createDenseMat(Assembler &assembler) {
    return DenseMat(assembler.num_nodes * assembler.vars_per_node);
}
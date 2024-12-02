#pragma once

#include "../cuda_utils.h"

// director classes:

template <typename Derived, typename T>
class BaseDirector {
  public:
    __HOST_DEVICE__ static void setMatSkew(const T a, const T b[], T C[]) {
        C[0] = 0.0;
        C[1] = -a * b[2];
        C[2] = a * b[1];

        C[3] = a * b[2];
        C[4] = 0.0;
        C[5] = -a * b[0];

        C[6] = -a * b[1];
        C[7] = a * b[0];
        C[8] = 0.0;
    }

}; // end of base director class

// uses CFRP to have compile-time inheritance
template <typename T, int offset = 3>
class LinearizedRotation : public BaseDirector<LinearizedRotation<T,offset>, T> {
  public:
    static const int32_t num_params = 3;

    // TODO fix so in base class only
    __HOST_DEVICE__ static void setMatSkew(const T a, const T b[], T C[]) {
        C[0] = 0.0;
        C[1] = -a * b[2];
        C[2] = a * b[1];

        C[3] = a * b[2];
        C[4] = 0.0;
        C[5] = -a * b[0];

        C[6] = -a * b[1];
        C[7] = a * b[0];
        C[8] = 0.0;
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeRotationMat(const T vars[], T C[]) {
        const T *q = &vars[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            // C = I - q^x
            setMatSkew(-1.0, q, C);
            C[0] = C[4] = C[8] = 1.0;

            C += 9;
            q += vars_per_node;
        }
    }

    __HOST_DEVICE__ static T evalDrillStrain(const T u0x[], const T Ct[]) {
        // compute rotation penalty
        return 0.5 * (Ct[3] + u0x[3] - Ct[1] - u0x[1]);
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeDirector(const T vars[], const T t[], T d[]) {
        const T *q = &vars[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            A2D::VecCrossCore<T>(q, t, d);
            t += 3; d += 3;
            q += vars_per_node;
        }
    }
    
}; // end of class LinearizedRotation
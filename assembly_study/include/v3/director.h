
#pragma once
#include "cuda_utils.h"

template <typename T, int offset_ = 3>
class LinearizedRotationV3 {
   public:
    static const int32_t offset = offset_;
    static const int32_t num_params = 3;

    // TODO : add rotatoin mat products
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
    __HOST_DEVICE__ static void computeRotationMat(const int inode, const T vars[], T C[]) {
        const T *q = &vars[vars_per_node * inode + offset];
        // C = I - q^x
        setMatSkew(-1.0, q, C);
        C[0] = C[4] = C[8] = 1.0;
    }

    template <int vars_per_node>
    __HOST_DEVICE__ static void computeRotationMatScalarProduct(const int inode, const T vars[],
                                                                const T x[], const T y[], T &xCy) {
        const T *q = &vars[vars_per_node * inode + offset];
        xCy = x[0] * (y[0] + q[2] * y[1] - q[1] * y[2]);
        xCy += x[1] * (y[0] * -q[2] + y[1] + q[0] * y[2]);
        xCy += x[2] * (y[0] * q[1] - y[1] * q[0] + y[2]);
    }

    template <int vars_per_node>
    __HOST_DEVICE__ static void computeRotationMatScalarProductSens(const int inode, const T x[],
                                                                    const T y[], const T &xCyb,
                                                                    T res[]) {
        T *qb = &res[vars_per_node * inode + offset];
        qb[0] += xCyb * (x[1] * y[2] - x[2] * y[1]);
        qb[1] += xCyb * (x[2] * y[0] - x[0] * y[2]);
        qb[2] += xCyb * (x[0] * y[1] - x[1] * y[0]);
    }

    __HOST_DEVICE__ static T evalDrillStrain(const T u0x[], const T Ct[]) {
        // compute rotation penalty
        return 0.5 * (Ct[3] + u0x[3] - Ct[1] - u0x[1]);
    }

    __HOST_DEVICE__ static T evalDrillStrain(const T &u01, const T &u10, const T &C01,
                                             const T &C10) {
        // compute rotation penalty
        return 0.5 * (C10 + u10 - u01 - C01);
    }

    __HOST_DEVICE__ static void evalDrillStrainSens(const T etb, T &u01b, T &u10b, T &C01b,
                                                    T &C10b) {
        u01b = -0.5 * etb;
        u10b = 0.5 * etb;
        C10b = 0.5 * etb;
        C01b = -0.5 * etb;
    }
};
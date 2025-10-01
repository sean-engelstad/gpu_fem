#pragma once
#include "../../../cuda_utils.h"

template <typename T>
class QuadConstQuadrature {
   public:
    // Required static data used by other classes
    static constexpr int32_t num_quad_pts = 1;

    // get one of the four quad pts
    __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
        pt[0] = 0.0;
        pt[1] = 0.0;
        return 1.0;
    }
};

template <typename T>
class QuadLinearQuadrature {
   public:
    // Required static data used by other classes
    static constexpr int32_t num_quad_pts = 4;
    static constexpr double irt3 = 0.577350269189626; // 1/sqrt(3)

    // get one of the four quad pts
    __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
        double quad_pts[] = {-irt3, irt3};
        double quad_wts[] = {1.0, 1.0};
        pt[0] = quad_pts[ind % 2];
        pt[1] = quad_pts[ind / 2];
        return quad_wts[ind % 2] * quad_wts[ind / 2];
    }
};

template <typename T>
class QuadQuadraticQuadrature {
    public:
    static constexpr int32_t num_quad_pts = 9;
    static constexpr double rt35 = 0.7745966692414834;

    __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
        double quad_pts[] = {-rt35, 0.0, rt35};
        double quad_wts[] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        pt[0] = quad_pts[ind % 3];
        pt[1] = quad_pts[ind / 3];
        return quad_wts[ind % 3] * quad_wts[ind / 3];
    }
};
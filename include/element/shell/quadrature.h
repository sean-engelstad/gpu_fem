#pragma once
#include "../../cuda_utils.h"

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
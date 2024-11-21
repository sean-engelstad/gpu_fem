#pragma once
#include "cuda_utils.h"

template <typename T>
class TriangleQuadrature {
 public:
  // Required static data used by other classes
  static constexpr int32_t num_quad_pts = 3;

  // get one of the three triangle quad points
  __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
    switch (ind) {
      case 0:
        pt[0] = 0.5;
        pt[1] = 0.5;
      case 1:
        pt[0] = 0.0;
        pt[1] = 0.5;
      case 2:
        pt[0] = 0.5;
        pt[1] = 0.0;
    }
    return 1.0/3.0;
  };
};

#pragma once
#include "cuda_utils.h"

template <typename T, int offset_ = 3>
class LinearizedRotationV3 {
   public:
    static const int32_t offset = offset_;
    static const int32_t num_params = 3;

    // TODO : add rotatoin mat products
};
#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../cuda_utils.h"
#include "chrono"
#include "cublas_v2.h"
#include "utils/_cusparse_utils.h"
#include "utils/_utils.h"

namespace CUSPARSE {

template <typename T>
void direct_cholesky_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                           bool can_print = false) {
    // TODO : direct Cholesky solve
}
}  // namespace CUSPARSE
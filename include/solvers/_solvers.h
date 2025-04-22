#pragma once

#include "_utils.h"
// #include "linear_static_cpu.h"

#ifdef USE_GPU
#ifdef USE_CUSPARSE
#include "linear_static_cusparse.h"
#endif  // CUSPARSE
#endif  // USE_GPU

#include "nonlinear_static.h"
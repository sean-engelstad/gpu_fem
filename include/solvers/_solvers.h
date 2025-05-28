#pragma once

#ifdef USE_GPU
#ifdef USE_CUSPARSE

#include "linear_static/_utils.h"
#include "linear_static/bicg_stab.h"
#include "linear_static/direct_LU.h"
#include "linear_static/direct_chol.h"
#include "linear_static/gmres.h"

#endif  // CUSPARSE
#endif  // USE_GPU

#include "nonlinear_static.h"
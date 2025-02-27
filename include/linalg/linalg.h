#include "bsr_mat.h"
#include "dense_mat.h"
#include "precond.h"
#include "vec.h"

// linear solvers on CPU
#ifdef USE_EIGEN
#include "solvers/eigen.h"
#endif

// linear solvers on GPU
#ifdef USE_GPU
#ifdef USE_CUSPARSE
#include "solvers/cusparse.h"
#endif
#endif
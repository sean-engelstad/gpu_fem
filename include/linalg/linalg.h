#include "bsr_mat.h"
#include "dense_mat.h"
#include "vec.h"

#ifdef USE_GPU
#include "solvers/cusparse.h"
#endif
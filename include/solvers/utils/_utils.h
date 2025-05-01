#pragma once
#include "../../linalg/bsr_mat.h"

template <class Mat, class Vec>
Vec permute_rhs(Mat mat, Vec rhs) {
    int *perm = mat.getPerm();
    int block_dim = mat.getBlockDim();
    return rhs.createPermuteVec(block_dim, perm);
}

template <class Mat, class Vec>
void inv_permute_soln(Mat mat, Vec &soln) {
    int *iperm = mat.getIPerm();
    int block_dim = mat.getBlockDim();
    soln.permuteData(block_dim, iperm);
    return;
}

#ifdef USE_CUSPARSE
namespace CUBLAS {
double get_vec_norm(DeviceVec<double> vec) {
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    double nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, vec.getSize(), vec.getPtr(), 1, &nrm_R));

    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    return nrm_R;
}

void axpy(double a, DeviceVec<double> x, DeviceVec<double> y) {
    // Y := a * x + Y with DeviceVec's
    cublasHandle_t cublasHandle2 = NULL;
    cublasCreate(&cublasHandle2);

    cublasDaxpy(cublasHandle2, y.getSize(), &a, x.getPtr(), 1, y.getPtr(), 1);

    cublasDestroy(cublasHandle2);
}
}  // namespace CUBLAS
#endif  // CUSPARSE
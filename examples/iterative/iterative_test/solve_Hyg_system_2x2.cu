#include "linalg/vec.h"
#include "cuda_utils.h"

int main() {
    using T = double;

    // 2x2 upper triangular linear solve of H * y = g
    T g_arr[2] = {5, 1};
    HostVec<T> g(2, &g_arr[0]);

    T H_arr[4] = {1, 0, 2, 1};
    // T H_arr[4] = {1, 0, 0, 1};
    HostVec<T> H(4, &H_arr[0]);

    auto d_g = g.createDeviceVec();
    auto d_H = H.createDeviceVec();
    auto d_soln = DeviceVec<T>(2);

    // copy d_g into d_soln since cublasDtrsv solves in place
    d_g.copyValuesTo(d_soln);
    auto h_soln0 = d_soln.createHostVec();

    printf("init rhs:");
    printVec<double>(2, h_soln0.getPtr());

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // solve the upper triangular system H * y = g
    // since d_Hred is stored row-major but this is column major, I need CUBLAS_OP_N not CUBLAS_OP_T
    CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                2, d_H.getPtr(), 2, d_soln.getPtr(), 1));

    auto h_soln = d_soln.createHostVec();
    printf("cpp soln:");
    printVec<double>(2, h_soln.getPtr());

    // the correct answer is
    T y_arr[2] = {3, 1};
    HostVec<T> y(2, &y_arr[0]);
    printf("py soln:");
    printVec<double>(2, y.getPtr());

    // check if residual is zero
    auto d_Hy = DeviceVec<T>(2);
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(cublasHandle, CUBLAS_OP_N, 2, 2, &alpha, d_H.getPtr(), 2, d_soln.getPtr(), 1, &beta, d_Hy.getPtr(), 1);

    // Compute residual: residual = Hy - g
    alpha = -1.0;
    cublasDaxpy(cublasHandle, 2, &alpha, d_g.getPtr(), 1, d_Hy.getPtr(), 1);

    // Compute the norm of the residual
    double resNorm;
    cublasDnrm2(cublasHandle, 2, d_Hy.getPtr(), 1, &resNorm);
    printf("Residual norm: %e\n", resNorm);


};
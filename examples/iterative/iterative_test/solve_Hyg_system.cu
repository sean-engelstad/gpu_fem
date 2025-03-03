#include "linalg/vec.h"
#include "cuda_utils.h"

int main() {
    using T = double;

    // 12x12 upper triangular linear solve
    T g_arr[12] = {-1.39035e-02,2.67838e-03,-6.10114e-04,1.83382e-04,-5.90929e-05,1.26729e-05,-5.37025e-06,1.05014e-06,-1.36319e-07,2.16767e-08,-1.59501e-09,8.50705e-11};
    HostVec<T> g(12, &g_arr[0]);

    T H_arr[144] = {6.28135e+00,2.17168e+00,-4.07264e+00,-2.89067e+00,-2.73921e+00,1.17444e+00,7.50228e-01,-2.58769e-01,9.85943e-01,-3.31270e+00,1.05184e+00,1.18151e-01,0.00000e+00,1.47361e+00,1.50326e+00,-1.12131e-02,-1.72812e-01,1.08854e+00,-5.71957e-01,-2.68403e-01,-6.14744e-01,-5.13172e-01,-3.11051e-01,-2.79346e-01,0.00000e+00,0.00000e+00,1.02646e+00,4.27503e-01,-1.78277e-01,2.27982e-01,-6.72395e-02,-1.55730e-04,-1.80262e-01,-9.74940e-02,-3.07996e-02,-6.17172e-02,0.00000e+00,0.00000e+00,0.00000e+00,1.12811e+00,4.48302e-01,8.93278e-01,-5.01751e-01,-2.59288e-01,-4.92248e-01,-1.84987e-01,-2.53253e-01,-2.81741e-01,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,8.29495e-01,7.90279e-01,-1.50474e-01,-1.40964e-01,-3.13114e-01,-1.16186e-01,-1.14682e-01,-1.24812e-01,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,6.14198e-01,7.34915e-01,1.26234e-01,1.19217e-01,2.80408e-01,-1.67810e-02,1.44005e-01,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,6.16477e-01,2.38106e-01,-2.33719e-01,-1.64283e-01,-1.61357e-01,-1.15175e-01,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,8.96420e-01,2.37384e-01,1.06226e-01,-9.24861e-03,5.45770e-02,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,7.71127e-01,2.17650e-01,-5.61532e-02,2.17692e-02,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,9.99158e-01,1.40095e-01,4.42230e-03,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,9.29482e-01,1.03351e-01,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,1.02229e+00};
    T H_arr_T[144];
    // have to transpose the matrix apparently for cublas to like it (needs to be column major?)
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            H_arr_T[i * 12 + j] = H_arr[j * 12 + i];
        }
    }

    HostVec<T> H(144, &H_arr_T[0]);

    auto d_g = g.createDeviceVec();
    auto d_H = H.createDeviceVec();
    auto d_soln = DeviceVec<T>(12);

    // copy d_g into d_soln since cublasDtrsv solves in place
    d_g.copyValuesTo(d_soln);
    auto h_soln0 = d_soln.createHostVec();

    printf("init rhs:");
    printVec<double>(12, h_soln0.getPtr());

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // solve the upper triangular system H * y = g
    // since d_Hred is stored row-major but this is column major, I need CUBLAS_OP_N not CUBLAS_OP_T
    CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                12, d_H.getPtr(), 12, d_soln.getPtr(), 1));

    auto h_soln = d_soln.createHostVec();
    printf("cpp soln:");
    printVec<double>(12, h_soln.getPtr());

    // the correct answer is
    T y_arr[12] = {-3.49138376e-03,  2.48665512e-03, -6.92562164e-04,  1.74563755e-04,
        -1.02759320e-04,  3.14710186e-05, -9.24538328e-06,  1.21734669e-06,
        -1.83098467e-07,  2.19365047e-08, -1.72527324e-09,  8.32156237e-11};
    HostVec<T> y(12, &y_arr[0]);
    printf("py soln:");
    printVec<double>(12, y.getPtr());

    // check if residual is zero
    auto d_Hy = DeviceVec<T>(12);
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(cublasHandle, CUBLAS_OP_N, 12, 12, &alpha, d_H.getPtr(), 12, d_soln.getPtr(), 1, &beta, d_Hy.getPtr(), 1);

    // Compute residual: residual = Hy - g
    alpha = -1.0;
    cublasDaxpy(cublasHandle, 12, &alpha, d_g.getPtr(), 1, d_Hy.getPtr(), 1);

    // Compute the norm of the residual
    double resNorm;
    cublasDnrm2(cublasHandle, 12, d_Hy.getPtr(), 1, &resNorm);
    printf("Residual norm: %e\n", resNorm);


};
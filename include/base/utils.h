#pragma once

#include "cuda_utils.h"
#include "stdlib.h"

template <typename T> __HOST_DEVICE__ void printVec(const int N, const T *vec);

template <> __HOST_DEVICE__ void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template <>
__HOST_DEVICE__ void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i]);
    }
    printf("\n");
}

__HOST_DEVICE__ bool is_unique(int N, int32_t *local_conn) {
    // check whether all values are different or not
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            if (local_conn[i] == local_conn[j]) {
                return false;
            }
        }
    }
    return true;
}

__HOST__ void make_unique_conn(int nstrides, int stride, int maxVal,
                               int32_t *conn) {
    for (int ielem = 0; ielem < nstrides; ielem++) {
        int32_t *local_conn = &conn[stride * ielem];
        while (!is_unique(stride, local_conn)) {
            for (int iloc = 0; iloc < stride; iloc++) {
                local_conn[iloc] = rand() % maxVal;
            }
        }
    }
}
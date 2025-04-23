#include <cmath> 
#include <cassert>
#include <cstdio>

template <typename T>
void genLaplaceCSR(int *rowp, int *cols, double *vals, int N, int nz, double *rhs) {
    // second order laplace operator on a square domain with nxn nodes for n^2 = N dof
    // linear system based off CUDA samples github here.. created on host first
    // https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradientPrecond

    int n = (int) sqrt((double)N);
    assert(n*n==N);
    printf("laplace dimension = %d\n", n);
    int idx = 0;

    // loop over the rows
    for (int i = 0; i < N ; i++) {
        int ix = i % n;
        int iy = i / n;

        rowp[i] = idx;

        // up
        if (iy > 0)
        {
            vals[idx] = 1.0;
            cols[idx] = i - n;
            idx++;
        }
        else
        {
            rhs[i] -= 1.0;
        }

        // left
        if (ix > 0) {
            vals[idx] = 1.0;
            cols[idx] = i - 1;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }

        // center
        vals[idx] = -4.0;
        cols[idx] = i;
        idx++;

        //right
        if (ix  < n - 1)
        {
            vals[idx] = 1.0;
            cols[idx] = i + 1;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

        // down
        if (iy  < n - 1)
        {
            vals[idx] = 1.0;
            cols[idx] = i + n;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }
    }

    rowp[N] = idx;
}

template <typename T>
void printVec(const int N, const T *vec);

template <>
void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template <>
void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i]);
    }
    printf("\n");
}
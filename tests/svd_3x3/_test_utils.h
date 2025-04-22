#include "a2dcore.h"

template <int N>
double rel_err(double vec1[N], double vec2[N]) {
    double max_rel_err = 0;
    for (int i = 0; i < N; i++) {
        double c_rel_err = std::abs((vec1[i] - vec2[i]) / vec2[i]);
        if (c_rel_err > max_rel_err) {
            max_rel_err = c_rel_err;
        }
    }
    return max_rel_err;
}

template <int N>
double abs_err(double vec1[N], double vec2[N]) {
    double max_abs_err = 0;
    for (int i = 0; i < N; i++) {
        double c_abs_err = std::abs(vec1[i] - vec2[i]);
        // printf("v1 %.4e, v2 %.4e, c_abs_err %.4e, max_abs_err %.4e\n", vec1[i], vec2[i], c_abs_err,
        //        max_abs_err);
        if (c_abs_err > max_abs_err) {
            max_abs_err = c_abs_err;
        }
    }
    return max_abs_err;
}
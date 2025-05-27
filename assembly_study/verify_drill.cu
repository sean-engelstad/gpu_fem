#include "../tests/test_commons.h"
#include "chrono"
#include "linalg/_linalg.h"

// shell imports for local performance optimization
#include "include/v1/v1.h"
#include "include/v3/v3.h"

template <typename T, class ElemGroup, class Data>
__global__ void test_drill_strain_kernel() {

    int iquad = threadIdx.x;
    int inode = iquad;
    Data physData{7e9, 0.3, 1e-2};

    T xpts[12], vars[24];
    for (int i = 0; i < 12; i++) {
        xpts[i] = 0.123 + 0.5234 * i + 0.00123 * i * i;
    }
    for (int i = 0; i < 24; i++) {
        vars[i] = -0.123 + 0.5234 * i - 0.00123 * i * i;
    }

    // XdinvT, Tmat
    T Tmatn[36], XdinvTn[36], detXdq;
    bool active_thread = true;
    ElemGroup::template compute_nodal_transforms<Data>(active_thread, inode, xpts, physData, &Tmatn[9 * inode], &XdinvTn[9 * inode]);
    ElemGroup::template compute_quadpt_transforms<Data>(active_thread, iquad, xpts, &detXdq);

    if (threadIdx.x == 0) {
        printf("Tmatn:");
        printVec<T>(9, &Tmatn[9 * inode]);
        printf("XdinvTn:");
        printVec<T>(9, &XdinvTn[9 * inode]);
    }

    // test for single element
    T local_res[24];
    memset(local_res, 0.0, 24 * sizeof(T));

    ElemGroup::add_drill_strain_quadpt_residual_fast(active_thread, iquad, vars, physData, &Tmatn[9 * inode], 
        &XdinvTn[9 * inode], detXdq, local_res);

    // if (threadIdx.x == 0) {
    //     printf("local_res:");
    //     printVec<T>(24, local_res);
    // }
}

template <typename T, int vars_per_node, class Data, class Basis, class Director>
void test_drill_strain_fwd(const T quad_pt[], const Data &physData, const T xpts[],
                           const T vars[]) {
    T et1;

    // original fwd
    T fn[12];
    ShellComputeNodeNormals<T, Basis>(xpts, fn);
    ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(quad_pt, physData.refAxis,
                                                                     xpts, vars, fn, &et1);
    T et2 = 5.50449708e-01;
    printf("\tet1 = %.8e\n", et1);
}

void test_v1() {
    using T = double;
    constexpr bool is_nonlinear = true; // true
    using Quad = QuadLinearQuadratureV1<T>;
    using Director = LinearizedRotationV1<T>;
    using Basis = ShellQuadBasisV1<T, Quad, 2>;
    using Data = ShellIsotropicDataV1<T, false>;
    using Physics = IsotropicShellV1<T, Data, is_nonlinear>;
    using ElemGroup1 = ShellElementGroupV1<T, Director, Basis, Physics>;

    Data physData{7e9, 0.3, 1e-2};

    T xpts[12], vars[24];
    for (int i = 0; i < 12; i++) {
        xpts[i] = 0.123 + 0.5234 * i + 0.00123 * i * i;
    }
    for (int i = 0; i < 24; i++) {
        vars[i] = -0.123 + 0.5234 * i - 0.00123 * i * i;
    }

    T quad_pt[2];
    Quad::getQuadraturePoint(0, quad_pt);

    test_drill_strain_fwd<T, 6, Data, Basis, Director>(quad_pt, physData, xpts, vars);
}

void test_v3() {
    using T = double;
    constexpr bool is_nonlinear = true; // true
    constexpr int kernel_option = 4;

    using Quad = QuadLinearQuadratureV3<T>;
    using Director = LinearizedRotationV3<T>;
    using Basis = ShellQuadBasisV3<T, Quad>;
    using Data = ShellIsotropicDataV3<T, false>;
    using Physics = IsotropicShellV3<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroupV3<T, Director, Basis, Physics, kernel_option>; //, full_strain>;
    using Assembler = ElementAssemblerV3<T, ElemGroup, VecType, BsrMat>;

    test_drill_strain_kernel<T, ElemGroup, Data><<<1,4>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
}


int main() {

    printf("\n");
    test_v1();
    printf("-------\n");
    test_v3();

    return 0;
}
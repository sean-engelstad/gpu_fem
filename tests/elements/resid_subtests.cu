#include "../test_commons.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "utils/local_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/shell_elem_group_v2.h"

// template <typename T, class Basis>
// void test_fast_interp(const T quad_pt[], const T xpts[]) {


//     T Xxi[3], Xeta[3];
//     Basis::template interpFieldsGrad<vars_per_node, 3>(quad_pt, xpts, Xxi, Xeta);
//     printf("")

//     T Xxi_2 = Basis::template interpFieldsGradLight<XI, vars_per_node>(0, quad_pt, xpts);
//     printf("Xxi_1 %.8e, Xxi_2 %.8e\n", Xxi[0], Xxi_2);
// }

template <typename T, int vars_per_node, class Data, class Basis, class Director>
void test_drill_strain_fwd(const T quad_pt[], const Data &physData, const T xpts[], const T vars[]) {
    T et1, et2;

    // original fwd
    T fn[12];
    ShellComputeNodeNormals<T, Basis>(xpts, fn);
    ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                quad_pt, physData.refAxis, xpts, vars, fn, &et1);

    // new fwd
    printf("\n----------\n");
    T Tmat[36], XdinvT[36];
    for (int inode = 0; inode < 4; inode++) {
        ShellComputeNodalTransforms<T, Data, Basis>(inode, xpts, physData, &Tmat[9*inode], &XdinvT[9*inode]);
    }
    ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(
                quad_pt, xpts, vars, Tmat, XdinvT, et2);

    double err = rel_err(et1, et2);
    bool passed = err < 1e-10;
    printTestReport("drill strain fwd", passed, err);
    printf("\tet1 = %.8e, et2 = %.8e\n", et1, et2);
}

template <typename T, class Data, class Basis, class ElemGroup1, class ElemGroup2>
void test_drill_strain_resid(const int iquad, const T xpts[], const T vars[], const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 1 for drill strains
    constexpr int debug_mode = 1;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars, physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[36], XdinvT[36];
    for (int inode = 0; inode < 4; inode++) {
        ShellComputeNodalTransforms<T, Data, Basis>(inode, xpts, physData, &Tmat[9*inode], &XdinvT[9*inode]);
    }
    ElemGroup2::template _add_drill_strain_quadpt_residual_fast<Data>(true, iquad, xpts, vars, physData, Tmat, XdinvT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("drill strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}

template <typename T, class Data, class Basis, class Quadrature, class ElemGroup1, class ElemGroup2>
void test_tying_strain_resid(const int iquad, const T xpts[], const T vars[], const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 2 for tying strains
    constexpr int debug_mode = 2;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars, physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[9], XdinvT[9], XdinvzT[9];
    ShellComputeQuadptTransforms<T, Data, Basis, Quadrature>(iquad, xpts, physData, Tmat, XdinvT, XdinvzT);
    ElemGroup2::template _add_tying_strain_quadpt_residual_fast<Data>(true, iquad, xpts, vars, physData, XdinvT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("tying strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}

template <typename T, class Data, class Basis, class Quadrature, class ElemGroup1, class ElemGroup2>
void test_bending_strain_resid(const int iquad, const T xpts[], const T vars[], const Data &physData) {
    T resid1[24], resid2[24];
    for (int i = 0; i < 24; i++) {
        resid1[i] = 0.0;
        resid2[i] = 0.0;
    }

    // original resid computation with debug_mode = 3 for bending strains
    constexpr int debug_mode = 3;
    ElemGroup1::template add_element_quadpt_residual<Data, debug_mode>(true, iquad, xpts, vars, physData, resid1);

    // new resid computation
    // printf("\n----------\n");
    T Tmat[9], XdinvT[9], XdinvzT[9];
    ShellComputeQuadptTransforms<T, Data, Basis, Quadrature>(iquad, xpts, physData, Tmat, XdinvT, XdinvzT);
    ElemGroup2::template _add_bending_strain_quadpt_residual_fast<Data>(true, iquad, xpts, vars, physData, Tmat, XdinvT, XdinvzT, resid2);

    double err = rel_err(24, resid1, resid2);
    bool passed = err < 1e-10;
    printTestReport("bending strain resid", passed, err);
    // printf("\tresid1:");
    // printVec<T>(24, resid1);
    // printf("\tresid2:");
    // printVec<T>(24, resid2);
}

int main() {

    using T = double;
    constexpr bool is_nonlinear = true; // true
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Data = ShellIsotropicData<T, true>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup1 = ShellElementGroup<T, Director, Basis, Physics>;
    using ElemGroup2 = ShellElementGroupV2<T, Director, Basis, Physics>;

    int iquad = 0;
    T quad_pt[2];
    quad_pt[0] = -0.57735;
    quad_pt[1] = 0.57735;
    constexpr int vpn = 6;

    T refAxis[3] = {1.0, 0.0, 0.0};
    Data physData{7e9, 0.3, 1e-2, refAxis};

    T xpts[12], vars[24];
    for (int i = 0; i < 12; i++) {
        xpts[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < 24; i++) {
        vars[i] = (double)rand() / RAND_MAX;
    }

    // smaller tests
    // test_normal_vecs<T, Basis>(0, quad_pt, xpts); // passed
    test_drill_strain_fwd<T, vpn, Data, Basis, Director>(quad_pt, physData, xpts, vars);

    // full resid tests against old GPU code
    test_drill_strain_resid<T, Data, Basis, ElemGroup1, ElemGroup2>(iquad, xpts, vars, physData);
    test_tying_strain_resid<T, Data, Basis, Quad, ElemGroup1, ElemGroup2>(iquad, xpts, vars, physData);
    test_bending_strain_resid<T, Data, Basis, Quad, ElemGroup1, ElemGroup2>(iquad, xpts, vars, physData);

    return 0;
}
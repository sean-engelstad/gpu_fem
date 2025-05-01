#include "a2dcore.h"
#include "a2ddefs.h"
#include "base/utils.h"
#include "shell/basis.h"
#include "shell/director.h"
#include "shell/physics/isotropic_shell.h"
#include "shell/shell_elem_group.h"
#include "shell/shell_utils.h"

template <bool is_nonlinear = false>
void testTyingStrainFwd() {
    using T = double;
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    T Xpts[12], fn[12], vars[24], d[12];
    T p_vars[24], p_d[12];
    A2D::ADScalar<T, 1> f_xpts[12], f_fn[12];  // think I need this just to use one type
    A2D::ADScalar<T, 1> f_vars[24], f_d[12];   // forward AD type

    // init values
    // static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < 12; i++) {
        Xpts[i] = 4.1234 + 0.2424 * i;
        fn[i] = 9.134 - 0.6535 * i * i;
        d[i] = -5.322 + 0.3252 * i + 5.453 * i * i;
        p_d[i] = 1.8732 + 0.9821 * i - 0.5453 * i * i;
        f_xpts[i].value = Xpts[i];
        f_xpts[i].deriv[0] = 0.0;
        f_fn[i].value = fn[i];
        f_fn[i].deriv[0] = 0.0;
        f_d[i].value = d[i];
        f_d[i].deriv[0] = p_d[i];
    }
    for (int i = 0; i < 24; i++) {
        vars[i] = 4.1332 + 0.7131 * i - 0.453 * i * i;
        p_vars[i] = 4.9752 - 2.9921 * i - 0.5323 * i * i;
        f_vars[i].value = vars[i];
        f_vars[i].deriv[0] = p_vars[i];
    }

    if constexpr (is_nonlinear) {
        printf("nonlinear tying strain Hfwd test:\n");
    } else {
        printf("linear tying strain Hfwd test:\n");
    }

    // analytic forward computation
    // A2D::Vec<T, Basis::num_all_tying_points> p_ety;
    T p_ety[Basis::num_all_tying_points];
    computeTyingStrainHfwd<T, Physics, Basis>(Xpts, fn, vars, d, p_vars, p_d, p_ety);
    printf("\tp_ety analytic[%d]:", Basis::num_all_tying_points);
    printVec<T>(Basis::num_all_tying_points, p_ety);

    // forward AD with ADScalar version
    using T2 = A2D::ADScalar<T, 1>;
    using Quad2 = QuadLinearQuadrature<T2>;
    using Director2 = LinearizedRotation<T2>;
    using Basis2 = ShellQuadBasis<T2, Quad2, 2>;
    using Data2 = ShellIsotropicData<T2, has_ref_axis>;
    using Physics2 = IsotropicShell<T2, Data2, is_nonlinear>;

    A2D::ADScalar<T, 1> f_ety[Basis::num_all_tying_points];
    computeTyingStrain<T2, Physics2, Basis2, is_nonlinear>(f_xpts, f_fn, f_vars, f_d, f_ety);
    T p_ety2[Basis::num_all_tying_points];
    for (int i = 0; i < Basis::num_all_tying_points; i++) {
        p_ety2[i] = f_ety[i].deriv[0];
    }
    printf("\tp_ety fwd AD[%d]:", Basis::num_all_tying_points);
    printVec<T>(Basis::num_all_tying_points, p_ety2);

    T ety_test_vec[Basis::num_all_tying_points];
    T ety_TD1 = 0.0, ety_TD2 = 0.0;
    for (int i = 0; i < Basis::num_all_tying_points; i++) {
        T rand_test = static_cast<double>(rand()) / RAND_MAX;
        ety_TD1 += p_ety[i] * rand_test;
        ety_TD2 += p_ety2[i] * rand_test;
    }

    printf("\tTD: analytic %.5e, fwd AD %.5e\n", ety_TD1, ety_TD2);
}

// hrev test TODO

int main(void) {
    // linear hfwd
    testTyingStrainFwd<false>();
    // nonlinear hfwd
    testTyingStrainFwd<true>();

    // TODO : hrev test next
}
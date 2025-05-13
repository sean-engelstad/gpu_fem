#include "../test_commons.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "utils//local_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "solvers/_solvers.h"

template <bool is_nonlinear = false>
void test_energy_resid(double h = 1e-30, bool print = false) {
    /** test Im{U(u+1j*h*p)}/h vs <p,r(u)>, or complex-step comparison of energy vs resid */

    using T = A2D_complex_t<double>;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

    // printf("running!\n");
    int num_bcs = 2;
    auto assembler = createOneElementAssembler<Assembler>(num_bcs);

    // init variables u
    int num_vars = assembler.get_num_vars();
    auto res = assembler.createVarsVec();
    auto h_vars = HostVec<T>(num_vars);
    auto h_vars_pert = HostVec<T>(num_vars);
    auto p_vars = HostVec<T>(num_vars);

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        if (is_nonlinear) h_vars[ivar] *= 1e6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        h_vars_pert[ivar] = h_vars[ivar] + h * T(0.0, 1.0) * p_vars[ivar];
    }

    // strain energy total derivative Im{U(x+1j*h*p)} / h
    T Uelem = 0.0;
    assembler.set_variables(h_vars_pert);
    assembler.add_energy(&Uelem);
    double Uelem_TD = A2D::ImagPart(Uelem) / h;

    assembler.set_variables(h_vars);
    assembler.add_residual(res);

    // now take <p, res(x)>
    double res_TD = 0.0;
    for (int i = 0; i < 24; i++) {
        res_TD += A2D::RealPart(res[i] * p_vars[i]);
    }

    double my_rel_err = rel_err(Uelem_TD, res_TD);
    bool passed = my_rel_err < 1e-10;
    if constexpr (is_nonlinear) {
        printTestReport("shell elem, linear dense, Im{U(u+p*h*j)}/h vs <p,r(u)>", passed,
                        my_rel_err);
    } else {
        printTestReport("shell elem, geomNL dense, Im{U(u+p*h*j)}/h vs <p,r(u)>", passed,
                        my_rel_err);
    }
    printf("\tUelem_TD %.4e, res_TD %.4e\n", Uelem_TD, res_TD);
}

template <bool is_nonlinear = false>
void test_resid_jac(double h = 1e-30, bool print = false) {
    /** test <p,r(u+1j*h*q)> vs <p,K_t(u)*q>, or complex-step comparison of resid vs energy */

    using T = A2D_complex_t<double>;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

    // printf("running!\n");
    int num_bcs = 2;
    auto assembler = createOneElementAssembler<Assembler>(num_bcs);

    // init variables u
    int num_vars = assembler.get_num_vars();
    auto res = assembler.createVarsVec();
    auto h_vars = HostVec<T>(num_vars);
    auto h_vars_pert = HostVec<T>(num_vars);
    auto p_vars = HostVec<T>(num_vars);
    auto q_vars = HostVec<T>(num_vars);
    DenseMat<VecType<T>> mat(num_vars);

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        if (is_nonlinear) h_vars[ivar] *= 1e6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        q_vars[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
        h_vars_pert[ivar] = h_vars[ivar] + h * T(0.0, 1.0) * p_vars[ivar];
    }

    // resid total derivative <p,r(u+1j*h*q)>
    assembler.set_variables(h_vars_pert);
    assembler.add_residual(res);
    double res_TD = 0.0;
    for (int i = 0; i < 24; i++) {
        res_TD += A2D::RealPart(A2D::ImagPart(res[i]) / h * q_vars[i]);
    }

    // now take <p,K_t(u)*q>
    assembler.set_variables(h_vars);
    assembler.add_jacobian(res, mat);
    double jac_TD = 0.0;
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            jac_TD += A2D::RealPart(mat[24 * i + j] * p_vars[i] * q_vars[j]);
        }
    }

    double my_rel_err = rel_err(res_TD, jac_TD);
    bool passed = my_rel_err < 1e-10;
    if constexpr (is_nonlinear) {
        printTestReport("shell elem, linear dense, <p,r(u+1j*h*q)> vs <p,K_t(u)*q>", passed,
                        my_rel_err);
    } else {
        printTestReport("shell elem, geomNL dense, <p,r(u+1j*h*q)> vs <p,K_t(u)*q>", passed,
                        my_rel_err);
    }
    printf("\tres_TD %.4e, jac_TD %.4e\n", res_TD, jac_TD);
}

int main(void) {
    // TODO : maybe also test with the BSRMat too?

    double h = 1e-30;
    bool print = true;
    test_energy_resid<false>(h, print);
    test_energy_resid<true>(h, print);
    test_resid_jac<false>(h, print);
    test_resid_jac<true>(h, print);
    return 0;
};
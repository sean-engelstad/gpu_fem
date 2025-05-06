#include "chrono"
#include "linalg/_linalg.h"
#include "utils/local_utils.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <bool is_nonlinear>
void test_elemres_GPU() {
    using T = double;
    bool print = false;

  // clang-format off
    T cpu_NL_res[] = {5.76706e+15,6.00566e+15,6.24202e+15,-6.39142e+13,-1.56746e+12,6.08050e+13,
        -6.47856e+15,-6.85395e+15,-7.22708e+15,-1.68776e+14,-3.75695e+12,1.61340e+14,-6.11549e+15,
        -5.57984e+15,-5.05506e+15,-1.36488e+14,-2.70770e+12,1.31150e+14,6.82699e+15,6.42812e+15,
        6.04012e+15,-3.16260e+13,-5.18197e+11,3.06153e+13};
    // clang-format off
    T cpu_LIN_res[] = {1.57657e+09,1.78364e+09,1.99466e+09,-4.48443e+05,1.89786e+05,8.53706e+05,
        -1.95664e+09,-2.18565e+09,-2.41860e+09,-4.37137e+05,4.63284e+05,1.44078e+06,-2.01710e+09,
        -2.18230e+09,-2.34974e+09,-5.43220e+05,3.28840e+05,1.27797e+06,2.39718e+09,2.58431e+09,
        2.77368e+09,-5.54519e+05,5.53505e+04,6.90911e+05};

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
    auto res = assembler.createVarsVec();
    int num_vars = assembler.get_num_vars();
    auto h_vars = HostVec<T>(num_vars);
    auto p_vars = HostVec<T>(num_vars);

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        if (is_nonlinear) h_vars[ivar] *= 1e6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    }

    auto vars = h_vars.createDeviceVec();
    assembler.set_variables(vars);
    

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_residual(res);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute total direc derivative of analytic residual
    auto h_res = res.createHostVec();
    T res_TD = A2D::VecDotCore<T, 24>(p_vars.getPtr(), h_res.getPtr());
    if (print) printf("Analytic residual\n");
    if (print) printf("res TD = %.8e\n", res_TD);

    // print data of host residual
    if (print) printf("res: ");
    if (print) printVec<double>(24, h_res.getPtr());

    if constexpr (is_nonlinear) {
        double max_ref = max(24, cpu_NL_res);
        double max_abs_err = abs_err(h_res, cpu_NL_res);
        double rel_err = max_abs_err / max_ref;
        bool passed = rel_err < 1e-5;
        printTestReport("shell elem-res geom nonlinear", passed, rel_err);
        printf("\tabs err %.4e, max ref %.4e, norm err %.4e\n", max_abs_err, max_ref, rel_err);
    } else {
        double max_ref = max(24, cpu_LIN_res);
        double max_abs_err = abs_err(h_res, cpu_LIN_res);
        double rel_err = max_abs_err / max_ref;
        bool passed = rel_err < 1e-5;
        printTestReport("shell elem-res linear", passed, rel_err);
        printf("\tabs err %.4e, max ref %.4e, norm err %.4e\n", max_abs_err, max_ref, rel_err);
    }
    

    // old way with rel err check
    // double max_rel_err = rel_err(h_res, cpu_LIN_res, 1e-9);
    // bool passed = max_rel_err < 1e-5;
    // printTestReport("shell elem-res linear", passed, max_rel_err);

    printKernelTiming(duration.count());
}

int main() {
    
    test_elemres_GPU<false>(); // linear
    test_elemres_GPU<true>(); // nonlinear
    return 0;
};
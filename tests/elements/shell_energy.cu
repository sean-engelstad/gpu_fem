#include "chrono"
#include "linalg/_linalg.h"
#include "utils/local_utils.h"
#include "../test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <bool is_nonlinear>
void test_Uelem_GPU() {
    using T = double;
    bool print = false;

  // clang-format off
    T cpu_LIN_energy = 6.97367614927571e+04;
    T cpu_NL_energy = 7.27806646240768e+16;

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
    DeviceVec<T> d_Uelem{1};
    assembler.add_energy(d_Uelem.getPtr());
    auto h_Uelem = d_Uelem.createHostVec();
    T Uelem = h_Uelem[0];
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    if (print) printf("Uelem = %.10e\n", Uelem);

    if constexpr (is_nonlinear) {
        double err = rel_err(Uelem, cpu_NL_energy);
        bool passed = err < 1e-10;
        printTestReport("shell Uelem geom nonlinear", passed, err);
    } else {
        double err = rel_err(Uelem, cpu_LIN_energy);
        bool passed = err < 1e-10;
        printTestReport("shell Uelem linear", passed, err);
    }

    printKernelTiming(duration.count());
}

int main() {
    
    test_Uelem_GPU<false>(); // linear
    test_Uelem_GPU<true>(); // nonlinear
    return 0;
};
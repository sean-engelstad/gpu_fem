#include "chrono"
#include "linalg/linalg.h"
#include "local_utils.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
    using T = A2D_complex_t<double>;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

    // printf("running!\n");
    int num_bcs = 2;
    auto assembler = createOneElementAssembler<Assembler>(num_bcs);

    // init variables u
    auto h_vars = assembler.createVarsHostVec();
    auto p_vars = assembler.createVarsHostVec();
    auto res = assembler.createVarsVec();

    // fixed perturbations of the host and pert vars
    double h = 1e-30;
    for (int ivar = 0; ivar < 24; ivar++) {
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        h_vars[ivar] += T(0.0, p_vars[ivar].real() * h);
    }
    
    auto vars = convertVecType<T>(h_vars);
    assembler.set_variables(vars);

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();

    assembler.add_residual(res);

    T Uenergy = 0.0;
    assembler.add_energy(Uenergy);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute total direc derivative of analytic residual
    double res_TD = A2D::ImagPart(Uenergy) / h;
    printf("Complex step residual\n");
    printf("res TD complex step = %.8e\n", res_TD);

    auto h_res = res.createHostVec();
    double res_TD_analytic = A2D::VecDotCore<T, 24>(p_vars.getPtr(), h_res.getPtr()).real();
    printf("res TD analytic = %.8e\n", res_TD_analytic);

    printf("took %d microseconds to run add residual\n", (int)duration.count());

    // print data of strain energy
    printf("Uenergy = %.8e, %.8e\n", A2D::RealPart(Uenergy), A2D::ImagPart(Uenergy));

    return 0;
};
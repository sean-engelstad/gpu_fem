#include "chrono"
#include "linalg/linalg.h"
#include "local_utils.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
    using T = double;

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

    // fixed perturbations of the host and pert vars
    double h = 1e-30;
    for (int ivar = 0; ivar < 24; ivar++) {
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    }
    
    auto vars = convertVecType<T>(h_vars);
    assembler.set_variables(vars);

    // time add energy method
    auto start = std::chrono::high_resolution_clock::now();

    T Uenergy = 0.0;
    assembler.add_energy(Uenergy);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute total direc derivative of analytic residual

    printf("took %d microseconds to run add residual\n", (int)duration.count());

    // print data of strain energy
    printf("Uenergy = %.8e\n", Uenergy);

    return 0;
};
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
    auto res = assembler.createVarsVec();

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    }
    
    auto vars = convertVecType<T>(h_vars);
    assembler.set_variables(vars);

    DenseMat<VecType<T>> mat(24);  

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();

    assembler.add_jacobian(res, mat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute total direc derivative of analytic residual
    auto h_res = res.createHostVec();
    T res_TD = A2D::VecDotCore<T, 24>(p_vars.getPtr(), h_res.getPtr());
    printf("Analytic residual\n");
    printf("res TD = %.8e\n", res_TD);

    // print data of host residual
    printf("res: ");
    printVec<double>(24, h_res.getPtr());

    printf("took %d microseconds to run add residual\n", (int)duration.count());

    return 0;
};
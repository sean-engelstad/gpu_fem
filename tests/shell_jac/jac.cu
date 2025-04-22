#include "chrono"
#include "linalg/_linalg.h"
#include "../shell_res/local_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

// get jacobian directional derivative analytically on the CPU

int main(void) {
    using T = double;

#ifdef NLINEAR
    constexpr bool is_nonlinear = true;
#else
    constexpr bool is_nonlinear = false;
#endif

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
    auto p_vars = HostVec<T>(num_vars);
    auto p_vars2 = HostVec<T>(num_vars);    

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
#ifdef NLINEAR
        h_vars[ivar] *= 1e6;
#endif
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
    }
    
    auto vars = h_vars.createDeviceVec();
    assembler.set_variables(vars);

    DenseMat<VecType<T>> mat(num_vars);  

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();

    assembler.add_jacobian(res, mat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print res, jac
    auto h_res = res.createHostVec();
    auto h_mat = mat.createHostVec();
    T jac_TD = 0.0;
    T res_TD = 0.0;
    for (int i = 0; i < 24; i++) {
        res_TD += p_vars[i] * h_res[i];
      for (int j = 0; j < 24; j++) {
        jac_TD += p_vars[i] * p_vars2[j] * h_mat[24*i+j];
      }
    }

    printf("Analytic Jacobian GPU\n");
    printf("res TD = %.8e\n", res_TD);
    printf("jac TD = %.8e\n", jac_TD);

    // print residual
    printf("res: ");
    printVec<double>(num_vars, h_res.getPtr());
    
    const double *h_mat_ptr = h_mat.getPtr();
    for (int i = 0; i < 24; i++) { // i < 2
        printf("kmat row %d: ", i);
        printVec<double>(num_vars, &h_mat_ptr[num_vars*i]);
    }

    printf("took %d microseconds to run add jacobian\n", (int)duration.count());

    return 0;
};
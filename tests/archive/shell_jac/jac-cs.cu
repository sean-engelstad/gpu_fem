#include "chrono"
#include "linalg/linalg.h"
#include "../shell_res/local_utils.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
    using T = std::complex<double>;

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
    int num_vars = assembler.get_num_vars();
    auto h_vars = assembler.createVarsHostVec();
    auto p_vars = assembler.createVarsHostVec();
    auto p_vars2 = assembler.createVarsHostVec();
    auto res = assembler.createVarsVec();

    // fixed perturbations of the host and pert vars
    double h = 1e-30;
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
        p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
        h_vars[ivar] += T(0.0, p_vars[ivar].real() * h);
    }
    
    auto vars = convertVecType<T>(h_vars);
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
    double jac_TD = 0.0;
    for (int i = 0; i < 24; i++) {
      for (int j = 0; j < 24; j++) {
        jac_TD += (p_vars[i] * p_vars2[j] * h_mat[24*i+j]).real();
      }
    }

    T temp = A2D::VecDotCore<T,24>(p_vars2.getPtr(), h_res.getPtr());
    double jac2_TD = A2D::ImagPart(temp) / h;

    printf("Analytic Jacobian\n");
    printf("jac TD analytic = %.8e\n", jac_TD);
    printf("jac TD comp-step = %.8e\n", jac2_TD);

    // print residual
    // printf("res: ");
    // printVec<double>(num_vars, h_res.getPtr());
    
    // const double *h_mat_ptr = h_mat.getPtr();
    // for (int i = 0; i < 2; i++) {
    //     printf("kmat row %d: ", i);
    //     printVec<double>(num_vars, &h_mat_ptr[num_vars*i]);
    // }

    printf("took %d microseconds to run add jacobian\n", (int)duration.count());

    return 0;
};
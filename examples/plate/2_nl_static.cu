#include "_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

int main() {
    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 100; // 300
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR factorization
    double fillin = 10.0; // 10.0
    bool print = true;
    assembler.symbolic_factorization(fillin, print);

    // get the loads
    double Q = 1e6; // load magnitude
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    // T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // newton solve
    int num_load_factors = 100, num_newton = 30;
    T min_load_factor = 0.05, max_load_factor = 1.0, abs_tol = 1e-6,
        rel_tol = 1e-4;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/uCRM_";
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler>(
        solve_func, kmat, loads, soln, assembler, res, rhs, vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, print);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate_nl.vtk");
};
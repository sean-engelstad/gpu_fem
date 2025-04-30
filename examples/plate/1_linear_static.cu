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
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 3;
    // int nxe = 100;
    // int nxe = 300;
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    auto bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.compute_full_LU_pattern(fillin, print);

    printf("rowp:");
    printVec<int>(bsr_data.nnodes+1, bsr_data.rowp);
    printf("cols:");
    printVec<int>(bsr_data.nnzb, bsr_data.cols);
    printf("perm:");
    printVec<int>(bsr_data.nnodes, bsr_data.perm);
    printf("iperm:");
    printVec<int>(bsr_data.nnodes, bsr_data.iperm);
    printf("elem_ind_map:");
    printVec<int>(bsr_data.nelems * bsr_data.nodes_per_elem * bsr_data.nodes_per_elem, 
        bsr_data.elem_ind_map);

    assembler.moveBsrDataToDevice();

    printf("here1\n");

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    printf("here2\n");

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    printf("here3\n");

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    printf("here3.5\n");
    assembler.apply_bcs(res);
    printf("here3.6\n");
    assembler.apply_bcs(kmat);
    printf("here4\n");

    // solve the linear system
    CUSPARSE::direct_LU_solve(kmat, loads, soln);

    printf("here5\n");

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    printf("resid_norm = %.4e\n", resid_norm);

};
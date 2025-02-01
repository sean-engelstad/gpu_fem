#include "assembler.h"
#include "base/utils.h"
#include "linalg/linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "shell/shell.h"

int main() {
    using T = double;

    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("CRM_box_2nd.bdf");

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    double E = 70e9, nu = 0.3, thick = 0.005; // material & thick properties

    // make the assembler from the uCRM mesh
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
    
    // BSR factorization
    double fillin = 10.0; // 10.0
    bool print = true;
    assembler.symbolic_factorization(fillin, print);

    // // temp debug
    // return 1;

    // init variables u;
    auto vars = assembler.createVarsVec();
    assembler.set_variables(vars);

    // setup matrix & vecs
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat, print);
    
    // -----------------------------
    // return 1; // temp debug stop (to fix this method on uCRM)


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    assembler.apply_bcs(res, print);
    assembler.apply_bcs(kmat, print);

    // check kmat here
    // printVec<double>(24, kmat.getPtr());

    // set the rhs for this problem
    // TODO : what loads to apply to the problem?
    T *my_loads = nullptr;
    int nvars = assembler.get_num_vars();
    memset(my_loads, 0.0, nvars * sizeof(double));

    // set fz = 10.0 everywhere
    double load_mag = 10.0;
    for (int ivar = 0; ivar < nvars; ivar++) {
        int idof = ivar % 6;
        if (idof == 2) {
            my_loads[ivar] = load_mag;
        }
    }

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // now do cusparse solve on linear static analysis
    auto start2 = std::chrono::high_resolution_clock::now();
    CUSPARSE::direct_LU_solve_old<T>(kmat, loads, soln, print);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

    // compute total direc derivative of analytic residual

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    write_to_csv<double>(h_soln.getPtr(), h_soln.getSize(), "csv/plate_soln.csv");
};
#include "_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"
#include <iostream>

/**
 solve on CPU with cusparse for debugging
 **/

int main(void) {
    using T = double;

    std::ios::sync_with_stdio(false); // always flush print immediately

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 100;
    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // perform a factorization on the rowPtr, colPtr (before creating matrix)
    double fillin = 10.0; // 10.0
    bool print = true;
    assembler.symbolic_factorization(fillin, print);

    // init variables u;
    auto vars = assembler.createVarsVec();
    assembler.set_variables(vars);

    // setup matrix & vecs
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat, print);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    assembler.apply_bcs(res, print);
    assembler.apply_bcs(kmat, print);

    // set the rhs for this problem
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    // printf("my_loads: ");
    // printVec<T>(54, my_loads);

    // it's currently taking a really long time to make 
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads, true);

    // printout kmat before solve (since cusparse solve changes kmat lower triangular)
    // also printout stiffness matrix
    bool debug_num_elements = nxe <= 10;
    if (debug_num_elements) {
        auto h_kmat = kmat.createHostVec();
        write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(), "csv/plate_kmat.csv");
    }
    
    // now do cusparse solve on linear static analysis
    auto start2 = std::chrono::high_resolution_clock::now();
    CUSPARSE::direct_LU_solve_old<T>(kmat, loads, soln, true);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

    // compute total direc derivative of analytic residual

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    auto h_loads = loads.createHostVec();

    // printf("took %d microseconds to run add jacobian\n", (int)duration.count());
    // printf("took %d microseconds to run cusparse solve\n",
    //        (int)duration2.count());

    // write the solution to binary file so I can read it in in python
    // always write this one out regardless of size
    write_to_csv<double>(h_loads.getPtr(), h_loads.getSize(), "csv/plate_loads.csv");
    write_to_csv<double>(h_soln.getPtr(), h_soln.getSize(), "csv/plate_soln.csv");

    auto bsrData = kmat.getBsrData();
    DeviceVec<int> d_rowPtr(bsrData.nnodes + 1, bsrData.rowPtr);
    auto h_rowPtr = d_rowPtr.createHostVec();
    // printf("h_rowPtr: ");
    // printVec<int>(bsrData.nnodes, h_rowPtr.getPtr());
    // printf("\n");
    DeviceVec<int> d_colPtr(bsrData.nnzb, bsrData.colPtr);
    auto h_colPtr = d_colPtr.createHostVec();

    if (debug_num_elements) {
        write_to_csv<int>(h_rowPtr.getPtr(), h_rowPtr.getSize(), "csv/plate_rowPtr.csv");
        write_to_csv<int>(h_colPtr.getPtr(), h_colPtr.getSize(), "csv/plate_colPtr.csv");
    }
    
    
    delete[] my_loads;

    return 0;
};
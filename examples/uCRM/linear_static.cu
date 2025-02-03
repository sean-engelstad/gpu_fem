#include "assembler.h"
#include "base/utils.h"
#include "linalg/linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "shell/shell.h"
#include "mesh/vtk_writer.h"

int main() {
    using T = double;

    auto start0 = std::chrono::high_resolution_clock::now();

    // uCRM mesh files can be found at: https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    TACSMeshLoader<T> mesh_loader{};
    // mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
    mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");
    

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
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);

    // set fz = 10.0 everywhere
    double load_mag = 10.0;
    double *h_loads_ptr = h_loads.getPtr();
    for (int inode = 0; inode < nnodes; inode++) {
        h_loads_ptr[6*inode+2] = load_mag;
    }

    #ifndef USE_GPU
    auto loads = h_loads.createHostVec();
    #else 
    auto loads = h_loads.createDeviceVec();
    #endif

    

    // auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // now do cusparse solve on linear static analysis
    CUSPARSE::direct_LU_solve_old<T>(kmat, loads, soln, print);

    // compute total direc derivative of analytic residual

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "uCRM.vtk");

    auto stop0 = std::chrono::high_resolution_clock::now();
    auto duration0 =
        std::chrono::duration_cast<std::chrono::microseconds>(stop0 - start0);
    double my_sec = duration0.count()/1e6;
    printf("total time %.4e seconds\n", my_sec);
};
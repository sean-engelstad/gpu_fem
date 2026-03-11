// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"
#include "multigrid/utils/fea.h"
#include <string>
#include <chrono>

#include "include/fetidp_assembler.h"

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

int main(int argc, char **argv) {
    using T = double;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    using FETIDP = FetidpSolver<T, Assembler, VecType, BsrMat>;

    auto start0 = std::chrono::high_resolution_clock::now();
    int nxe = 4;
    int nye = nxe;
    double SR = 1e3; // slenderness = length / thickness
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye; // for now (should have 25 grids)
    printf("WARNING: FETI_DP currently only supports clamped conditions.");
    printf("Can support in future, just haven't coded yet (some DOF would be interface vs interior,");
    printf(" prob duplicate)\n");
    auto assembler = createPlateClampedAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, 
        nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    auto& bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern(); // because only need fillin on subdomains now
    assembler.moveBsrDataToDevice();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    auto fetidp = new FETIDP(cublasHandle, cusparseHandle, assembler, kmat);
    int nxs = 2, nys = 2;
    bool close_hoop = false; // off for plates, on for closed cylinders
    fetidp->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);

    // // get the loads
    // double Q = 1.0; // load magnitude
    // T *my_loads = getPlateLoads<T, Basis, Physics>(nxe, nye, Lx, Ly, Q);

    // auto loads = assembler.createVarsVec(my_loads);
    // assembler.apply_bcs(loads);

    // // setup kmat and initial vecs
    // auto soln = assembler.createVarsVec();
    // auto res = assembler.createVarsVec();
    // auto vars = assembler.createVarsVec();

    // // assemble the kmat
    // assembler.add_jacobian_fast(kmat);
    // assembler.apply_bcs(res);
    // assembler.apply_bcs(kmat);

    // auto end0 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> startup_time = end0 - start0;
    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto start1 = std::chrono::high_resolution_clock::now();

    // // solve the linear system
    // CUSPARSE::direct_LU_solve(kmat, loads, soln);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> solve_time = end1 - start1;
    // int nx = nxe + 1;
    // int ndof = nx * nx * 6;
    // double total = startup_time.count() + solve_time.count();
    // size_t bytes_per_double = sizeof(double);
    // double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    // printf("plate direct solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem (MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_mb);


    // // print some of the data of host residual
    // auto h_soln = soln.createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}
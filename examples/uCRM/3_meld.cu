
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "coupled/meld.h"


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

int main() {
    using T = double;

    auto start0 = std::chrono::high_resolution_clock::now();

    // uCRM mesh files can be found at: https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
    // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");
    

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

    double E = 70e9, nu = 0.3, thick = 0.005; // material & thick properties

    // make the assembler from the uCRM mesh
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // get the disps and loads
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);
    double load_mag = 10.0;
    double *h_loads_ptr = h_loads.getPtr();
    for (int inode = 0; inode < nnodes; inode++) {
        h_loads_ptr[6*inode+2] = load_mag;
    }
    auto fa = h_loads.createDeviceVec();
    assembler.apply_bcs(fa);

    // make the disps a copy of loads (same)
    auto temp = fa.createHostVec();
    auto us = temp.createDeviceVec();

    // get xcoords from the assembler and associated BDF
    auto xpts = assembler.getXpts();

    // make the transfer scheme object
    T beta = 10.0, Hreg = 1e-2;
    int nn = 32, sym = -1;
    auto meld = MELD<T>(xpts, xpts, beta, nn, sym, Hreg);
    meld.initialize();

    // transfer the displacements
    auto ua = meld.transferDisps(us);

    // writeout the transferred displacements
    auto h_ua = ua.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_ua, "uCRM_ua.vtk");

    // TODO: writeout the transferred loads
};
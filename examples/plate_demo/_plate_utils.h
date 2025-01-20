#pragma once
#include "assembler.h"
#include "linalg/linalg.h"
#include "math.h"
#include <fstream>
#include <vector>

template <class Assembler>
Assembler createPlateAssembler(int nxe, int nye, double Lx, double Ly, double E,
                               double nu, double thick) {

    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;

    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    - In the very thin-walled regime (low thick) becomes
    CPT or Kirchoff plate theory with no transverse shear effects
    - PDE for Kirchoff plate theory, linear static analysis
        D * nabla^4 w = q(x,y)
        w = 0, simply supported
    - if transverse loads q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
      [one half-wave each direction], then solution is:
        w(x,y) = A * sin(pi * x / a) * sin(pi * y / b)
        with A = Q / D / pi^4 / (1/a^4 + 1 / b^4 + 2 / a^2 b^2)

    - simply supported BCs are:
        on negative x2 edge: dof 23
        on negative x1 edge: dof 13
        on (0,0) corner : dof 123456
        on pos x2 edge: dof 3
        on pos x1 edge: dof 3
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;
    int num_elements = nxe * nye;

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // (0,0) corner with dof 123456
    for (int idof = 0; idof < 6; idof++) {
        my_bcs.push_back(idof);
    }
    // negative x2 edge with dof 23
    for (int ix = 1; ix < nnx; ix++) {
        int iy = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode + 1); // dof 2 for v
        my_bcs.push_back(6 * inode + 2); // dof 3 for w
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);
        my_bcs.push_back(6 * inode + 2);

        // pos x1 edge
        ix = nnx - 1;
        inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode + 2); // corresp dof 3 for w
    }
    // pos x2 edge
    for (int ix = 1; ix < nnx - 1; ix++) {
        int iy = nny - 1;
        int inode = nnx * iy + ix;
        // printf("new bc = %d\n", 6 * inode + 2);
        my_bcs.push_back(6 * inode + 2); // corresp dof 3 for w
    }

    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            // TODO : issue with defining conn out of order like this, needs to
            // be sorted now?""
            int nodes[] = {nnx * iye + ixe, nnx * iye + ixe + 1,
                           nnx * (iye + 1) + ixe + 1, nnx * (iye + 1) + ixe};
            for (int inode = 0; inode < Basis::num_nodes; inode++) {
                elem_conn[Basis::num_nodes * ielem + inode] = nodes[inode];
            }
        }
    }

    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the x-coordinates of the panel
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = Lx / nxe;
    T dy = Ly / nye;
    for (int iy = 0; iy < nny; iy++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * iy + ix;
            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            xpt_node[0] = dx * ix;
            xpt_node[1] = dy * iy;
            xpt_node[2] = 0.0;
        }
    }

    HostVec<Data> physData(num_elements, Data(E, nu, thick));

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn,
                        xpts, bcs, physData);

    return assembler;
}

template <typename T, class Phys>
T *getPlateLoads(int nxe, int nye, double Lx, double Ly, double load_mag) {

    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    make the load set for this mesh
    q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;

    T dx = Lx / nxe;
    T dy = Ly / nye;

    double PI = 3.1415926535897;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // technically we should be integrating this somehow or distributing this
    // among the elements somehow..
    // the actual rhs is integral q(x,y) * phi_i(x,y) dxdy, fix later if want
    // better error conv.
    for (int iy = 0; iy < nny; iy++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * iy + ix;
            T x = ix * dx, y = iy * dy;
            T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
            my_loads[Phys::vars_per_node * inode + 2] = nodal_load;
        }
    }
    return my_loads;
}

template <typename T, class Phys>
T *getPlatePointLoad(int nxe, int nye, double Lx, double Ly, double load_mag) {

    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    make the load set for this mesh
    q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;

    int ix = nnx / 2;
    int iy = nny / 2;
    int inode = nnx * iy + ix;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    my_loads[Phys::vars_per_node * inode + 2] = load_mag;
    return my_loads;
}

void write_to_binary(const double *array, size_t size,
                     const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::ios_base::failure("Failed to open file for writing");
    }
    out.write(reinterpret_cast<const char *>(array), size * sizeof(double));
    out.close();
}

template <typename T>
void write_to_csv(const T *array, size_t size, const std::string &filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::ios_base::failure("Failed to open file for writing");
    }
    for (size_t i = 0; i < size; ++i) {
        out << array[i];
        if (i != size - 1) {
            out << ",";
        }
    }
    out << "\n";
    out.close();
}
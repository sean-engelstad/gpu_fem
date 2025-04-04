#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "linalg/vec.h"

template <typename T>
T convertNanToZero(T value) {
    // return std::isnan(value) ? 0.0f : value;

    // temporarily keep nans
    return value;
}

template <typename T>
void printGridToVTK(int nnx, int nny, HostVec<T> &x0, HostVec<T> &u, std::string filename) {
    // NOTE : better to use F5 binary for large cases, we will handle that
    // later
    using namespace std;
    string sp = " ";
    string dataType = "double64";

    ofstream myfile;
    myfile.open(filename);
    myfile << "# vtk DataFile Version 2.0\n";
    myfile << "TACS GPU shell writer\n";
    myfile << "ASCII\n";

    // make an unstructured grid even though it is really structured
    myfile << "DATASET UNSTRUCTURED_GRID\n";
    int num_nodes = nnx * nny;
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    // print all the xpts coordinates
    double *xpts_ptr = x0.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // print all the cells
    int nelems = (nnx - 1) * (nny - 1);
    int nodes_per_elem = 4;
    int num_elem_nodes = nelems * (nodes_per_elem + 1);
    myfile << "CELLS " << nelems << " " << num_elem_nodes << "\n";

    int nxe = nnx - 1, nye = nny - 1;
    for (int iy = 0; iy < nye; iy++) {
        for (int ix = 0; ix < nxe; ix++) {
            int istart = iy * nnx + ix;
            myfile << sp << 4;
            myfile << sp << istart;
            myfile << sp << istart + 1;
            myfile << sp << istart + nnx + 1;
            myfile << sp << istart + nnx;
            myfile << "\n";
        }
    }

    // cell type 9 is for CQUAD4 basically
    myfile << "CELL_TYPES " << nelems << "\n";
    for (int ielem = 0; ielem < nelems; ielem++) {
        myfile << 9 << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << convertNanToZero(u[3 * inode]) << sp;
        myfile << convertNanToZero(u[3 * inode + 1]) << sp;
        myfile << convertNanToZero(u[3 * inode + 2]) << "\n";
    }

    myfile.close();
}

template <typename T>
HostVec<T> makeGridMesh(int nnx, int nny, T Lx, T Ly, T z0) {
    int N = nnx * nny;
    HostVec<T> x0(3 * N);
    T pi = 3.14159265358979323846;
    printf("z0 = %.4e\n", z0);

    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);
    for (int iy = 0; iy < nny; iy++) {
        T eta = iy * 1.0 / (nny - 1);
        T yfac = sin(pi * eta);
        for (int ix = 0; ix < nnx; ix++) {
            T xi = ix * 1.0 / (nnx - 1);
            T xfac = sin(pi * xi);

            int ind = iy * nnx + ix;
            x0[3 * ind] = ix * dx;
            x0[3 * ind + 1] = iy * dy;
            x0[3 * ind + 2] = z0 * xfac * yfac;
            // printf("ix %d xi %.4e, iy %d eta %.4e\n", ix, xi, iy, eta);
            // printf("zval = %.4e, xfac %.4e, yfac %.4e\n", x0[3*ind+2], xfac, yfac);
        }
    }

    return x0;
}

template <typename T>
HostVec<T> makeInPlaneShearDisp(HostVec<T> &x0, T angleDeg) {
    int N = x0.getSize() / 3;
    HostVec<T> u(3 * N);
    T angleRad = angleDeg * 3.14159265 / 180.0;

    for (int inode = 0; inode < N; inode++) {
        T *xpt = &x0[3 * inode];
        T *upt = &u[3 * inode];
        upt[0] = tan(angleRad / 2.0) * xpt[1];
        upt[1] = tan(angleRad / 2.0) * xpt[0];
        upt[2] = 0.0;
    }

    return u;
}

template <typename T>
HostVec<T> makeCustomDisp(HostVec<T> &x0, T scale) {
    int N = x0.getSize() / 3;
    HostVec<T> u(3 * N);
    T pi = 3.14159265;

    for (int inode = 0; inode < N; inode++) {
        T *xpt = &x0[3 * inode];
        T *upt = &u[3 * inode];
        upt[0] = sin(4 * pi * xpt[0]) * cos(6 * pi * xpt[1]) + 0.5 * cos(7 * pi * xpt[0] * xpt[1]);
        upt[1] = cos(5 * pi * xpt[0]) * sin(4 * pi * xpt[1]) + 0.3 * sin(6 * pi * xpt[0] * xpt[1]);
        upt[2] = sin(3 * pi * xpt[0]) * cos(5 * pi * xpt[1]) + 0.4 * cos(6 * pi * xpt[0] * xpt[1]);
        for (int i = 0; i < 3; i++) {
            upt[i] *= scale;
        }
    }

    return u;
}

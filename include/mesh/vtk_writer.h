#pragma once
#include "assembler.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

template <class Assembler, class Vec>
void printToVTK(Assembler assembler, Vec soln, std::string filename) {
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
    int num_nodes = assembler.get_num_nodes();
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    // print all the xpts coordinates
    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();

    double *xpts_ptr = h_xpts.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2]
               << "\n";
    }

    // print all the cells
    int num_elems = assembler.get_num_elements();
    int nodes_per_elem = Assembler::vars_nodes_per_elem;
    int num_elem_nodes = num_elems * (nodes_per_elem + 1); // repeats here
    myfile << "CELLS " << num_elems << " " << num_elem_nodes << "\n";

    auto d_vars_conn = assembler.getConn();
    auto h_vars_conn = d_vars_conn.createHostVec();
    int *conn_ptr = h_vars_conn.getPtr();

    const int32_t local_perm[4] = {0, 1, 3, 2};
    for (int ielem = 0; ielem < num_elems; ielem++) {
        const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
        myfile << nodes_per_elem;
        for (int inode = 0; inode < nodes_per_elem; inode++) {
            myfile << sp << elem_conn[local_perm[inode]];
        }
        myfile << "\n";
    }

    // cell type 9 is for CQUAD4 basically
    myfile << "CELL_TYPES " << num_elems << "\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << 9 << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[6 * inode] << sp;
        myfile << soln[6 * inode + 1] << sp;
        myfile << soln[6 * inode + 2] << "\n";
    }

    myfile.close();
}
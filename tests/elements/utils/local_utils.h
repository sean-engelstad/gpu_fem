#pragma once
#include "assembler.h"
#include "linalg/_linalg.h"

template <class Assembler>
Assembler createOneElementAssembler(int num_bcs) {
    int num_elements = 1;
    int num_geo_nodes = 4;
    int num_vars_nodes = 4;

    HostVec<int> bcs(num_bcs);
    bcs.randomize(num_vars_nodes);

    // need pointer here otherwise goes out of scope
    int32_t *_geo_conn = new int32_t[4]{0, 1, 2, 3};
    HostVec<int32_t> geo_conn(4, _geo_conn);

    int32_t *_vars_conn = new int32_t[4]{0, 1, 2, 3};
    HostVec<int32_t> vars_conn(4, _vars_conn);

    // set the xpts randomly for this example
    using Geo = typename Assembler::Geo;
    using T = typename Assembler::T;
    int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
    HostVec<T> xpts(num_xpts);
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
        xpts[ixpt] = 1.0345452 + 2.23123432 * ixpt + 0.323 * ixpt * ixpt;
    }

    double E = 70e9, nu = 0.3, t = 0.005;  // aluminum plate
    using Data = typename Assembler::Data;
    HostVec<Data> physData(num_elements, Data(E, nu, t));

    // make the assembler
    Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData);

    // so that the bsr_data is on device (otherwise data on host)
    assembler.moveBsrDataToDevice();

    return assembler;
}
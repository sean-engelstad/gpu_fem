#include "analysis.h"

int main(void) {
    using T = double;

    using Quad = TriangleQuadrature<T>;
    using Geo = LinearTriangleGeo<T,Quad>;
    using Basis = QuadraticTriangleBasis<T,Quad>;
    using Physics = PlaneStressPhysics<T,Quad>;
    using Group = ElementGroup<T, Geo, Basis, Physics>;
    using Assembler = ElementAssembler<T, Group>;

    int num_nodes = 500;
    int num_elements = 1000;
    Assembler assembler(num_nodes, num_elements);

    int ndof = Assembler::vars_per_node * num_nodes;
    T *res = new T[ndof];
    assembler.add_residual(res);

    return 0;
}
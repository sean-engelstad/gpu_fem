#include "analysis.h"

int main(void) {
    using T = double;

    const A2D::GreenStrainType strain = A2D::GreenStrainType::LINEAR;

    using Quad = TriangleQuadrature<T>;
    using Geo = LinearTriangleGeo<T,Quad>;
    using Basis = QuadraticTriangleBasis<T,Quad>;
    using Physics = PlaneStressPhysics<T,Quad,strain>;
    using Group = ElementGroup<T, Geo, Basis, Physics>;
    using Data = typename Physics::IsotropicData;
    using Assembler = ElementAssembler<T, Group>;

    int num_nodes = 500;
    int num_elements = 1000;

    // initialize ElemData
    double E = 70e9, nu = 0.3, t = 0.005; // aluminum plate
    Data elemData[num_elements];
    for (int ielem = 0; ielem < num_elements; ielem++) {
        elemData[ielem] = Data(E, nu, t);
    }

    // make the assembler
    Assembler assembler(num_nodes, num_elements, elemData);

    int ndof = Assembler::vars_per_node * num_nodes;
    T *res = new T[ndof];
    assembler.add_residual(res);

    return 0;
}
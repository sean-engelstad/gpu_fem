#include "assembler.h"
#include "shell/shell.h"
#include "chrono"
#include "a2dcore.h"

// get residual directional derivative with complex-step of energy method

int main(void) {
    using T = A2D_complex_t<double>;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T,has_ref_axis>;

    constexpr bool isNonlinear = false;
    using Physics = IsotropicShell<T, Data, isNonlinear>;
    
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>; 
    using Assembler = ElementAssembler<T, ElemGroup>;

    // printf("running!\n");

    int num_elements = 1;
    int num_geo_nodes = 4;
    int num_vars_nodes = 4;

    // make fake element connectivity for testing
    int N = Geo::num_nodes * num_elements;
    int32_t geo_conn[] = { 0, 1, 2, 3 };

    // randomly generate the connectivity for the variables / basis
    int N2 = Basis::num_nodes * num_elements;
    int32_t vars_conn[] = { 0, 1, 2, 3 };

    // set the xpts randomly for this example
    int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
    T *xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
      xpts[ixpt] = 1.0345452 + 2.23123432 * ixpt + 0.323 * ixpt * ixpt;
    }

    // initialize ElemData
    double E = 70e9, nu = 0.3, t = 0.005; // aluminum plate
    Data elemData[num_elements];
    for (int ielem = 0; ielem < num_elements; ielem++) {
        elemData[ielem] = Data(E, nu, t);
    }

    // make the assembler
    Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, elemData);

    // define variables here for testing different vars inputs
    // set some host data to zero
    int32_t num_vars = assembler.get_num_vars();
    T *h_vars = new T[num_vars];
    memset(h_vars, 0.0, num_vars * sizeof(T));

    bool nz_vars = true;
    if (nz_vars) {
      for (int ivar = 0; ivar < num_vars; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
      }
    }

    // perturbation vector for directional derivative
    double *p_vars = new double[num_vars];
    memset(p_vars, 0.0, num_vars * sizeof(double));
    for (int ivar = 0; ivar < num_vars; ivar++) {
      p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    }

    // complex step perturbations of vars by p_vars * h
    double h = 1e-30;
    for (int ivar = 0; ivar < num_vars; ivar++) {
      h_vars[ivar] = h_vars[ivar] + T(0.0, p_vars[ivar] * h);
    }

    // for (int i = 0; i < 24; i++) {
    //   printf("h_vars[%d] = %.8e\n", i, A2D::ImagPart(h_vars[i]));
    // }

    // set perturbed variables in
    #ifdef USE_GPU
    T *d_vars;
    cudaMalloc((void**)&d_vars, num_vars * sizeof(T));
    cudaMemcpy(d_vars, h_vars, num_vars * sizeof(T), cudaMemcpyHostToDevice);   
    assembler.set_variables(d_vars);
    #else // USE_GPU
    assembler.set_variables(h_vars);
    #endif

    // define the residual vector (host or device)
    T *h_residual = new T[num_vars];
    memset(h_residual, 0.0, num_vars * sizeof(T));
    #ifdef USE_GPU
    T *d_residual;
    cudaMalloc((void**)&d_residual, num_vars * sizeof(T));
    cudaMemset(d_residual, 0.0, num_vars * sizeof(T));     
    #endif

    T Uenergy = 0.0;

    // time energy method
    auto start = std::chrono::high_resolution_clock::now();

    assembler.add_energy(Uenergy);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute residual total direc derivative of complex-step method of energy
    double res_TD = A2D::ImagPart(Uenergy) / h;
    printf("Complex step residual\n");
    printf("res TD = %.8e\n", res_TD);

    printf("took %d microseconds to run add residual\n", (int)duration.count());

    // print data of strain energy
    printf("Uenergy = %.8e, %.8e\n", A2D::RealPart(Uenergy), A2D::ImagPart(Uenergy));

    return 0;
};
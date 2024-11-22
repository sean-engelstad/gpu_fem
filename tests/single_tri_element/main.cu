#include "analysis.h"

int main(void) {
  using T = float;

  // this GPU can't do atomic add with double due to older compute capability
  // using T = double;

  const A2D::GreenStrainType strain = A2D::GreenStrainType::LINEAR;

  using Quad = TriangleQuadrature<T>;
  using Geo = LinearTriangleGeo<T, Quad>;
  using Basis = QuadraticTriangleBasis<T, Quad>;
  using Physics = PlaneStressPhysics<T, Quad, strain>;
  using Group = ElementGroup<T, Geo, Basis, Physics>;
  using Data = typename Physics::IsotropicData;
  using Assembler = ElementAssembler<T, Group>;

  int num_geo_nodes = 3;
  int num_vars_nodes = 6;
  int num_elements = 1;

  // simple test for a singular triangular element
  // of a mesh with:
  //      node 0 - (0,0)
  //      node 1 - (2,1)
  //      node 2 - (1,2)

  // make fake element connectivity for testing
  int N = Geo::num_nodes * num_elements;
  int32_t geo_conn[N] = {0, 1, 2};

  // randomly generate the connectivity for the variables / basis
  int N2 = Basis::num_nodes * num_elements;
  int32_t vars_conn[N2] = {0, 1, 2, 3, 4, 5};

  // set the xpts randomly for this example
  int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
  T xpts[num_xpts] = {0, 0, 2, 1, 1, 2};

  // initialize ElemData
  double E = 70e9, nu = 0.3, t = 0.005;  // aluminum plate
  Data elemData[num_elements];
  for (int ielem = 0; ielem < num_elements; ielem++) {
    elemData[ielem] = Data(E, nu, t);
  }

  // make the assembler
  Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn,
                      vars_conn, xpts, elemData);

  // disp field is u = (x+y)*alpha, v = (x-y)*alpha
  int32_t num_vars = assembler.get_num_vars();
  T h_vars[num_vars] = {0.0, 0.0, 3, 1, 3, -1, 3, 0, 1.5, -0.5, 1.5, 0.5};

  // scale by alpha later
  T alpha = 1.0;
  for (int idof = 0; idof < num_vars; idof++) {
    h_vars[idof] *= alpha;
  }

#ifdef USE_GPU
  T *d_vars;
  cudaMalloc((void **)&d_vars, num_vars * sizeof(T));
  cudaMemcpy(d_vars, h_vars, num_vars * sizeof(T), cudaMemcpyHostToDevice);
  assembler.set_variables(d_vars);
#else  // USE_GPU
  assembler.set_variables(h_vars);
#endif

  // define the residual vector (host or device)
  T *h_residual = new T[num_vars];
  memset(h_residual, 0.0, num_vars * sizeof(T));
#ifdef USE_GPU
  T *d_residual;
  cudaMalloc((void **)&d_residual, num_vars * sizeof(T));
  cudaMemset(d_residual, 0.0, num_vars * sizeof(T));
#endif

  int num_vars2 = num_vars * num_vars;
  T *h_mat = new T[num_vars2];
  memset(h_mat, 0.0, num_vars2 * sizeof(T));
#ifdef USE_GPU
  T *d_mat;
  cudaMalloc((void **)&d_mat, num_vars2 * sizeof(T));
  cudaMemset(d_mat, 0.0, num_vars2 * sizeof(T));
#endif

  // #ifdef USE_GPU
  //   assembler.add_residual(d_residual);
  //   cudaMemcpy(h_residual, d_residual, num_vars * sizeof(T),
  //              cudaMemcpyDeviceToHost);
  // #else
  //   assembler.add_residual(h_residual);
  // #endif

// call add jacobian
#ifdef USE_GPU
  assembler.add_jacobian(d_residual, d_mat);
  cudaMemcpy(h_residual, d_residual, num_vars * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mat, d_mat, num_vars2 * sizeof(T), cudaMemcpyDeviceToHost);
#else
  assembler.add_jacobian(h_residual, h_mat);
#endif

  printf("done with script\n");

  // print data of host residual
  for (int i = 0; i < num_vars; i++) {
    printf("res[%d] = %.8e\n", i, h_residual[i]);
  }

  for (int i = 0; i < num_vars; i++) {
    for (int j = 0; j < num_vars; j++) {
      printf("mat[%d,%d] = %.8e\n", i, j, h_mat[num_vars * i + j]);
    }
  }

  return 0;
};

// TODO : would like to just rewrite the full assembler later, let's do less intrusive wrappers for
// assembler from BDF, plate, cylinder or other geometries like what I did in the python for mitc
// shell..

// #include <cusparse_v2.h>
// #include <thrust/device_vector.h>

// #include "cublas_v2.h"
// #include "cuda_utils.h"
// #include "linalg/vec.h"
// #include "element.cuh"

// #include "element/shell/shell_elem_group.h"
// #include "element/shell/shell_elem_group.cuh"

// class ShellMultigridAssembler {
//     // lighter weight assembler for geometric multigrid (single grid here)
//     // TODO : would like to rewrite this from scratch again.. so more parallel, NZ pattern on GPU
//   public:

//     using T = double;
//     using Quad = QuadLinearQuadrature<T>;
//     using Basis = ShellQuadBasis<T, Quad, 2>;
//     using Geo = Basis::Geo;
//     using Data = ShellIsotropicData<T, false>;
//     using Physics = IsotropicShell<T, Data, false>;
//     using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;

//     static constexpr int xpts_per_node = 3;
//     static constexpr int vars_per_node = 6;

//     ShellMultigridAssembler(int num_nodes_, int num_elements_, DeviceVec<int> d_elem_conn,
//         DeviceVec<int> d_bcs, DeviceVec<T> d_xpts, DeviceVec<Data> d_data) :
//         num_nodes(num_nodes_), num_elements(num_elements_) {

//         elem_conn = d_elem_conn;
//         bcs = d_bcs;
//         xpts = d_xpts;
//         data = d_data;

//         num_dof = num_nodes * vars_per_node;

//         // zero vars
//         vars = DeviceVec<T>(num_dof);

//         // construct initial bsr data
//         h_bsr_data = BsrData(num_elements, num_nodes, 4, 6, elem_conn.getPtr());
//         d_bsr_data = h_bsr_data.createDeviceBsrData();
//     }

//     void compute_nofill_pattern() {
//         // compute nofill pattern (with desired ordering as well..) for coloring

//     }

//     void assembleJacobian() {

//         // assemble the Kmat
//         dim3 block(1, 24, 4); // (1 elems_per_block, 24 DOF per elem, 4 quadpts per elem)
//         int nblocks = (num_elements + block.x - 1) / block.x;
//         dim3 grid(nblocks);

//         add_jacobian_gpu<T, ElemGroup, Data, 1, DeviceVec, BsrMat><<<grid, block>>>(
//             num_nodes, num_elements, elem_conn, xpts, vars, physData, res, K_mat);

//         CHECK_CUDA(cudaDeviceSynchronize());
//     }

//     // public data
//     int num_nodes, num_dof, num_elements;
//     DeviceVec<int> bcs, elem_conn;
//     DeviceVec<T> xpts, vars;
//     DeviceVec<Data> data;
//     BsrData bsr_data;
//     BsrMat K_mat, Dinv_mat;
// };

// try running two kernels at once in different streams
// cudaStream_t stream1, stream2;
// cudaStreamCreate(&stream1);
// cudaStreamCreate(&stream2);
// add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec, 1>  // drill strains
//     <<<grid, block, 0, stream1>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData,
//     res);
// add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec, 2>  // bending + tying strains
//     <<<grid, block, 0, stream2>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData,
//     res);
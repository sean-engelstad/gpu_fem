// compute  residual
void add_residual_cpu(T* res) {

    // Needs to be shared in GPU
    T elem_res[ndof_per_node * nelems];

    for (int e = 0; e < num_elements_in_group; e++ ) {
        for (int q = 0; q < num_quads; q++) {
            // Loop-body
            loop_body(e, q);
        }
    }
    // Populate the global residual vector

}


// compute  residual
__global__ void add_residual_gpu(T* res) {

    // Needs to be shared in GPU
     __shared__ T elem_res[ndof_per_node * nelems];

    e = thread_idx % num_quads;
    q = thread_idx / num_quads;`
    loop_body(e, q);

    // Populate the global residual vector via atomic add

}

// compute  residual
template <int32_t elems_per_block>
__global__ void add_residual_unified(T* res) {

    // Needs to be shared in GPU
    __shared__ T elem_res[elems_per_block][ndof_per_node * nelems];

    int elem_stride, q_stride;
    #ifdef USE_GPU
        elem_stride = num_elements_in_group;
        q_stride = num_quads;
    #else
        elem_stride = 1;
        q_stride = 1;
    #endif

    // does having the for loop slow down the GPU version even if only iterate one time?
    // can we do some constexpr stuff for the stride?
    // looks nicer w/o double for loop for GPU version (readability) => maybe we just use separate methods then
    for (int e = threadIdx.x + blockDim.x * blockIdx.x; e < num_elements_in_group; e += elem_stride ) {
        for (int q = threadIdx.y; q < num_quads; q+= q_stride) {
            // Loop-body
            loop_body(e, q);
        }
    }

    // Populate the global residual vector via atomic add

}
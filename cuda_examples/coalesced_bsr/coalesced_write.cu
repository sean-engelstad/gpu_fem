#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

template <typename T>
__device__ __host__ void printVec(const int N, const T *vec);

template <>
__device__ __host__ void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template <>
__device__ __host__ void printVec<float>(const int N, const float *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", (int)vec[i]);
    }
    printf("\n");
}

template <>
__device__ __host__ void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", (int)vec[i]);
    }
    printf("\n");
}

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

template <typename T>
__global__ void coalesced_bsr_kernel_1block(T *global_out, bool print) {
    /* this first kernel handles only one nodal block (and only goes up to 31 values not full 36, still illustrates general idea) */

    // see PR https://github.com/smdogroup/gpu_fem/pull/52 for more details and an explanation
    int iquad = threadIdx.x;
    int ideriv = threadIdx.y;
    int ithread = 4 * ideriv + iquad;

    T elem_mat_col[24];
    memset(elem_mat_col, 0.0, 24 * sizeof(T));
    // write nz in only one element block col (zero elsewhere so we know only values from block col 0 should be used)
    for (int i = 0; i < 6; i++) {
        elem_mat_col[i] = 6 * ideriv + i;
    }

    // here are the maps for the first 31 values, but we can more efficiently combine some of the maps into one and use uint32_t later
    // threads                    {  0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31 }
    // constexpr bool loc_val_mask[] = {1,1,1,1, 0,0,1,1, 0,0,0,0,   1, 1, 1, 1,  0, 0, 1, 1,  0, 0, 0, 0,  1, 1, 1, 1,  0, 0, 1, 1  };
    // constexpr int8_t src_lanes1[] = {0,1,2,3, 0,0,0,0, 0,0,0,0,   8, 9, 10,11, 0, 0, 14,15, 0, 0, 0, 0,  16,17,18,19, 16,17,0, 0  };
    // constexpr int8_t src_lanes2[] = {0,0,0,0, 0,1,6,7, 4,5,6,7,   0, 0, 0, 0,  8, 9, 0, 0,  12,13,14,15, 0, 0, 0, 0,  0, 0, 22,23 };

    // just use same src_lanes but then you will later figure out which value to choose
    constexpr uint32_t val_mask_bits =
    0b11001111000011001111000011001111;
    bool use_val1 = (val_mask_bits >> ithread) & 1;
    constexpr int8_t src_lanes[] = {0,1,2,3,  0,1,6,7,  4,5,6,7,  8,9,10,11,  8,9,14,15,  12,13,14,15,  16,17,18,19, 16,17,22,23 };

    // we use offset 2 every other row to make sure we have the right values to send..
    bool odd_row = (ideriv)%2;
    int8_t ind = (iquad - 2 * odd_row) % 24;

    // each thread holds two values to communicate to other warps (since some threads have to send two values like thread 8 to threads 12 and 16)
    T loc_val = elem_mat_col[ind];
    T orig_loc_val = loc_val; // just for printing map
    int8_t src_lane = src_lanes[ithread];
    loc_val = __shfl_sync(0xFFFFFFFF, loc_val, src_lane);
    
    T off_loc_val = elem_mat_col[4 + ind];
    T orig_off_loc_val = off_loc_val;
    off_loc_val = __shfl_sync(0xFFFFFFFF, off_loc_val, src_lane);

    // overall value with GPU friendly ternary operator
    T val = use_val1 ? loc_val : off_loc_val;
    // if (print) printf("ithread %d, loc_val %d, off_loc_val %d, mask %d, val %d\n", ithread, orig_loc_val, orig_off_loc_val, use_val1, val);

    // now write into the global_out
    int nelem = blockIdx.x;
    global_out[32 * nelem + ithread] = val; // in this case val == ithread for illustrating the process, but not true in FEM case (it's a stiffness value)
}

template <typename T>
__global__ void coalesced_bsr_kernel_4blocks(T *global_out, bool print) {
    /* this first kernel handles only one nodal block (and only goes up to 31 values not full 36, still illustrates general idea) */

    int ielem = blockIdx.x;
    T *global_loc = &global_out[36 * 4 * ielem];

    // see PR https://github.com/smdogroup/gpu_fem/pull/52 for more details and an explanation
    int iquad = threadIdx.x;
    int ideriv = threadIdx.y;
    int warp_row = ideriv % 8;
    int ithread = 4 * ideriv + iquad;
    int iwarp = ithread / 32;
    int ithread_warp = ithread - 32 * iwarp;

    T elem_mat_col[24];
    memset(elem_mat_col, 0.0, 24 * sizeof(T));
    // write nz in only one element block col (zero elsewhere so we know only values from block col 0 should be used)
    for (int i = 0; i < 6; i++) {
        elem_mat_col[i] = 6 * ideriv + i;
    }

    /*
        FIRST coalesced write..
    */

    // just use same src_lanes but then you will later figure out which value to choose
    // could write these maps ourselves for different configurations of num quadpts, etc.
    constexpr uint32_t val_mask_bits =
    0b11001111000011001111000011001111;
    bool use_val1 = (val_mask_bits >> ithread_warp) & 1;
    constexpr int8_t src_lanes[] = {0,1,2,3,  0,1,6,7,  4,5,6,7,  8,9,10,11,  8,9,14,15,  12,13,14,15,  16,17,18,19, 16,17,22,23 };

    // get the row within a warp
    

    // we use offset 2 every other row to make sure we have the right values to send..
    bool odd_row = (warp_row)%2;
    int8_t ind = (iquad - 2 * odd_row) % 24;

    // each thread holds two values to communicate to other warps (since some threads have to send two values like thread 8 to threads 12 and 16)
    T loc_val = elem_mat_col[ind];
    // int orig_loc_val = loc_val; // just for printing map
    int8_t src_lane = src_lanes[ithread_warp];
    loc_val = __shfl_sync(0xffffffff, loc_val, src_lane);
    
    T off_loc_val = elem_mat_col[4 + ind];
    // int orig_off_loc_val = off_loc_val;
    off_loc_val = __shfl_sync(0xffffffff, off_loc_val, src_lane);

    // overall value with GPU friendly ternary operator
    T val = use_val1 ? loc_val : off_loc_val;
    // if (print) printf("ithread %d, loc_val %d, off_loc_val %d, mask %d, val %d\n", ithread, orig_loc_val, orig_off_loc_val, use_val1, val);

    // now write into the global_out (this part works right now)
    // printf("thread %d, in warp thread %d, writing to location %d\n", ithread, ithread_warp, 48 * iwarp + ithread_warp);
    int nwarp_rows = 32 / blockDim.x;
    int nvals_per_warp = nwarp_rows * 6; // 6 = dof_per_node
    global_loc[nvals_per_warp * iwarp + ithread_warp] = val; // in this case val == ithread for illustrating the process, but not true in FEM case (it's a stiffness value)

    // check coalesced write by printout here
    // if (print) printf("thread %d writing to %ld\n", threadIdx.y * blockDim.x + threadIdx.x, &global_out[nvals_per_warp * iwarp + ithread_warp] - global_out);


    /*
        SECOND coalesced write.. TBD
    */
    loc_val = elem_mat_col[ind];
    off_loc_val = elem_mat_col[4 + ind];

    // for print (just want to see the remaining vals >= 32 now as 0-31 format)
    int loc_val_print = loc_val < 32 ? 0 : loc_val - 32;
    int off_loc_val_print = off_loc_val < 32 ? 0 : off_loc_val - 32;
    // if (iwarp == 0) printf("thread %d, loc_val %d, off_loc_val %d\n", ithread_warp, loc_val_print, off_loc_val_print);

    // new map (only use up to 15 of the values, after that can just be 0)
    // loc_mask =                   {0, 0, 0, 0,   1, 1, 1, 1,   0, 0, 1, 1,  0, 0, 0, 0 , etc. };
    constexpr int8_t src_lanes2[] = {20,21,22,23,  24,25,26,27,  24,25,30,31, 28,29,30,31 }; // use mod 16
    constexpr uint32_t val_mask_bits2 =
    0b0000110011110000;
    int ithread_hwarp = ithread_warp % 16;
    bool use_val2 = (val_mask_bits2 >> ithread_hwarp) & 1;

    // now communicate across warps
    int8_t src_lane2 = src_lanes2[ithread_hwarp];
    loc_val = __shfl_sync(0xffffffff, loc_val, src_lane2);
    off_loc_val = __shfl_sync(0xffffffff, off_loc_val, src_lane2);
    val = use_val2 ? loc_val : off_loc_val;
    
    if (32 + ithread_warp < 48) 
        global_loc[nvals_per_warp * iwarp + 32 + ithread_warp] = val;
}

int main() {
    /* 
    1-BLOCK: first test a single block write 
    */

    using T = float;
    // using T = double;

    // global out for 36 values in a single bsr nodal block (6x6)
    T *global_out;
    int nelems = 10000;
    CHECK_CUDA(cudaMalloc((void **)&global_out, nelems * 32 * sizeof(T)));
    CHECK_CUDA(cudaMemset(global_out, 0.0, 32 * nelems * sizeof(T)));

    // 8 deriv columns make up a single warp, 4 quadpts
    dim3 block(4,8,1);    
    dim3 grid(nelems);
    bool print = false; // done with this one so false
    coalesced_bsr_kernel_1block<<<grid, block>>>(global_out, print);
    CHECK_CUDA(cudaDeviceSynchronize());

    // write out the result (to check coalesced)
    T *h_out = new T[36];
    CHECK_CUDA(cudaMemcpy(h_out, global_out, 36 * sizeof(T), cudaMemcpyDeviceToHost));
    printf("coalesced 1 block out: ");
    printVec<T>(36, h_out);

    /* 
    4 BLOCKS:  (for a whole elem block column of a shell element) 
    */
    
    T *global_out2;
    CHECK_CUDA(cudaMalloc((void **)&global_out2, nelems * 4 * 36 * sizeof(T)));
    CHECK_CUDA(cudaMemset(global_out2, 0.0, nelems * 4 * 36 * sizeof(T)));

    dim3 block2(4,24,1);    
    dim3 grid2(nelems);
    print = true;
    coalesced_bsr_kernel_4blocks<<<grid2, block2>>>(global_out2, print);
    CHECK_CUDA(cudaDeviceSynchronize());

    // write out the result (to check coalesced)
    T *h_out2 = new T[4 * 36]; // 144 total values
    CHECK_CUDA(cudaMemcpy(h_out2, global_out2, 4 * 36 * sizeof(T), cudaMemcpyDeviceToHost));
    printf("coalesced 4 blocks out: ");
    printVec<T>(4 * 36, h_out2);
}


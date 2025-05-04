#include "../test_commons.h"

int main() {
    bool print = false;
    
    const int N = 6;
    double vec[N] = {0, 1, 2, 3, 4, 5};
    int perm[N] = {4, 3, 1, 0, 2, 5};
    int iperm[N] = {3, 2, 4, 1, 0, 5}; // can check this, it's correct

    int *d_perm, *d_iperm;
    CHECK_CUDA(cudaMalloc((void **)&d_perm, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_iperm, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_perm, perm, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iperm, iperm, N * sizeof(int), cudaMemcpyHostToDevice));

    // orig vec
    auto h_vec = HostVec<double>(N, vec);
    auto d_vec = h_vec.createDeviceVec();

    /**
    Our convention is the following:
    iperm : old entry => new entry
    perm  : new entry => old entry

    so if we apply iperm first to the 0 to N-1 list, 
    suppose entry 0 goes to 3 like in the current iperm
    then now entry 3 has 0 in it, and our iperm_vec has the inverse map of iperm aka perm

    iperm maps old to new entries (from 0 to N-1 vec)
    now new entries contain the old entries (aka this is the perm vec)
    perm vec maps new to old entries and will return to 0 to N-1 map
    */

    // permute data with forwards perm
    int block_dim = 1;
    d_vec.permuteData(block_dim, d_iperm);
    auto iperm_vec = d_vec.createHostVec();

    // now iperm vec again
    d_vec.permuteData(block_dim, d_perm);
    auto vec2 = d_vec.createHostVec();
    
    // check vec2 matches original vec and that perm_vec == perm
    int iperm_vec_i[N], vec_i[N], vec2_i[N];
    for (int i = 0; i < N; i++) {
        iperm_vec_i[i] = (int)iperm_vec[i];
        vec_i[i] = (int)vec[i];
        vec2_i[i] = (int)vec2[i];
    }

    if (print) {
        printf("vec:");
        printVec<int>(N, vec_i);
        printf("iperm{vec} => perm:");
        printVec<int>(N, iperm_vec_i);
        printf("vec2:");
        printVec<int>(N, vec2_i);
    }

    int abs_err_perm = abs_err<int>(N, iperm_vec_i, perm);
    int abs_err_final = abs_err<int>(N, vec2_i, vec_i);
    bool passed = abs_err_final == 0 && abs_err_perm == 0;
    double t_abs_err = max(abs_err_perm, abs_err_final);
    printTestReport("permute vector", passed, t_abs_err);
}
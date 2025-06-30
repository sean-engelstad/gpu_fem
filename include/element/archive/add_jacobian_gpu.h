/* memory writes archive */
static constexpr int write_case = 1;
if constexpr (write_case == 1) {
    // warp reduction over quadpts for jac
    int lane = local_thread % 32;
    int group_start = (lane / 4) * 4;
    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_mat_col[idof];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        // warp broadcast
        lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
        local_mat_col[idof] = lane_val;
    }

    int elem_block_row = ideriv / Phys::vars_per_node;
    int elem_inner_row = ideriv % Phys::vars_per_node;
    mat.addElementMatRow(active_thread, elem_block_row, elem_inner_row, global_elem, iquad,
                         Quadrature::num_quad_pts, Phys::vars_per_node, Basis::num_nodes,
                         vars_elem_conn, local_mat_col);

    // if (iquad == 0) {
    //     int nderiv = blockDim.y;
    //     int elem_block_row = ideriv / Phys::vars_per_node;
    //     int elem_inner_row = ideriv % Phys::vars_per_node;
    //     mat.addElementMatRow(active_thread, elem_block_row, elem_inner_row, global_elem, ideriv,
    //     nderiv,
    //         Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);
    // }
} else if constexpr (write_case == 2) {
    // first warp reduction and shfl sync
    int warp_ind = local_thread / 32;
    int warp_size = 32;
    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    int num_warps = num_threads / warp_size;
    int local_warp_ind = local_thread % warp_size;
    int group_start = (warp_ind / 4) * 4;

    __shared__ T block_col_buffer[4][36];

    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_mat_col[idof];
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        // warp broadcast
        lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
        local_mat_col[idof] = lane_val;
    }

    // coalesced write patterns (as coalesced as possible to global memory)
    // first write into shared

    // outer loop for each of the 4 nodal block columns
    for (int elem_block_col = 0; elem_block_col < Basis::num_nodes; elem_block_col++) {
        // this one has to be strided unfortunately from local to shared (unavoidable)
        // write transposed here? this is not actually parallel here
        int elem_block_row = ideriv / Phys::vars_per_node;
        for (int idof = iquad; idof < Phys::vars_per_node; idof += Quadrature::num_quad_pts) {
            block_col_buffer[elem_block_row][6 * ideriv + idof] =
                local_mat_col[Phys::vars_per_node * elem_block_col + idof];
        }

        // each warp considers one block at a time (first 3 elem_block_rows done first, then last
        // one)
        for (int elem_block_rowg = warp_ind; elem_block_rowg < Basis::num_nodes;
             elem_block_rowg += num_warps) {
            mat.addElementNodalBlock(active_thread, elem_block_rowg, elem_block_col, global_elem,
                                     local_warp_ind, warp_size, Phys::vars_per_node,
                                     Basis::num_nodes, block_col_buffer[elem_block_rowg]);
        }
    }
} else if constexpr (write_case == 3) {
    // nearly coalesced (uses only 36 total memory writes across all 576 values so average of 16
    // values at once or half a warp at once, optimal would be 32 writes, but is tricker to code, so
    // we're close enough to that) NOTE : assuming one elem_per_block at the moment.. can generalize
    // later if get greater occupancy..
    int warp_size = 32;
    int nwarp_rows = warp_size / blockDim.x;
    int iwarp = ideriv / nwarp_rows;
    int warp_row = ideriv % nwarp_rows;
    int ithread_warp = thread_xy - 32 * iwarp;

    // define the custom coalesced warp shuffle maps
    constexpr uint32_t val_mask_bits = 0b11001111000011001111000011001111;
    constexpr int8_t src_lanes[] = {0, 1, 2,  3,  0,  1,  6,  7,  4,  5,  6,  7,  8,  9,  10, 11,
                                    8, 9, 14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 22, 23};
    constexpr uint32_t val_mask_bits2 =
        0b0000110011110000;  // only considers half the # threads for second write (16)
    constexpr int8_t src_lanes2[] = {20, 21, 22, 23, 24, 25, 26, 27,
                                     24, 25, 30, 31, 28, 29, 30, 31};  // use mod 16

    // each warp parallelizes over the elem_block_rows not block cols (can't do both, only 3
    // parallel at a time)
    for (int elem_block_col = 0; elem_block_col < Basis::num_nodes; elem_block_col++) {
        // two coalesced memory writes intended => gets split into 3 when across node block
        // boundaries

        /* FIRST coalesced memory write, does warp shuffle them coalesced write */
        bool odd_row = (warp_row) % 2;
        int8_t ind = (Phys::vars_per_node * elem_block_col + iquad - 2 * odd_row) % 24;
        T val1 = local_mat_col[ind];
        T val2 = local_mat_col[ind + 4];  // 4 = num_quad_pts (can put in later)
        int8_t src_lane = src_lanes[ithread_warp];
        val1 = __shfl_sync(0xFFFFFFFF, val1, src_lane);
        val2 = __shfl_sync(0xFFFFFFFF, val2, src_lane);
        bool use_val1 = (val_mask_bits >> ithread_warp) & 1;
        T val = use_val1 ? val1 : val2;

        // now write into global (using new function for that), different from above because we'll
        // move ot different location in second write
        int c_warp_row = ithread_warp / Phys::vars_per_node;
        int c_elem_row = nwarp_rows * iwarp + c_warp_row;
        int elem_block_row = c_elem_row / Phys::vars_per_node;
        int inn_col = ithread_warp % Phys::vars_per_node;
        int inn_row = c_elem_row % Phys::vars_per_node;
        int inn_block_ind = Phys::vars_per_node * inn_row + inn_col;

        mat.coalescedBsrWrite(active_thread, elem_block_row, elem_block_col, global_elem,
                              Basis::num_nodes, inn_block_ind, val);

        /* SECOND coalesced memory write*/
        val1 = local_mat_col[ind];
        val2 = local_mat_col[ind + 4];
        int ithread_hwarp = ithread_warp % 16;
        bool use_val2 = (val_mask_bits2 >> ithread_hwarp) & 1;
        int8_t src_lane2 = src_lanes2[ithread_hwarp];
        val1 = __shfl_sync(0xffffffff, val1, src_lane2);
        val2 = __shfl_sync(0xffffffff, val2, src_lane2);
        val = use_val2 ? val1 : val2;

        c_warp_row = (ithread_warp + 32) / Phys::vars_per_node;
        c_elem_row = nwarp_rows * iwarp + c_warp_row;
        elem_block_row = c_elem_row / Phys::vars_per_node;
        inn_col = (ithread_warp + 32) % Phys::vars_per_node;
        inn_row = c_elem_row % Phys::vars_per_node;
        inn_block_ind = Phys::vars_per_node * inn_row + inn_col;

        // new condition is active thread and 32 + ithread_warp < 48 where 48 is based on # elem
        // rows in warp bool active_thread2 = active_thread &&
        mat.coalescedBsrWrite(active_thread && ithread_warp < 16, elem_block_row, elem_block_col,
                              global_elem, Basis::num_nodes, inn_block_ind, val);
    }
} else if constexpr (write_case == 4) {
    // nearly coalesced (uses only 36 total memory writes across all 576 values so average of 16
    // values at once or half a warp at once, optimal would be 32 writes, but is tricker to code, so
    // we're close enough to that) NOTE : assuming one elem_per_block at the moment.. can generalize
    // later if get greater occupancy.. int nwarp_rows =  / blockDim.x;
    int iwarp = ideriv / 8;
    int warp_row = ideriv % 8;
    int lane = thread_xy - 32 * iwarp;

    // define the custom coalesced warp shuffle maps
    constexpr uint32_t bool_pattern = 0b000011001111;  // 12-bit mask
    constexpr uint32_t src_pattern = 0b111111000000;   // 12-bit mask

    // each warp parallelizes over the elem_block_rows not block cols (can't do both, only 3
    // parallel at a time)
    for (int elem_block_col = 0; elem_block_col < Basis::num_nodes; elem_block_col++) {
        // two coalesced memory writes intended => gets split into 3 when across node block
        // boundaries

        /* FIRST coalesced memory write, does warp shuffle them coalesced write */
        bool odd_row = (warp_row) % 2;
        int ind = (Phys::vars_per_node * elem_block_col + iquad - 2 * odd_row) % 24;
        bool src_bool = (src_pattern >> (lane % 12)) & 1;
        int src_lane = 8 * (lane / 12) + lane % 4 + 4 * src_bool;
        bool use_val1 = (bool_pattern >> (lane % 12)) & 1;

        T val_lane = use_val1 ? __shfl_sync(0xFFFFFFFF, local_mat_col[ind], src_lane)
                              : __shfl_sync(0xFFFFFFFF, local_mat_col[ind + 4], src_lane);

        // if (blockIdx.x == 0) printf("thread %d, lane %d, src_lane %d, use_val1 %d\n", thread_xy,
        // lane, src_lane, (int) use_val1); return;

        // now write into global with this
        int c_elem_row = 8 * iwarp + lane / 6;
        int elem_block_row = c_elem_row / 4;
        int inn_row = c_elem_row % 6;
        int inn_col = lane % 6;
        int inn_block_ind = Phys::vars_per_node * inn_row + inn_col;

        mat.coalescedBsrWrite(active_thread, elem_block_row, elem_block_col, global_elem,
                              Basis::num_nodes, inn_block_ind, val_lane);

        /* SECOND coalesced memory write*/
        if (lane < 16) {
            lane += 32;
            src_bool = (src_pattern >> (lane % 12)) & 1;
            src_lane = 8 * (lane / 12) + lane % 4 + 4 * src_bool;
            use_val1 = (bool_pattern >> (lane % 12)) & 1;
            val_lane = use_val1 ? __shfl_sync(0xFFFFFFFF, local_mat_col[ind], src_lane)
                                : __shfl_sync(0xFFFFFFFF, local_mat_col[ind + 4], src_lane);

            c_elem_row = 8 * iwarp + lane / 6;
            elem_block_row = c_elem_row / 4;
            inn_row = c_elem_row % 6;
            inn_col = lane % 6;
            inn_block_ind = Phys::vars_per_node * inn_row + inn_col;

            // new condition is active thread and 32 + ithread_warp < 48 where 48 is based on # elem
            // rows in warp bool active_thread2 = active_thread &&
            mat.coalescedBsrWrite(active_thread, elem_block_row, elem_block_col, global_elem,
                                  Basis::num_nodes, inn_block_ind, val_lane);

            lane -= 32;
        }
    }
}
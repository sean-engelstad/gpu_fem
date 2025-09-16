// auto d_d = DeviceVec<T>(ndiag_vals);
        // d_diag_inv_vals = d_diag_inv_vals_vec.getPtr();

        // copy original D for DEBUG  (since LU decomp messes up comparison..)
        // auto d_diag_vals_copy = DeviceVec<T>(ndiag_vals);
        // cudaMemcpy(d_diag_vals_copy.getPtr(), d_diag_vals.getPtr(), ndiag_vals * sizeof(T),
        //            cudaMemcpyDeviceToDevice);

        if (use_cublas) {
            // make ptr-ptr objects.. that point to the D and Dinv single ptrs
            d_diag_LU_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            nblocks = (nnodes + 31) / 32;
            dim3 grid2(nblocks);
            k_singleToDoublePointer<T>
                <<<grid2, block>>>(nnodes, block_dim, d_diag_vals.getPtr(), d_diag_LU_batch_ptr);

            d_temp_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            k_singleToDoublePointerVec<T>
                <<<grid2, block>>>(nnodes, block_dim, d_temp, d_temp_batch_ptr);

            // T **d_diag_inv_batch_ptr = DeviceVec<T *>(nnodes).getPtr();
            // k_singleToDoublePointer<T>
            //     <<<grid2, block>>>(nnodes, block_dim, d_diag_inv_vals, d_diag_inv_batch_ptr);

            // // get row scaling..
            // T *d_diag_scales = DeviceVec<T>(N).getPtr();
            // // divide D by the row scales
            // nblocks = (N + 31) / 32;
            // dim3 grid3(nblocks);
            // k_computeDiagRowScales<T>
            //     <<<grid3, block>>>(nnodes, block_dim, d_diag_vals.getPtr(), d_diag_scales);

            // // then we'll do local 6x6 inverses D => Dinv of the block diag matrix into
            // // d_diag_inv_vals
            // cudaMemcpy(d_diag_inv_vals, d_diag_vals.getPtr(), ndiag_vals * sizeof(T),
            //            cudaMemcpyDeviceToDevice);

            // now use cublas to do diag inv in batch (on diag), other option is to use cusparse
            // if this is slow.. first we do in-place LU decomp,
            // https://docs.nvidia.com/cuda/cublas/ first an in-place LU decomp P*A = L*U (with
            // pivots on each 6x6 nodal block) in-place on d_diag_vals
            d_piv = DeviceVec<int>(nnodes * block_dim).getPtr();
            d_info = DeviceVec<int>(nnodes).getPtr();
            cublasDgetrfBatched(cublasHandle, block_dim, d_diag_LU_batch_ptr, block_dim, d_piv,
                                d_info, nnodes);  // LU decomp in place here..

            // really singular block diag => need to use LU batched can't get accurate full D^-1
            // directly

            // then do an inversion from d_diag_vals LU decomp => d_diag_inv_vals ptr
            // get ri batched is really inaccurate..
            // this is really inaccurate on first call especially if matrices ill-conditioned, often
            // are.. NOTE : could do newton-schulz refinement for matrix inversion:
            //    X_{k+1} = X_k * (2I - D * X_k)
            // OR like here I'm just going to set D = S * A where S is scaling 6x6 diag matrix
            // and A has ones on diag, so scaled to O(1), thus each row is normalized by diag(S)
            // cublasDgetriBatched(cublasHandle, block_dim, d_diag_batch_ptr, block_dim, d_piv,
            //                     d_diag_inv_batch_ptr, block_dim, d_info, nnodes);

            // // undo the row scalings..
            // k_reapplyDiagRowScales<T><<<grid3, block>>>(nnodes, block_dim, d_diag_scales,
            //                                             d_diag_vals.getPtr(), d_diag_inv_vals);

        }  // end of cublas

// // DEBUG: check the 6x6 diag and diag inv matrices..
        // T *h_diag_vals = d_diag_vals.createHostVec().getPtr();
        // // printf("step 4\n");
        // for (int inode = 0; inode < 3; inode++) {
        //     printf("node %d D vs Dinv\n", inode);

        //     for (int icol = 0; icol < 6; icol++) {
        //         cudaMemset(d_temp, 0.0, N * sizeof(T));
        //         k_setSingleVal<<<1, 1>>>(N, 6 * inode + icol, 1.0, d_temp);

        //         printf("\tD[:,%d]: ", icol);
        //         // sym so actually row printout here
        //         printVec<T>(6, &h_diag_vals[36 * inode + 6 * icol]);

        //         // test the Dinv using getrsbatched on unit vecs
        //         cublasDgetrsBatched(cublasHandle, CUBLAS_OP_N, block_dim, 1, d_diag_LU_batch_ptr,
        //                             block_dim, d_piv, d_temp_batch_ptr, block_dim, d_info,
        //                             nnodes);

        //         T *h_temp = new T[6];
        //         cudaMemcpy(h_temp, &d_temp[6 * inode], 6 * sizeof(T), cudaMemcpyDeviceToHost);

        //         printf("\tLU=>Dinv[:,%d]: ", icol);
        //         // sym so actually row printout here
        //         printVec<T>(6, h_temp);
        //     }
        // }

        // test t

        // printf("here1\n");
        // d_diag_vals.free();
        // // delete[] d_diag_batch_ptr;
        // // delete[] d_diag_inv_batch_ptr;
        // printf("here2\n");

    
        // print out one LU matrix..
                // T *h_diag_LU_vals = DeviceVec<T>(36, d_diag_LU_vals_color).createHostVec().getPtr();
                // printf("h diag LU vals (one node): ");
                // for (int i = 0; i < 36; i++) {
                //     printf("%.12e, ", h_diag_LU_vals[i]);
                // }
                // printf("\n");
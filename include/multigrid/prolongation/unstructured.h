#pragma once

#include "unstruct_utils.h"
#include "unstructured.cuh"
#include "coupled/locate_point.h"

template <class Basis, bool is_bsr_ = false>
class UnstructuredProlongation {
   public:
    using T = double;
    static constexpr bool structured = false;
    static constexpr bool assembly = true;
    static constexpr int is_bsr = is_bsr_;

    static void assemble_matrices(int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems,
                                  T *d_n2e_xis, BsrMat<DeviceVec<T>> &P_mat,
                                  BsrMat<DeviceVec<T>> &PT_mat) {
        // get some data out..
        auto P_bsr_data = P_mat.getBsrData();
        auto PT_bsr_data = PT_mat.getBsrData();
        int nnodes_fine = P_bsr_data.nnodes;
        // int nnodes_coarse = PT_bsr_data.nnodes;
        int *d_coarse_iperm = PT_bsr_data.iperm;
        int *d_fine_iperm = P_bsr_data.iperm;
        int block_dim = P_bsr_data.block_dim;

        // assemble P mat
        dim3 block(32);
        dim3 grid((nnodes_fine + 31) / 32);
        int *d_P_rowp = P_bsr_data.rowp, *d_P_cols = P_bsr_data.cols;
        T *d_P_vals = P_mat.getPtr();
        k_prolong_mat_assembly<T, Basis, is_bsr>
            <<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                              nnodes_fine, d_fine_iperm, d_P_rowp, d_P_cols, block_dim, d_P_vals);

        // assemble PT mat
        int *d_PT_rowp = PT_bsr_data.rowp, *d_PT_cols = PT_bsr_data.cols;
        T *d_PT_vals = PT_mat.getPtr();
        k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block>>>(
            d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, nnodes_fine,
            d_fine_iperm, d_PT_rowp, d_PT_cols, block_dim, d_PT_vals);
    }

    static void prolongate(cusparseHandle_t handle, cusparseMatDescr_t &descr_P,
                           BsrMat<DeviceVec<T>> prolong_mat, DeviceVec<T> perm_coarse_soln_in,
                           DeviceVec<T> perm_dx_fine) {
        // get important data & vecs out
        // printf("unstruct prolong fast start\n");
        auto P_bsr_data = prolong_mat.getBsrData();
        int *d_cols = P_bsr_data.cols, nnzb = P_bsr_data.nnzb;
        T *d_vals = prolong_mat.getPtr();

        // now do cusparse Bsrmv.. for P @ coarse_soln => dx_fine (permuted nodes order)
        if constexpr (is_bsr) {
            T a = 1.0, b = 0.0;
            int *d_rowp = P_bsr_data.rowp;
            int mb = P_bsr_data.mb, nb = P_bsr_data.nb, block_dim = P_bsr_data.block_dim;
            CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
                                                &a, descr_P, d_vals, d_rowp, d_cols, block_dim,
                                                perm_coarse_soln_in.getPtr(), &b,
                                                perm_dx_fine.getPtr()));
        } else {
            // else CSR case (same transfer stencil for each dof per node)
            dim3 block(32);
            dim3 grid((nnzb + 31) / 32);

            int *d_rows = P_bsr_data.rows;
            k_csr_mat_vec<T><<<grid, block>>>(nnzb, 6, d_rows, d_cols, d_vals,
                                            perm_coarse_soln_in.getPtr(), perm_dx_fine.getPtr());
        }

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("unstruct prolong fast end\n");
    }

    static void restrict_defect(cusparseHandle_t handle, cusparseMatDescr_t descr_PT,
                                BsrMat<DeviceVec<T>> restrict_mat, DeviceVec<T> fine_defect_in,
                                DeviceVec<T> coarse_defect_out) {
        // get important data & vecs out
        // printf("unstruct restrict fast start\n");
        auto PT_bsr_data = restrict_mat.getBsrData();
        int *d_cols = PT_bsr_data.cols, nnzb = PT_bsr_data.nnzb;
        T *d_vals = restrict_mat.getPtr();

        if constexpr (is_bsr) {
            T a = 1.0, b = 0.0;
            int *d_rowp = PT_bsr_data.rowp;
            int mb = PT_bsr_data.mb, nb = PT_bsr_data.nb, block_dim = PT_bsr_data.block_dim;
            CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
                                                &a, descr_PT, d_vals, d_rowp, d_cols, block_dim,
                                                fine_defect_in.getPtr(), &b,
                                                coarse_defect_out.getPtr()));
        } else {
            dim3 block(32);
            dim3 grid((nnzb + 31) / 32);

            int *d_rows = PT_bsr_data.rows;
            k_csr_mat_vec<T><<<grid, block>>>(nnzb, 6, d_rows, d_cols, d_vals, fine_defect_in.getPtr(),
                                            coarse_defect_out.getPtr());
        }

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("unstruct restrict defect fast end\n");
    }

    template <class Basis>
    void init_unstructured_grid_maps(ShellGrid &coarse_grid, int ELEM_MAX = 4) {
        /* initialize the unstructured mesh prolongation map */
        // TBD, want to get the coarse nodes

        auto start0 = std::chrono::high_resolution_clock::now();

        // -------------------------------------------------------
        // 1) prelim prolongation maps and data here..
        T *h_xpts_fine = assembler.getXpts().createHostVec().getPtr();
        T *h_xpts_coarse = coarse_grid.assembler.getXpts().createHostVec().getPtr();
        int nnodes_fine = assembler.get_num_nodes();
        int nnodes_coarse = coarse_grid.assembler.get_num_nodes();

        int min_bin_size = 10;
        auto *locator = new LocatePoint<T>(h_xpts_coarse, nnodes_coarse, min_bin_size);

        int nn = 6;  // number of coarse node nearest neighbors
        int *nn_conn = new int[nn * nnodes_fine];
        // temp work arrays for each point
        int *indx = new int[nn];
        T *dist = new T[nn];

        // also get fine nearest neighbors for fine dzeta estimates later..

        // fine to coarse node nearest neighbors
        for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
            T loc_xfine[3];
            memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));  // is this part necessary?

            locator->locateKClosest(nn, indx, dist, loc_xfine);

            for (int k = 0; k < nn; k++) {
                nn_conn[nn * inode_f + k] = indx[k];
            }
        }
        delete[] indx;
        delete[] dist;

        // fine to fine node nearest neighbors
        // need more nearest neighbors for fine node dzeta estimate
        int nn_f = 20;  // number of coarse node nearest neighbors
        auto *locator_fine = new LocatePoint<T>(h_xpts_fine, nnodes_fine, min_bin_size);
        int *nn_conn_fine = new int[nn_f * nnodes_fine];
        indx = new int[nn_f];
        dist = new T[nn_f];

        for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
            T loc_xfine[3];
            memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));  // is this part necessary?
            locator_fine->locateKClosest(nn_f, indx, dist, loc_xfine);

            for (int k = 0; k < nn_f; k++) {
                nn_conn_fine[nn_f * inode_f + k] = indx[k];
            }
        }

        // printf("nn conn: ");
        // printVec<int>(30, nn_conn);

        // -------------------------------------------------------
        // 2) get coarse elements for each coarse node
        auto d_coarse_conn_vec = coarse_grid.assembler.getConn();
        int *h_coarse_conn = d_coarse_conn_vec.createHostVec().getPtr();
        int *cnode_elem_cts = new int[nnodes_coarse];
        memset(cnode_elem_cts, 0.0, nnodes_coarse * sizeof(int));
        int *cnode_elem_ptr = new int[nnodes_coarse + 1];
        int num_coarse_elems = coarse_grid.assembler.get_num_elements();
        ncoarse_elems = num_coarse_elems;
        for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
            for (int iloc = 0; iloc < 4; iloc++) {
                int cnode = h_coarse_conn[4 * ielem + iloc];
                cnode_elem_cts[cnode] += 1;
            }
        }

        // printf("cnode elem cts: ");
        // printVec<int>(nnodes_coarse, cnode_elem_cts);

        // like rowp here (says where and how many elems this node is a part of)
        cnode_elem_ptr[0] = 0;
        for (int inode = 0; inode < nnodes_coarse; inode++) {
            cnode_elem_ptr[inode + 1] = cnode_elem_ptr[inode] + cnode_elem_cts[inode];
        }
        int n_coarse_node_elems = cnode_elem_ptr[nnodes_coarse];

        // now we put which elems each coarse node is connected to (like cols array here)
        // reset row cts to 0, so you can trick which local elem you're writing in..
        int *_cnode_next = new int[nnodes_coarse];
        memcpy(_cnode_next, cnode_elem_ptr, nnodes_coarse * sizeof(int));
        int *cnode_elems = new int[n_coarse_node_elems];
        for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
            for (int iloc = 0; iloc < 4; iloc++) {
                int cnode = h_coarse_conn[4 * ielem + iloc];
                cnode_elems[_cnode_next[cnode]++] = ielem;
            }
        }

        // printf("cnode elem ptr: ");
        // printVec<int>(nnodes_coarse + 1, cnode_elem_ptr);
        // printf("cnode elems (cols): ");
        // printVec<int>(n_coarse_node_elems, cnode_elems);

        // -----------------------------------------------------------
        // 2.5 ) get the components for each coarse and fine node
        int num_fine_elems = assembler.get_num_elements();
        int *h_fine_elem_comps = assembler.getElemComponents().createHostVec().getPtr();
        int *h_coarse_elem_comps =
            coarse_grid.assembler.getElemComponents().createHostVec().getPtr();
        int *h_fine_conn = assembler.getConn().createHostVec().getPtr();

        // printf("h_fine_elem_comps: ");
        // printVec<int>(100, h_fine_elem_comps);

        // printf("h_fine_conn: ");
        // printVec<int>(30, h_fine_conn);

        int *f_ecomps_cts = new int[nnodes_fine];
        memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
        int *f_ecomps_ptr0 = new int[nnodes_fine + 1];
        for (int ielem = 0; ielem < num_fine_elems; ielem++) {
            // int fine_comp = h_fine_elem_comps[ielem];
            for (int iloc = 0; iloc < 4; iloc++) {
                int fnode = h_fine_conn[4 * ielem + iloc];
                // printf("fine elem %d, node %d\n", ielem, fnode);
                f_ecomps_cts[fnode]++;
            }
        }

        f_ecomps_ptr0[0] = 0;
        for (int inode = 0; inode < nnodes_fine; inode++) {
            f_ecomps_ptr0[inode + 1] = f_ecomps_ptr0[inode] + f_ecomps_cts[inode];
        }
        int f_ecomps_nnz0 = f_ecomps_ptr0[nnodes_fine];
        int *f_ecomps_comp0 = new int[f_ecomps_nnz0];
        // reset counts to fill comp
        memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
        memset(f_ecomps_comp0, -1, f_ecomps_nnz0 * sizeof(int));
        for (int ielem = 0; ielem < num_fine_elems; ielem++) {
            int fine_comp = h_fine_elem_comps[ielem];
            for (int iloc = 0; iloc < 4; iloc++) {
                int fnode = h_fine_conn[4 * ielem + iloc];
                int start = f_ecomps_ptr0[fnode];
                int write = f_ecomps_cts[fnode] + start;
                // check if this comp already written in or not..
                bool new_comp = true;
                for (int jp = start; jp < write; jp++) {
                    int _comp = f_ecomps_comp0[jp];
                    if (_comp == fine_comp) new_comp = false;
                }

                if (new_comp) {
                    f_ecomps_comp0[write] = fine_comp;
                    f_ecomps_cts[fnode]++;
                }
            }
        }

        // now that we've prevented repeats, adjust ptr size and write new comp and new nnz
        int *f_ecomps_ptr = new int[nnodes_fine + 1];
        f_ecomps_ptr[0] = 0;
        for (int inode = 0; inode < nnodes_fine; inode++) {
            f_ecomps_ptr[inode + 1] = f_ecomps_ptr[inode] + f_ecomps_cts[inode];
        }
        int f_ecomps_nnz = f_ecomps_ptr[nnodes_fine];
        int *f_ecomps_comp = new int[f_ecomps_nnz];
        // reset counts to fill comp
        memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
        memset(f_ecomps_comp0, -1, f_ecomps_nnz0 * sizeof(int));
        for (int ielem = 0; ielem < num_fine_elems; ielem++) {
            int fine_comp = h_fine_elem_comps[ielem];
            for (int iloc = 0; iloc < 4; iloc++) {
                int fnode = h_fine_conn[4 * ielem + iloc];
                int start = f_ecomps_ptr[fnode];
                int write = f_ecomps_cts[fnode] + start;
                // check if this comp already written in or not..
                bool new_comp = true;
                for (int jp = start; jp < write; jp++) {
                    int _comp = f_ecomps_comp[jp];
                    if (_comp == fine_comp) new_comp = false;
                }

                if (new_comp) {
                    f_ecomps_comp[write] = fine_comp;
                    f_ecomps_cts[fnode]++;
                }
            }
        }

        // ----------------------------------------------------------------
        // 3) get the coarse element(s) for each fine node (that it's contained in)

        int *fine_nodes_celem_cts = new int[nnodes_fine];
        memset(fine_nodes_celem_cts, 0, nnodes_fine * sizeof(int));
        int *fine_nodes_celems = new int[ELEM_MAX * nnodes_fine];
        memset(fine_nodes_celems, -1, ELEM_MAX * nnodes_fine * sizeof(int));
        T *fine_node_xis = new T[2 * ELEM_MAX * nnodes_fine];
        memset(fine_node_xis, 0.0, 2 * ELEM_MAX * nnodes_fine * sizeof(T));
        int ntot_elems = 0;

        for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
            T *fine_node_xpts = &h_xpts_fine[3 * inode_f];
            // printf("inode_f %d\n", inode_f);

            // should be ~24 elems examined per coarse node (can parallelize this on GPU if needed)
            for (int i_nn = 0; i_nn < nn; i_nn++) {
                int inode_c = nn_conn[nn * inode_f + i_nn];

                for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                    int ielem_c = cnode_elems[jp];
                    // get coarse element component
                    int c_comp = h_coarse_elem_comps[ielem_c];

                    // check among comp ptr to see if fine node also belongs to this component
                    bool match_comp = false;
                    for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                        int f_comp = f_ecomps_comp[jjp];
                        match_comp += f_comp == c_comp;
                        // if (print_node) printf("%d ", f_comp);
                    }
                    // if (print_node) printf(", match_comp %d\n", (int)match_comp);

                    T coarse_elem_xpts[12];
                    get_elem_xpts<T>(ielem_c, h_coarse_conn, h_xpts_coarse, coarse_elem_xpts);

                    // bool print_debug = inode_f == 1098;
                    bool print_debug = false;

                    T xi[3];
                    get_comp_coords<T, Basis>(coarse_elem_xpts, fine_node_xpts, xi, print_debug);

                    // determine whether xi[3] is in bounds or not; xi & eta in [-1,1] and zeta in
                    // [-2,2] for max thick (or can ignore zeta..)
                    // bool node_in_elem = xis_in_elem<T>(xi, dzeta);
                    bool node_in_elem = xis_in_elem<T>(xi, match_comp);

                    if (node_in_elem) {
                        // check if element already in old elems of this node
                        int nelems_prev = fine_nodes_celem_cts[inode_f];
                        bool new_elem = true;
                        for (int i = 0; i < nelems_prev; i++) {
                            int prev_elem = fine_nodes_celems[ELEM_MAX * inode_f + i];
                            new_elem = new_elem && (prev_elem != ielem_c);
                        }

                        if (new_elem) {
                            fine_nodes_celem_cts[inode_f]++;
                            ntot_elems++;
                            fine_nodes_celems[ELEM_MAX * inode_f + nelems_prev] = ielem_c;
                            fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev] = xi[0];
                            fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev + 1] = xi[1];
                        }
                    }
                }
            }
        }

        // now convert from cts to rowp, cols style as diff # elems per fine node
        int *fine_node2elem_ptr = new int[nnodes_fine + 1];
        fine_node2elem_ptr[0] = 0;
        int *fine_node2elem_elems = new int[ntot_elems];
        T *fine_node2elem_xis = new T[2 * ntot_elems];

        for (int inode = 0; inode < nnodes_fine; inode++) {
            int ct = fine_nodes_celem_cts[inode];
            fine_node2elem_ptr[inode + 1] = fine_node2elem_ptr[inode] + ct;
            int start = fine_node2elem_ptr[inode];

            for (int i = 0; i < ct; i++) {
                int src_block = ELEM_MAX * inode + i, dest_block = start + i;
                fine_node2elem_elems[dest_block] = fine_nodes_celems[src_block];
                fine_node2elem_xis[2 * dest_block] = fine_node_xis[2 * src_block];
                fine_node2elem_xis[2 * dest_block + 1] = fine_node_xis[2 * src_block + 1];
            }
        }

        // printf("h_n2e_ptr: ");
        // printVec<int>(3, fine_node2elem_ptr);
        // printf("h_n2e_elems: ");
        // printVec<int>(20, fine_node2elem_elems);
        // printf("h_n2e_xis: ");
        // printVec<T>(40, fine_node2elem_xis);

        // put these maps on the device now
        d_n2e_ptr = HostVec<int>(nnodes_fine + 1, fine_node2elem_ptr).createDeviceVec().getPtr();
        d_n2e_elems = HostVec<int>(ntot_elems, fine_node2elem_elems).createDeviceVec().getPtr();
        d_n2e_xis = HostVec<T>(2 * ntot_elems, fine_node2elem_xis).createDeviceVec().getPtr();
        d_coarse_conn = d_coarse_conn_vec.getPtr();
        n2e_nnz = ntot_elems;
        // and also store these maps on the coarser mesh also
        coarse_grid.restrict_d_n2e_ptr = d_n2e_ptr;
        coarse_grid.restrict_d_n2e_elems = d_n2e_elems;
        coarse_grid.restrict_d_n2e_xis = d_n2e_xis;
        coarse_grid.restrict_nnodes_fine = nnodes;
        coarse_grid.restrict_n2e_nnz = ntot_elems;

        // assemble the prolongation matrix and its transpose on the device (in permuted form)
        // for fast UnstructuredProlongation using assembled matrix
        // --------------------------------------------------------

        // now call assembler of P and PT matrices
        if constexpr (Prolongation::assembly) {
            // int mb = nnodes_fine, nb = nnodes_coarse;
            int *h_perm = DeviceVec<int>(nnodes_fine, d_perm).createHostVec().getPtr();
            int *h_iperm = DeviceVec<int>(nnodes_fine, d_iperm).createHostVec().getPtr();
            int *h_coarse_iperm =
                DeviceVec<int>(nnodes_coarse, coarse_grid.d_iperm).createHostVec().getPtr();

            // compute the pattern here first, TODO is to clean this up and use less mem if possible
            // at least free extra stuff
            int *h_prol_row_cts = new int[nnodes_fine];
            int *h_prolT_row_cts = new int[nnodes_coarse];
            memset(h_prol_row_cts, 0.0, nnodes_fine * sizeof(int));
            memset(h_prolT_row_cts, 0.0, nnodes_coarse * sizeof(int));
            // TODO : clean much of this up with sparse symbolic later.. though sparse symbolic uses
            // connectivity, not generating non-square matrices usually uses elem_conn which is like
            // all of [1,2,7,8] connected, not the same as here where I have pairs [1] x [4,5,7,8]
            // elements fine to coarse maybe need to make some new methods in sparse symbolic.. (but
            // let's get it running first)

            for (int inf = 0; inf < nnodes_fine; inf++) {
                int perm_inf = h_iperm[inf];
                // int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
                std::set<int> conn_c_nodes;

                for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
                    int ielem_c = fine_node2elem_elems[jp];
                    const int *c_elem_nodes = &h_coarse_conn[4 * ielem_c];

                    // get xi to interp the fine node (so we can check if some local nodes in element aren't really used)
                    // like when the fine node is on the edge node. Can just compute transpose interp map, BUT ONLY DO THIS AFTER TRY SMOOTH PROLONG WITHOUT PRUNING MEM HERE FIRST

                    for (int loc_node = 0; loc_node < 4; loc_node++) {
                        int inc = c_elem_nodes[loc_node];
                        int perm_inc = h_coarse_iperm[inc];

                        // unique col cts?
                        conn_c_nodes.insert(perm_inc);
                    }
                }

                h_prol_row_cts[perm_inf] += conn_c_nodes.size();
                for (int perm_inc : conn_c_nodes) {
                    h_prolT_row_cts[perm_inc]++;
                }
            }

            // printf("h_prol_row_cts:");
            // printVec<int>(100, h_prol_row_cts);

            // now construct rowp for P and PT
            int *h_prol_rowp = new int[nnodes_fine + 1];
            int *h_prolT_rowp = new int[nnodes_coarse + 1];
            h_prol_rowp[0] = 0;
            h_prolT_rowp[0] = 0;

            for (int inf = 0; inf < nnodes_fine; inf++) {
                h_prol_rowp[inf + 1] = h_prol_rowp[inf] + h_prol_row_cts[inf];
            }
            for (int inc = 0; inc < nnodes_coarse; inc++) {
                h_prolT_rowp[inc + 1] = h_prolT_rowp[inc] + h_prolT_row_cts[inc];
            }

            // now construct cols
            int P_nnzb = h_prol_rowp[nnodes_fine];
            int PT_nnzb = h_prolT_rowp[nnodes_coarse];
            int *h_prol_cols = new int[P_nnzb];
            int *h_prolT_cols = new int[PT_nnzb];
            int *P_next = new int[nnodes_fine];  // helper arrays to insert values into sparsity
            memcpy(P_next, h_prol_rowp, nnodes_fine * sizeof(int));
            int *PT_next = new int[nnodes_coarse];
            memcpy(PT_next, h_prolT_rowp, nnodes_coarse * sizeof(int));
            // loop back through previous maps to get the sparsity
            // use perm_inf order here to help with PT sparsity
            for (int perm_inf = 0; perm_inf < nnodes_fine; perm_inf++) {
                int inf = h_perm[perm_inf];
                // int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
                std::set<int> conn_c_nodes;
                for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
                    int ielem_c = fine_node2elem_elems[jp];
                    const int *c_elem_nodes = &h_coarse_conn[4 * ielem_c];
                    for (int loc_node = 0; loc_node < 4; loc_node++) {
                        int inc = c_elem_nodes[loc_node];
                        int perm_inc = h_coarse_iperm[inc];

                        // unique col cts?
                        conn_c_nodes.insert(perm_inc);
                    }
                }

                for (int perm_inc : conn_c_nodes) {
                    h_prol_cols[P_next[perm_inf]++] = perm_inc;    // P cols
                    h_prolT_cols[PT_next[perm_inc]++] = perm_inf;  // PT cols
                }
            }

            // double check all values filled?
            // for (int i = 0; i < nnodes_fine; i++) {
            //     printf("P[i=%d,:] : ", i);
            //     for (int jp = h_prol_rowp[i]; jp < h_prol_rowp[i+1]; jp++) {
            //         int j = h_prol_cols[jp];
            //         printf("%d, ", j);
            //     }
            //     printf("\n");
            // }

            int *h_P_rows = new int[P_nnzb];
            int *h_PT_rows = new int[PT_nnzb];
            for (int inode = 0; inode < nnodes_fine; inode++) {
                for (int jp = h_prol_rowp[inode]; jp < h_prol_rowp[inode + 1]; jp++) {
                    h_P_rows[jp] = inode;
                }
            }
            for (int inode = 0; inode < nnodes_coarse; inode++) {
                for (int jp = h_prolT_rowp[inode]; jp < h_prolT_rowp[inode + 1]; jp++) {
                    h_PT_rows[jp] = inode;
                }
            }
            int *d_P_rows = HostVec<int>(P_nnzb, h_P_rows).createDeviceVec().getPtr();
            int *d_PT_rows = HostVec<int>(PT_nnzb, h_PT_rows).createDeviceVec().getPtr();

            // now put these on the device with BsrData objects for P and PT
            // TODO : later is to just assemble P and then write my own transpose Bsrmv method in
            // CUDA
            int P_block_dim = Prolongation::is_bsr ? block_dim : 1; // if !is_bsr then it does same prolong on each dof_per_node
            int block_dim2 = P_block_dim * P_block_dim;  // should be 36
            auto d_P_rowp = HostVec<int>(nnodes_fine + 1, h_prol_rowp).createDeviceVec().getPtr();
            auto d_P_cols = HostVec<int>(P_nnzb, h_prol_cols).createDeviceVec().getPtr();
            auto d_P_vals = DeviceVec<T>(block_dim2 * P_nnzb);
            auto P_bsr_data = BsrData(nnodes_fine, P_block_dim, P_nnzb, d_P_rowp, d_P_cols, d_perm,
                                      d_iperm, false);
            P_bsr_data.mb = nnodes_fine, P_bsr_data.nb = nnodes_coarse;
            P_bsr_data.rows = d_P_rows;
            P_mat = BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals);

            auto d_PT_rowp =
                HostVec<int>(nnodes_coarse + 1, h_prolT_rowp).createDeviceVec().getPtr();
            auto d_PT_cols = HostVec<int>(PT_nnzb, h_prolT_cols).createDeviceVec().getPtr();
            auto d_PT_vals = DeviceVec<T>(block_dim2 * PT_nnzb);
            auto PT_bsr_data = BsrData(nnodes_coarse, P_block_dim, PT_nnzb, d_PT_rowp, d_PT_cols,
                                       coarse_grid.d_perm, coarse_grid.d_iperm, false);
            PT_bsr_data.mb = nnodes_coarse, PT_bsr_data.nb = nnodes_fine;
            PT_bsr_data.rows = d_PT_rows;  // for each nnz, which row is it (not same as rowp),
                                           // helps for efficient mat-prods
            auto PT_mat = BsrMat<DeviceVec<T>>(
                PT_bsr_data, d_PT_vals);  // only store this matrix on the coarse grid

            // printf("assemble matrices pre\n");
            // printf("assemble P and PT matrices with nnodes_fine %d and nnodes_coarse %d\n",
            // nnodes_fine, nnodes_coarse);
            Prolongation::assemble_matrices(d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, P_mat,
                                            PT_mat);
            CHECK_CUDA(cudaDeviceSynchronize());
            // printf("assemble matrices post\n");
            coarse_grid.restrict_PT_mat = PT_mat;

            // TEMP DEBUG
            // T *h_P_vals = d_P_vals.createHostVec().getPtr();
            // printf("h_P_vals:\n");
            // for (int i = 0; i < 5; i++) {
            //     printf("P mat row %d: ", i);
            //     for (int jp = h_prol_rowp[i]; jp < h_prol_rowp[i+1];  jp++) {
            //         int j = h_prol_cols[jp];
            //         T val = h_P_vals[jp];
            //         printf("[%d] %.3e, ", j, val);
            //     }
            //     printf("\n");
            // }

            // matrix descriptions for Bsrmv..
            descrP = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descrP));
            CHECK_CUSPARSE(cusparseSetMatType(descrP, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descrP, CUSPARSE_INDEX_BASE_ZERO));

            coarse_grid.restrict_descrPT = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&coarse_grid.restrict_descrPT));
            CHECK_CUSPARSE(
                cusparseSetMatType(coarse_grid.restrict_descrPT, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(
                cusparseSetMatIndexBase(coarse_grid.restrict_descrPT, CUSPARSE_INDEX_BASE_ZERO));

        }  // end of Prolongation::assembly case..

        // printf("done with init unstructured grid maps\n");
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> P_PT_time = end0 - start0;
        printf("unstructured grid P,PT assembly in %.2e sec\n", P_PT_time.count());

        // TBD: free up temp arrays
    }
};
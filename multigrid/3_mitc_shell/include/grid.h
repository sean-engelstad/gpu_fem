// geom multigrid for the shells
#pragma once
#include <cusparse_v2.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/bsr_mat.h"
#include "solvers/linear_static/bsr_direct_LU.h"

// local includes for shell multigrid
#include "grid.cuh"
#include "prolongation/_prolong.h"  // for prolongations
#include "prolongation/unstruct_utils.h"
#include "vtk.h"

// for unstructured multigrid
#include "coupled/locate_point.h"

enum SMOOTHER : short {
    MULTICOLOR_GS,
    MULTICOLOR_GS_FAST,
    LEXIGRAPHIC_GS,
};

template <class Assembler, class Prolongation, SMOOTHER smoother>
class ShellGrid {
   public:
    using T = double;
    using I = long long int;

    ShellGrid() = default;

    ShellGrid(Assembler &assembler_, int N_, BsrMat<DeviceVec<T>> Kmat_, DeviceVec<T> d_rhs_,
              HostVec<int> h_color_rowp_, bool full_LU_ = false)
        : N(N_), full_LU(full_LU_) {
        Kmat = Kmat_;
        d_rhs = d_rhs_;
        h_color_rowp = h_color_rowp_;
        block_dim = 6;
        nnodes = N / 6;

        assembler = assembler_;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        // init helper methods
        if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
            buildColorLocalRowPointers();
        }
        initCuda();
        if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
            buildDiagInvMat();
        }
        if (smoother == LEXIGRAPHIC_GS && !full_LU) {
            initLowerMatForGaussSeidel();
        }
    }

    static ShellGrid *buildFromAssembler(Assembler &assembler, T *h_loads, bool full_LU = false,
                                         bool reorder = true) {
        // only do full LU factor on coarsest grid..

        // BSR symbolic factorization
        // must pass by ref to not corrupt pointers
        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        if (full_LU) {
            // coarsest grid just gets AMD ordering and full LU pattern (AMD uses less memory for
            // direct solves + tends to be fast)
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
            num_colors = 1;
            _color_rowp = new int[2];
            _color_rowp[0] = 0;
            _color_rowp[1] = assembler.get_num_nodes();

        } else {
            if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
                if (reorder) {
                    bsr_data.multicolor_reordering(
                        num_colors,
                        _color_rowp);  // TODO : add this method.. (I guess I
                                       // can just do host for now..)
                } else {
                    num_colors = 1;
                    _color_rowp = new int[2];
                    _color_rowp[0] = 0;
                    _color_rowp[1] = assembler.get_num_nodes();
                }
            } else if (smoother == LEXIGRAPHIC_GS) {
                if (reorder) {
                    bsr_data.RCM_reordering(1);
                }
                // default or no colors..
                num_colors = 1;
                _color_rowp = new int[2];
                _color_rowp[0] = 0;
                _color_rowp[1] = assembler.get_num_nodes();
            }

            // after done multicolor or whatever reordering for finer meshes that use weak linear
            // solvers for GMG compute nofill pattern
            bsr_data.compute_nofill_pattern();
        }
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);

        assembler.moveBsrDataToDevice();
        auto loads = assembler.createVarsVec(h_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto soln = assembler.createVarsVec();
        auto res = assembler.createVarsVec();
        auto vars = assembler.createVarsVec();
        int N = vars.getSize();

        // assemble the kmat
        assembler.add_jacobian(res, kmat);
        assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);

        return new ShellGrid(assembler, N, kmat, loads, h_color_rowp, full_LU);
    }

    template <class Basis>
    void init_unstructured_grid_maps(ShellGrid &coarse_grid, int ELEM_MAX = 4) {
        /* initialize the unstructured mesh prolongation map */
        // TBD, want to get the coarse nodes

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
        int n_coarse_node_elems = 4 * num_coarse_elems;
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

        // now we put which elems each coarse node is connected to (like cols array here)
        // reset row cts to 0, so you can trick which local elem you're writing in..
        memset(cnode_elem_cts, 0.0, nnodes_coarse * sizeof(int));
        int *cnode_elems = new int[n_coarse_node_elems];
        for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
            for (int iloc = 0; iloc < 4; iloc++) {
                int cnode = h_coarse_conn[4 * ielem + iloc];
                int ind = cnode_elem_ptr[cnode] + cnode_elem_cts[cnode];
                cnode_elems[ind] = ielem;
                cnode_elem_cts[cnode] += 1;
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
        // printf("h_fine_conn: ");
        // printVec<int>(30, h_fine_conn);

        int *f_ecomps_cts = new int[nnodes_fine];
        memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
        int *f_ecomps_ptr0 = new int[nnodes_fine + 1];
        for (int ielem = 0; ielem < num_fine_elems; ielem++) {
            int fine_comp = h_fine_elem_comps[ielem];
            for (int iloc = 0; iloc < 4; iloc++) {
                int fnode = h_fine_conn[4 * ielem + iloc];
                // printf("fine elem %d, node %d\n", ielem, fnode);
                f_ecomps_cts[fnode]++;
            }
        }

        printf("f_ecomps_cts: ");
        printVec<int>(30, f_ecomps_cts);

        f_ecomps_ptr0[0] = 0;
        for (int inode = 0; inode < nnodes_fine; inode++) {
            f_ecomps_ptr0[inode + 1] = f_ecomps_ptr0[inode] + f_ecomps_cts[inode];
        }
        int f_ecomps_nnz0 = f_ecomps_ptr0[nnodes_fine - 1];
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

        printf("f_ecomps_cts v2: ");  // no repeats allowed
        printVec<int>(30, f_ecomps_cts);

        printf("f_ecomps_comp v1: ");
        printVec<int>(60, f_ecomps_comp0);

        // now that we've prevented repeats, adjust ptr size and write new comp and new nnz
        int *f_ecomps_ptr = new int[nnodes_fine + 1];
        f_ecomps_ptr[0] = 0;
        for (int inode = 0; inode < nnodes_fine; inode++) {
            f_ecomps_ptr[inode + 1] = f_ecomps_ptr[inode] + f_ecomps_cts[inode];
        }
        int f_ecomps_nnz = f_ecomps_ptr[nnodes_fine - 1];
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

        // now count number of unique..

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

            // should be ~24 elems examined per coarse node (can parallelize this on GPU if needed)
            for (int i_nn = 0; i_nn < nn; i_nn++) {
                int inode_c = nn_conn[nn * inode_f + i_nn];

                // get nn xpts for checking dzeta each element
                // T fine_nn_xpts[3 * nn_f];
                // get_nodal_xpts<T>(nn_f, &nn_conn_fine[nn_f * inode_f], h_xpts_fine,
                // fine_nn_xpts);

                for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                    int ielem_c = cnode_elems[jp];
                    // get coarse element component
                    int c_comp = h_coarse_elem_comps[ielem_c];

                    // int print_node = inode_f == 24064 || inode_f == 24095;
                    // if (print_node) {
                    //     printf(
                    //         "fine node %d connected to celem %d with c_comp %d\n\tfine node "
                    //         "comps:",
                    //         inode_f, ielem_c, c_comp);
                    // }

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

                    // // estimate of dzeta in nearest neighbors..
                    // T dzeta = get_dzeta_estimate<T, Basis>(nn_f, coarse_elem_xpts, fine_nn_xpts);
                    // // if (inode_f == 11133) {
                    // //     printf("node %d dzeta estimate %.2e\n", inode_f, dzeta);
                    // // }

                    // bool print_debug = inode_f == 1098;
                    bool print_debug = false;

                    T xi[3];
                    get_comp_coords<T, Basis>(coarse_elem_xpts, fine_node_xpts, xi, print_debug);

                    // temp DEBUG (see if rounding improves defects..)
                    // for (int m = 0; m < 2; m++) {
                    //     if (abs(xi[m]) < 1e-3) xi[m] = 0.0;
                    //     if (abs(xi[m] - 1) < 1e-2) xi[m] = 1.0;
                    //     if (abs(xi[m] + 1) < 1e-2) xi[m] = -1.0;
                    // }

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

                    // if (inode_f == 11133) {  //  || inode_f == 1097
                    //     // DEBUG
                    //     std::string result = node_in_elem ? "yes" : "no";
                    //     printf("fine node %d in coarse elem %d : %s\n", inode_f, ielem_c,
                    //            result.c_str());
                    //     printf("\txis: ");
                    //     printVec<T>(3, xi);

                    //     // get shell normal:
                    //     T fn[3];
                    //     ShellComputeCenterNormal<T, Basis>(coarse_elem_xpts, fn);
                    //     printf("\tshell normal: ");
                    //     printVec<T>(3, fn);
                    // }
                }
            }
        }

        // printf("-------\nfine_nodes_celem_cts: ");
        // printVec<int>(nnodes_fine, fine_nodes_celem_cts);
        // printf("-------\nfine nodes celems: ");
        // printVec<int>(ELEM_MAX * nnodes_fine, fine_nodes_celems);

        // for (int inode = 0; inode < nnodes_fine; inode++) {
        //     printf("fine node %d : celem_ct %d, celems ", inode, fine_nodes_celem_cts[inode]);
        //     for (int j = 0; j < 4; j++) {
        //         printf("%d ", fine_nodes_celems[4 * inode + j]);
        //     }
        //     printf("\n\t");
        //     for (int j = 0; j < 4; j++) {
        //         printf("xi%d %.2e eta%d %.2e,", j, fine_node_xis[8 * inode + 2 * j], j,
        //                fine_node_xis[8 * inode + 2 * j + 1]);
        //     }
        //     printf("\n");
        // }

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

        // printf("-------\nfine_node2elem_ptr: ");
        // printVec<int>(nnodes_fine + 1, fine_node2elem_ptr);
        // printf("-------\nfine_node2elem_elems: ");
        // printVec<int>(ntot_elems, fine_node2elem_elems);
        // printf("-------\nfine_node2elem_xis: ");
        // printVec<T>(2 * ntot_elems, fine_node2elem_xis);

        // for (int inode = 0; inode < nnodes_fine; inode++) {
        //     int start = fine_node2elem_ptr[inode];
        //     int ct = fine_node2elem_ptr[inode + 1] - start;
        //     printf("v2, fine node %d : celem_ct %d, celems ", inode, ct);
        //     for (int j = 0; j < ct; j++) {
        //         printf("%d ", fine_node2elem_elems[start + j]);
        //     }
        //     printf("\n\t");
        //     for (int j = 0; j < ct; j++) {
        //         printf("xi%d %.2e eta%d %.2e,", j, fine_node2elem_xis[2 * start + 2 * j], j,
        //                fine_node2elem_xis[2 * start + 2 * j + 1]);
        //     }
        //     printf("\n");
        // }

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

        // TBD: free up temp arrays
    }

    void initCuda() {
        // init handles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_defect = DeviceVec<T>(N);
        d_soln = DeviceVec<T>(N);
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_weights = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();
        d_int_temp = DeviceVec<int>(N).getPtr();

        // get perm pointers
        d_perm = Kmat.getPerm();
        d_iperm = Kmat.getIPerm();
        auto d_bsr_data = Kmat.getBsrData();
        d_elem_conn = d_bsr_data.elem_conn;
        nelems = d_bsr_data.nelems;

        // copy rhs into defect
        d_rhs.permuteData(block_dim, d_iperm);  // permute rhs to permuted form
        cudaMemcpy(d_defect.getPtr(), d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        // d_defect.permuteData(block_dim, d_iperm);

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

        // also init kmat for direct LU solves..
        if (full_LU) {
            d_kmat_lu_vals = DeviceVec<T>(Kmat.get_nnz()).getPtr();
            CHECK_CUDA(cudaMemcpy(d_kmat_lu_vals, d_kmat_vals, Kmat.get_nnz() * sizeof(T),
                                  cudaMemcpyDeviceToDevice));

            // ILU(0) factor on full LU pattern
            CUSPARSE::perform_ilu0_factorization(
                cusparseHandle, descr_kmat_L, descr_kmat_U, info_kmat_L, info_kmat_U, &kmat_pBuffer,
                nnodes, kmat_nnzb, block_dim, d_kmat_lu_vals, d_kmat_rowp, d_kmat_cols, trans_L,
                trans_U, policy_L, policy_U, dir);
        }
    }

    void initLowerMatForGaussSeidel() { /* init L+D matrix for lexigraphic or RCM Gauss-seidel */

        // init kmat descriptor for L+D matrix (no ilu0 factor, this is just the matrix itself
        // nofill)
        cusparseCreateMatDescr(&descr_kmat_L);
        cusparseSetMatIndexBase(descr_kmat_L, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr_kmat_L, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatFillMode(descr_kmat_L, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr_kmat_L, CUSPARSE_DIAG_TYPE_NON_UNIT);  // includes diag here..
        cusparseCreateBsrsv2Info(&info_kmat_L);

        // get buffer size
        int pbufferSize;
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
            cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, descr_kmat_L, d_kmat_vals, d_kmat_rowp,
            d_kmat_cols, block_dim, info_kmat_L, &pbufferSize));
        cudaMalloc(&kmat_pBuffer, pbufferSize);

        // compute symbolic analysis for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, nnodes, kmat_nnzb,
                                                descr_kmat_L, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                                block_dim, info_kmat_L, policy_L, kmat_pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void buildDiagInvMat() {
        // first need to construct rowp and cols for diagonal (fairly easy)
        int *h_diag_rowp = new int[nnodes + 1];
        diag_inv_nnzb = nnodes;
        int *h_diag_cols = new int[nnodes];
        h_diag_rowp[0] = 0;

        for (int i = 0; i < nnodes; i++) {
            h_diag_rowp[i + 1] = i + 1;
            h_diag_cols[i] = i;
        }

        // on host, get the pointer locations in Kmat of the block diag entries..
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // printf("h_color_rowp:");
        // printVec<int>(h_color_rowp.getSize() + 1, h_color_rowp.getPtr());
        // printf("nnodes %d\n", nnodes);
        // printf("h_kmat_rowp:");
        // printVec<int>(20, h_kmat_rowp);
        // printf("h_kmat_cols:");
        // printVec<int>(100, h_kmat_cols);

        // now copy to device
        d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
        d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

        // create the bsr data object on device
        auto d_diag_bsr_data =
            BsrData(nnodes, 6, diag_inv_nnzb, d_diag_rowp, d_diag_cols, nullptr, nullptr, false);
        delete[] h_diag_rowp;
        delete[] h_diag_cols;

        // now allocate DeviceVec for the values
        int ndiag_vals = block_dim * block_dim * nnodes;
        auto d_diag_vals = DeviceVec<T>(ndiag_vals);

        int *h_kmat_diagp = new int[nnodes];
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
                int block_col = h_kmat_cols[jp];
                // printf("row %d, col %d\n", block_row, block_col);
                if (block_row == block_col) {
                    h_kmat_diagp[block_row] = jp;
                }
            }
        }

        int *d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

        // call the kernel to copy out diag vals first
        dim3 block(32);
        int nblocks = (ndiag_vals + 31) / 32;
        dim3 grid(nblocks);
        k_copyBlockDiagFromBsrMat<T>
            <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_vals.getPtr());
        delete[] h_kmat_rowp;
        delete[] h_kmat_cols;

        // bool use_cublas = true;  // getriBatched very slow for in-place, LU batched not working
        // rn
        bool use_cublas = false;

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

        if (!use_cublas) {
            // use cusparse to get Dinv as LU factor and then we just do triang solves on
            // block - diag perform_ilu0_factorization();
            d_diag_LU_vals = d_diag_vals.getPtr();  // just copy these pointers..
            // printf("performing ILU(0) factor\n");

            CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                 trans_U, policy_L, policy_U, dir);
            // printf("did ILU(0) factor on block-diag D(K)\n");
        }

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

        // and make a BsrMat for it..
        D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_diag_vals);
        // printf("here3\n");
    }

    void buildColorLocalRowPointers() {
        // build local row pointers for row-slicing by color (of Kmat)
        // int *h_color_vals_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

        // init the color pointers
        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();  // says which rows in d_kmat_rowp are each color
        h_color_bnz_ptr =
            new int[num_colors + 1];  // says which block nz bounds for each color in cols, Kmat
        h_color_local_rowp_ptr =
            new int[num_colors + 1];  // pointer for bounds of d_color_local_rowps
        int *h_color_local_rowps = new int[nnodes + num_colors];

        // copy kmat pointers to host
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // build each pointer..
        h_color_bnz_ptr[0] = 0;
        h_color_local_rowp_ptr[0] = 0;
        int offset = 0;
        for (int icolor = 0; icolor < num_colors; icolor++) {
            int brow_start = color_rowp[icolor], brow_end = color_rowp[icolor + 1];
            int bnz_start = h_kmat_rowp[brow_start], bnz_end = h_kmat_rowp[brow_end];

            int nnzb_color = bnz_end - bnz_start;
            h_color_bnz_ptr[icolor + 1] = h_color_bnz_ptr[icolor] + nnzb_color;

            // now set the local rowp arrays for this color
            int nbrows_color = brow_end - brow_start;
            h_color_local_rowp_ptr[icolor + 1] = h_color_local_rowp_ptr[icolor] + nbrows_color + 1;
            h_color_local_rowps[offset] = 0;
            for (int local_row = 0; local_row < nbrows_color; local_row++) {
                int row_diff =
                    h_kmat_rowp[brow_start + local_row + 1] - h_kmat_rowp[brow_start + local_row];
                h_color_local_rowps[local_row + 1 + offset] =
                    h_color_local_rowps[local_row + offset] + row_diff;
            }
            offset += nbrows_color + 1;
        }

        delete[] h_kmat_rowp;
        delete[] h_kmat_cols;

        d_color_local_rowps =
            HostVec<int>(nnodes + num_colors, h_color_local_rowps).createDeviceVec().getPtr();
    }

    void direct_solve(bool print = false) {
        // T defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
        // printf("\tdirect solve, ||defect|| = %.4e\n", defect_nrm);

        // do a direct solve on the coarsest grid, with current defect.. (multicolor GS not
        // reliable) we keep permuted form, so I have to undo that from direct solve..
        // bool permute_inout = false;
        // CUSPARSE::direct_LU_solve<T>(Kmat, d_defect, d_soln, print, permute_inout);
        // this routine destroys and recreates handles .. may cause problems if calling multiple
        // times I think

        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, nnodes, kmat_nnzb,
                                             &alpha, descr_kmat_L, d_kmat_lu_vals, d_kmat_rowp,
                                             d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(),
                                             d_temp, policy_L, kmat_pBuffer));

        // triangular solve U*y = z
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes, kmat_nnzb,
                                             &alpha, descr_kmat_U, d_kmat_lu_vals, d_kmat_rowp,
                                             d_kmat_cols, block_dim, info_kmat_U, d_temp,
                                             d_soln.getPtr(), policy_U, kmat_pBuffer));

        // T soln_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_soln.getPtr(), 1, &soln_nrm));
        // printf("\tdirect solve, ||soln|| = %.4e\n", soln_nrm);
    }

    T getDefectNorm() {
        T def_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &def_nrm));
        return def_nrm;
    }

    T getResidNorm() {
        /* double check the linear system is actually solved */

        // copy rhs into the resid
        cudaMemcpy(d_resid, d_rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

        // subtract A * x where A = Kmat
        T a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_soln.getPtr(), &b, d_resid));

        // get resid nrm now
        T resid_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_nrm));
        return resid_nrm;
    }

    void smoothDefect(int n_iters, bool print = false, int print_freq = 10, T omega = 1.0,
                      bool rev_colors = false) {
        /* calls either multicolor smoother or lexigraphic GS depending on tempalte smoother type */
        if (smoother == MULTICOLOR_GS) {
            multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega, rev_colors);
        } else if (smoother == MULTICOLOR_GS_FAST) {
            multicolorBlockGaussSeidel_fast(n_iters, print, print_freq, omega, rev_colors);
        } else if (smoother == LEXIGRAPHIC_GS) {
            lexigraphicBlockGS(n_iters, print, print_freq);
        }
    }

    void lexigraphicBlockGS(int n_iters, bool print = false, int print_freq = 10) {
        // this is lexigraphic or RCM GS (RCM if more general mesh..)

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        T a, b;

        for (int iter = 0; iter < n_iters; iter++) {
            // 1) (L+D)*dx = defect with triang solve
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, &alpha, descr_kmat_L, d_kmat_vals,
                d_kmat_rowp, d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(), d_temp,
                policy_L, kmat_pBuffer));  // prob only need U^-1 part for block diag.. TBD

            // 2) update d_soln += d_temp (aka dx)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            // 3) compute new defect = prev_defect - A * dx
            a = -1.0,
            b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    void multicolorBlockGaussSeidel_slow(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool rev_colors = false) {
        // slower version of do multicolor BSRmat block gauss-seidel on the defect
        // slower in the sense that it uses full mat-vec and full triang solves (does work right)
        // would like a faster version with color slicing next..

        // T init_defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        // if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();

        // DEBUG
        // n_solns = n_iters * num_colors;
        // h_solns = new T *[n_solns];

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                // printf("\t\titer %d, color %d\n", iter, icolor);

                int _icolor2 = (_icolor + iter) % 4;  // permute order as you go
                // int _icolor2 = _icolor;  // no permutations.. about the same either way..

                int icolor = rev_colors ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                T *d_temp_color2 = &d_temp2[block_dim * start];
                cudaMemset(d_temp2, 0.0, N * sizeof(T));  // DEBUG
                cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * block_dim * sizeof(T),
                           cudaMemcpyDeviceToDevice);

                // T *h_defect = DeviceVec<T>(N, d_defect.getPtr()).createHostVec().getPtr();
                // printf("h_defect slow, color %d with grid nnodes = %d\n", icolor, nnodes);
                // printVec<T>(nrows_color, &h_defect[block_dim * start]);

                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, &alpha, descr_L,
                    d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2, d_resid,
                    policy_L, pBuffer));  // prob only need U^-1 part for block diag.. TBD

                // T *h_resid = DeviceVec<T>(N, d_resid).createHostVec().getPtr();
                // printf("h_resid_slow, color %d\n", icolor);
                // printVec<T>(nrows_color, &h_resid[block_dim * start]);

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                                     diag_inv_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                                     d_diag_rowp, d_diag_cols, block_dim, info_U,
                                                     d_resid, d_temp, policy_U, pBuffer));

                // T *h_temp = DeviceVec<T>(N, d_temp).createHostVec().getPtr();
                // printf("h_temp slow, color %d\n", icolor);
                // printVec<T>(nrows_color, &h_temp[block_dim * start]);

                // 2) update soln x_color += dx_color
                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega,
                b = 1.0;  // so that defect := defect - mat*vec
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

            }  // next color iteration

            // printf("iter %d, done with color iterations\n", iter);

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    void multicolorBlockGaussSeidel_fast(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool rev_colors = false) {
        // faster version

        // T init_defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        // if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        // printf("mc BGS-fast with # colors = %d\n", num_colors);

        // DEBUG
        // n_solns = n_iters * num_colors;
        // h_solns = new T *[n_solns];

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                // printf("\t\titer %d, color %d\n", iter, icolor);

                int _icolor2 = (_icolor + iter) % 4;  // permute order as you go
                // int _icolor2 = _icolor;  // no permutations.. about the same either way..

                int icolor = rev_colors ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                T *d_temp_color2 = &d_temp2[block_dim * start];
                cudaMemset(d_temp2, 0.0, N * sizeof(T));  // DEBUG
                cudaMemcpy(d_temp_color2, d_defect_color, nrows_color * sizeof(T),
                           cudaMemcpyDeviceToDevice);

                int block_dim2 = block_dim * block_dim;
                int diag_inv_nnzb_color = nblock_rows_color;
                T *d_diag_LU_vals_color = &d_diag_LU_vals[start * block_dim2];

                // T *h_defect = DeviceVec<T>(N, d_defect.getPtr()).createHostVec().getPtr();
                // printf("h_defect fast, color %d with nnodes %d\n", icolor, nnodes);
                // printVec<T>(nrows_color, &h_defect[block_dim * start]);

                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_L, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_L,
                    d_temp_color2, d_resid, policy_L,
                    pBuffer));  // prob only need U^-1 part for block diag.. TBD

                // T *h_resid = DeviceVec<T>(N, d_resid).createHostVec().getPtr();
                // printf("h_resid, color fast: %d\n", icolor);
                // printVec<T>(nrows_color, h_resid);
                // // printVec<T>(nrows_color, h_resid);

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_U, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_U,
                    d_resid, d_temp_color, policy_U, pBuffer));

                // T *h_temp = DeviceVec<T>(N, d_temp).createHostVec().getPtr();
                // printf("h_temp_fast, color %d\n", icolor);
                // printVec<T>(nrows_color, &h_temp[block_dim * start]);

                // 2) update soln x_color += dx_color

                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega,
                b = 1.0;  // so that defect := defect - mat*vec
                // can't row slice this to remove redundant color * 0 subblock computations
                // because K^T * vec not supported in cusparseDbsrmv routine.. see doc
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

            }  // next color iteration

            // printf("iter %d, done with color iterations\n", iter);

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    void prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in, int n_smooth = 0) {
        // prolongate from coarser grid to this fine grid
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        if constexpr (Prolongation::structured) {
            Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                     d_weights);
        } else {
            Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                     d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_temp_vec);
        d_temp_vec.permuteData(block_dim, d_iperm);

        // rescale coarse-fine using 1DOF min energy step
        // since FEA restrict and prolong operations are not energy minimally scaled
        // if u = u0 + omega * s, with s the proposed d_temp or du here (or line search)
        // then min energy omega from 1DOF galerkin is omega = <s, defect> / <s, Ks>
        // so need 2 dot prods, one SpMV, see 'multigrid/_python_demos/4_gmg_shell/1_mg.py' also
        T sT_defect;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_defect.getPtr(), 1, d_temp, 1, &sT_defect));

        if (n_smooth > 0) {
            // do smoothing on the coarse-fine update (kind of like AMG where you do smoothing on
            // the prolongation operator) except here it's less efficient => does it on the
            // prolongation step every time.. this will tell me whether AMG or some hybrid smoothing
            // of the prolongation operator of GMG too would help

            // copy current soln and defect into new temp vectors? then put these vecs in their
            // place to do MGS-defect?
        }

        T a = 1.0, b = 0.0;  // K * d_temp + 0 * d_temp2 => d_temp2
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));

        T sT_Ks;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));
        T omega = sT_defect / sT_Ks;
        // if (debug) printf("omega = %.2e\n", omega);

        // now add coarse-fine dx into soln and update defect (with u = u0 + omega * d_temp)
        a = omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        a = -omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_defect.getPtr(), 1));
    }

    void restrict_defect(int nelems_fine, int *d_iperm_fine, DeviceVec<T> fine_defect_in) {
        // transfer from finer mesh to this coarse mesh
        cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T));  // reset defect

        if constexpr (Prolongation::structured) {
            Prolongation::restrict_defect(nelems_fine, d_iperm, d_iperm_fine, fine_defect_in,
                                          d_defect, d_weights);
        } else {
            Prolongation::restrict_defect(restrict_nnodes_fine, d_elem_conn, restrict_d_n2e_ptr,
                                          restrict_d_n2e_elems, restrict_d_n2e_xis, d_iperm,
                                          d_iperm_fine, fine_defect_in, d_defect, d_weights);
        }

        // apply bcs to the defect again (cause it will accumulate on the boundary by backprop)
        // apply bcs is on un-permuted data
        d_defect.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_defect);
        d_defect.permuteData(block_dim, d_iperm);

        // reset soln (with bcs zero here, TBD others later)
        cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T));
    }

    void prolongate_debug(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in,
                          std::string file_prefix = "", std::string file_suffix = "",
                          int n_smooth = 0) {
        // call main prolongate
        if (n_smooth == 0) {
            prolongate(d_coarse_iperm, coarse_soln_in);
        } else {
            smoothed_prolongate(d_coarse_iperm, coarse_soln_in, n_smooth);
        }

        // DEBUG : write out the cf update, defect update and before and after defects
        auto h_cf_update = d_temp_vec.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift[3] = {0.0, -1.5, 1.5};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_update, file_prefix + "post2_cf_soln" + file_suffix, xpts_shift);

        auto h_cf_loads = DeviceVec<T>(N, d_temp2).createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift2[3] = {0.0, -1.5, 3.0};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_loads, file_prefix + "post3_cf_loads" + file_suffix, xpts_shift2);

        auto h_defect2 = d_defect.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift3[3] = {0.0, -1.5, 4.5};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_defect2, file_prefix + "post4_cf_fin_defect" + file_suffix, xpts_shift3);
    }

    void smoothed_prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in, int n_smooth = 0) {
        // prolongate from coarser grid to this fine grid (with a smoothing step, more for
        // debugging) if really need this (like an AMG hybrid step here), you should smooth
        // prolongation matrix itself first..
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        if constexpr (Prolongation::structured) {
            Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                     d_weights);
        } else {
            Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                     d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_temp_vec);
        d_temp_vec.permuteData(block_dim, d_iperm);

        // copy out the old defect into resid temporarily
        CHECK_CUDA(cudaMemcpy(d_resid, d_defect.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        // compute contributing defect and copy in temporarily replacing d_defect
        T a = -1.0, b = 0.0;  // -K * d_temp + 0 * d_defect => d_defect
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_defect.getPtr()));
        CHECK_CUDA(cudaMemcpy(d_soln.getPtr(), d_temp, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // before we coarse-fine rescale with 1DOF min energy the update.. let's do a smoothing of
        // the update
        multicolorBlockGaussSeidel_fast(n_smooth);

        // now copy back to d_temp and d_temp2 and old d_defect
        // CHECK_CUDA(cudaMemcpy(d_temp2, d_defect.getPtr(), N * sizeof(T),
        // cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_temp, d_soln.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_defect.getPtr(), d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // rescale coarse-fine using 1DOF min energy step
        // since FEA restrict and prolong operations are not energy minimally scaled
        // if u = u0 + omega * s, with s the proposed d_temp or du here (or line search)
        // then min energy omega from 1DOF galerkin is omega = <s, defect> / <s, Ks>
        // so need 2 dot prods, one SpMV, see 'multigrid/_python_demos/4_gmg_shell/1_mg.py' also
        T sT_defect;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_defect.getPtr(), 1, d_temp, 1, &sT_defect));

        a = 1.0, b = 0.0;  // K * d_temp + 0 * d_temp2 => d_temp2
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));

        T sT_Ks;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));
        T omega = sT_defect / sT_Ks;
        // if (debug) printf("omega = %.2e\n", omega);

        // now add coarse-fine dx into soln and update defect (with u = u0 + omega * d_temp)
        a = omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        a = -omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_defect.getPtr(), 1));
    }

    // data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_rhs, d_defect, d_soln, d_temp_vec;
    T *d_temp, *d_temp2, *d_resid, *d_weights;
    int *d_perm, *d_iperm;
    const int *d_elem_conn;
    HostVec<int> h_color_rowp;
    int *d_int_temp;

    // DEBUG
    int n_solns;
    T **h_solns;

    // unstruct prolong data at the finer mesh level (for prolong)
    // this is in someways describing the P matrix operator.. (could reformulate as that)
    int *d_n2e_ptr, *d_n2e_elems, *d_coarse_conn, n2e_nnz, ncoarse_elems;
    T *d_n2e_xis;

    // unstruct prolong at the coarser mesh level (for restrict)
    int *restrict_d_n2e_ptr, *restrict_d_n2e_elems, restrict_nnodes_fine, restrict_n2e_nnz;
    T *restrict_d_n2e_xis;

    // turn off private during debugging
   private:  // private data for cusparse and cublas
    // private data
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // color rowp and nnzb pointers data for row-slicing
    int *h_color_bnz_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    T *d_diag_vals, *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // and simiarly for Kmat a few differences
    bool full_LU;  // full LU only for coarsest mesh
    cusparseMatDescr_t descr_kmat_L = 0, descr_kmat_U = 0;
    bsrsv2Info_t info_kmat_L = 0, info_kmat_U = 0;
    void *kmat_pBuffer = 0;
};
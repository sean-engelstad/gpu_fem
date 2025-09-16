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
#include <set>
#include <chrono>

// for unstructured multigrid
#include "coupled/locate_point.h"

enum SMOOTHER : short {
    MULTICOLOR_GS,
    MULTICOLOR_GS_FAST,
    MULTICOLOR_GS_FAST2,
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
        // if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2) {
        //     buildColorLocalRowPointers();
        // }
        initCuda();
        if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2) {
            buildDiagInvMat();
            if (smoother == MULTICOLOR_GS_FAST2) buildTransposeColorMatrices();
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
            if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2) {
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
                    // bsr_data.AMD_reordering();
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
        auto start0 = std::chrono::high_resolution_clock::now();
        assembler.add_jacobian(res, kmat);
        assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = end0 - start0;
        printf("\tassemble kmat time %.2e\n", assembly_time.count());


        return new ShellGrid(assembler, N, kmat, loads, h_color_rowp, full_LU);
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
            int mb = nnodes_fine, nb = nnodes_coarse;
            int *h_perm = DeviceVec<int>(nnodes_fine, d_perm).createHostVec().getPtr();
            int *h_iperm = DeviceVec<int>(nnodes_fine, d_iperm).createHostVec().getPtr();
            int *h_coarse_iperm = DeviceVec<int>(nnodes_coarse, coarse_grid.d_iperm).createHostVec().getPtr();

            // compute the pattern here first, TODO is to clean this up and use less mem if possible at least free extra stuff
            int *h_prol_row_cts = new int[nnodes_fine];
            int *h_prolT_row_cts = new int[nnodes_coarse];
            memset(h_prol_row_cts, 0.0, nnodes_fine * sizeof(int));
            memset(h_prolT_row_cts, 0.0, nnodes_coarse * sizeof(int));
            // TODO : clean much of this up with sparse symbolic later.. though sparse symbolic uses connectivity, not generating non-square matrices usually
            // uses elem_conn which is like all of [1,2,7,8] connected, not the same as here where I have pairs [1] x [4,5,7,8] elements fine to coarse
            // maybe need to make some new methods in sparse symbolic.. (but let's get it running first)

            
            for (int inf = 0; inf < nnodes_fine; inf++) {
                int perm_inf = h_iperm[inf];
                int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
                std::set<int> conn_c_nodes;
                
                for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
                    int ielem_c = fine_node2elem_elems[jp];
                    const int *c_elem_nodes = &h_coarse_conn[4 * ielem_c];

                    // I might be including some extra sparsity for some nodes that are not contributing (like when fine node is on edge)
                    // TBD could come back later and decrease the size? would maybe double or 1.5x the nnz

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
            int *P_next = new int[nnodes_fine]; // helper arrays to insert values into sparsity
            memcpy(P_next, h_prol_rowp, nnodes_fine * sizeof(int));
            int *PT_next = new int[nnodes_coarse];
            memcpy(PT_next, h_prolT_rowp, nnodes_coarse * sizeof(int));
            // loop back through previous maps to get the sparsity
            // use perm_inf order here to help with PT sparsity
            for (int perm_inf = 0; perm_inf < nnodes_fine; perm_inf++) {
                int inf = h_perm[perm_inf];
                int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
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
                    h_prol_cols[P_next[perm_inf]++] = perm_inc; // P cols
                    h_prolT_cols[PT_next[perm_inc]++] = perm_inf; // PT cols
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
            // TODO : later is to just assemble P and then write my own transpose Bsrmv method in CUDA
            int P_block_dim = 1;
            // int P_block_dim = block_dim; // would lead to bsr here..

            int block_dim2 = P_block_dim * P_block_dim; // should be 36
            auto d_P_rowp = HostVec<int>(nnodes_fine + 1, h_prol_rowp).createDeviceVec().getPtr();
            auto d_P_cols = HostVec<int>(P_nnzb, h_prol_cols).createDeviceVec().getPtr();
            auto d_P_vals = DeviceVec<T>(block_dim2 * P_nnzb);
            auto P_bsr_data = BsrData(nnodes_fine, P_block_dim, P_nnzb, d_P_rowp, d_P_cols, d_perm, d_iperm, false);
            P_bsr_data.mb = nnodes_fine, P_bsr_data.nb = nnodes_coarse;
            P_bsr_data.rows = d_P_rows;

            P_mat = BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals);
            // TODO : technically I need to have BsrData capable of non-square matrices also

            auto d_PT_rowp = HostVec<int>(nnodes_coarse + 1, h_prolT_rowp).createDeviceVec().getPtr();
            auto d_PT_cols = HostVec<int>(PT_nnzb, h_prolT_cols).createDeviceVec().getPtr();
            auto d_PT_vals = DeviceVec<T>(block_dim2 * PT_nnzb);
            auto PT_bsr_data = BsrData(nnodes_coarse, P_block_dim, PT_nnzb, d_PT_rowp, d_PT_cols, 
                coarse_grid.d_perm, coarse_grid.d_iperm, false);
            PT_bsr_data.mb = nnodes_coarse, PT_bsr_data.nb = nnodes_fine;
            PT_bsr_data.rows = d_PT_rows; // for each nnz, which row is it (not same as rowp), helps for efficient mat-prods
            auto PT_mat = BsrMat<DeviceVec<T>>(PT_bsr_data, d_PT_vals); // only store this matrix on the coarse grid
            
            // printf("assemble matrices pre\n");
            printf("assemble P and PT matrices with nnodes_fine %d and nnodes_coarse %d\n", nnodes_fine, nnodes_coarse);
            Prolongation::assemble_matrices(d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, P_mat, PT_mat);
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
            CHECK_CUSPARSE(cusparseSetMatType(coarse_grid.restrict_descrPT, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(coarse_grid.restrict_descrPT, CUSPARSE_INDEX_BASE_ZERO));

        } // end of Prolongation::assembly case..

        // printf("done with init unstructured grid maps\n");
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> P_PT_time = end0 - start0;
        printf("unstructured grid P,PT comp in %.2e sec\n", P_PT_time.count());

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

        // use cusparse to get Dinv as LU factor and then we just do triang solves on
        // block - diag perform_ilu0_factorization();
        d_diag_LU_vals = d_diag_vals.getPtr();  // just copy these pointers..
        // printf("performing ILU(0) factor\n");

        CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                                d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                trans_U, policy_L, policy_U, dir);
        // printf("did ILU(0) factor on block-diag D(K)\n");

        // compute new Dinv_vals with LU operator..
        // build_lu_inv_operator = false; // previous version
        build_lu_inv_operator = smoother == MULTICOLOR_GS_FAST2 && !full_LU; // dense matrix should not modify LU vals on coarsest grid..

        if (build_lu_inv_operator) { 

            // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
            // to get effective matrix.. need six temp vectors..
            auto d_dinv_vals = DeviceVec<T>(ndiag_vals);

            for (int i = 0; i < block_dim; i++) {
                // set d_temp to ei (one of e1 through e6 per block)
                cudaMemset(d_temp, 0.0, N * sizeof(T));
                dim3 block(32);
                dim3 grid((nnodes + 31) / 32);
                k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

                // debug
                // T *h_temp = d_temp_vec.createHostVec().getPtr();
                // printf("h_temp step %d: ", i);
                // printVec<T>(12, h_temp);

                // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha,
                    descr_L, d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L,
                    d_temp, d_resid, policy_L,
                    pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha,
                    descr_U, d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_U,
                    d_resid, d_temp2, policy_U, pBuffer));

                // now copy temp2 into columns of new operator
                dim3 grid2((N + 31) / 32);
                k_setLUinv_operator<T><<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vals.getPtr());
            } // this works!

            D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_dinv_vals);
            d_diag_LU_vals = d_dinv_vals.getPtr(); // now LU vals technically holds Dinv form of LU as linear operator (no triang solves needed since only on diag)

        } else {
            D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_diag_vals);
        }
    }

    void buildTransposeColorMatrices() {
        // build color transpose matrices (one for each color)
        // only needed for fastest MC-BGS method..

        // compute the sparsity patterns of each col-subcolor matrix N x N_c (as N x N_c matrix)
        // stored in one data structure (basicaly each nodal block is stored in different order, with a few extra pointers than the Kmat)
        // kernels for some of these steps (initialization / assembly) may not be fully optimized yet..

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();  // says which rows in d_kmat_rowp are each color

        int **h_color_submat_rowp = new int*[num_colors];
        int **h_color_submat_rows = new int*[num_colors];
        int **h_color_submat_cols = new int*[num_colors];
        int *h_color_nnzb = new int[num_colors]; // what is the nnzb for each submat

        // copy kmat pointers to host
        int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // get sparsity for each color-sliced matrix
        for (int icolor = 0; icolor < num_colors; icolor++) {
            int start_node = color_rowp[icolor], end_node = color_rowp[icolor + 1];
            int ncols = end_node - start_node;
            int mb = nnodes, nb = ncols; // dimensions of column sub-matrix

            // construct a rowp and cols for each sub-matrix
            int *_row_cts = new int[nnodes];
            int *_rowp = new int[nnodes + 1];
            _rowp[0] = 0;
            for (int i = 0; i < nnodes; i++) {
                for (int jp = h_kmat_rowp[i], jp < h_kmat_rowp[i+1]; jp++) {
                    j = h_kmat_cols[jp];
                    if (start_node <= j && j < end_node) {
                        _row_cts[i]++;
                    }
                }
                _rowp[i+1] = _rowp[i] + _row_cts[i];
            }
            h_color_submat_rowp[icolor] = _rowp;

            int _nnzb = _rowp[nnodes];
            int *_rows = new int[_nnzb];
            for (int i = 0; i < nnodes; i++) {
                for (int jp = _rowp[i], jp < _rowp[i+1]; jp++) {
                    _rows[jp] = i;
                }
            }
            h_color_submat_rows[icolor] = _rows;

            int *_cols = new int[_nnzb];
            int *_next = new int[nnodes]; // help for inserting matrix
            memcpy(_next, _rowp, nnodes * sizeof(int));
            for (int i = 0; i < nnodes; i++) {
                for (int jp = h_kmat_rowp[i], jp < h_kmat_rowp[i+1]; jp++) {
                    j = h_kmat_cols[jp];
                    if (start_node <= j && j < end_node) {
                        j2 = j - start_node;
                        _cols[_next[i]++] = j2;
                    }
                }
            }
            delete[] _next;
            h_color_submat_cols[icolor] = _cols;
        }

        // now double check that, better way is maybe a transpose mat-vec product.. but that may not be as fast as you think?
        for (int icolor = 0; icolor < num_colors; icolor++) {
            // submat comes out of T **d_submat_vals then by color..

            // put pointers on device..
            k_copy_color_submat<T><<<grid, block>>>(nnodes, start, end, color_rows, color_cols, kmat_rowp, kmat_cols, kmat_vals, submat_vals);


            
        }

        // now copy kmat into the color sub-matrices..
        


        // 1) compute the sparsity patterns of the full transpose matrix
        // --------------------------------------------------------------
        // since sym matrix, the sparsity of transpose is the same, values the same right?

        // 2) copy values into the transpose matrix
        // ----------------------------------------
        // again from step 1, no need to do anything here..

        // 3) compute row-sliced (by color) rowp, cols of color submatrices (of transpose)
        // -------------------------------------------------------------------------------
        
        // do need the color-sliced rowp, cols (for only certain columns and a sub-matrix)
        // some parts here similar to next method buildColorLocalRowPointers
        // we do reuse a few states from that..



    }

    // void buildColorLocalRowPointers() {
    //     // build local row pointers for row-slicing by color (of Kmat)
    //     // int *h_color_vals_ptr, *h_color_local_rowp_ptr, *d_color_local_rowps;

    //     // init the color pointers
    //     int num_colors = h_color_rowp.getSize() - 1;
    //     int *color_rowp = h_color_rowp.getPtr();  // says which rows in d_kmat_rowp are each color
    //     h_color_bnz_ptr =
    //         new int[num_colors + 1];  // says which block nz bounds for each color in cols, Kmat
    //     h_color_local_rowp_ptr =
    //         new int[num_colors + 1];  // pointer for bounds of d_color_local_rowps
    //     int *h_color_local_rowps = new int[nnodes + num_colors];

    //     // copy kmat pointers to host
    //     int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
    //     int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

    //     // build each pointer..
    //     h_color_bnz_ptr[0] = 0;
    //     h_color_local_rowp_ptr[0] = 0;
    //     int offset = 0;
    //     for (int icolor = 0; icolor < num_colors; icolor++) {
    //         int brow_start = color_rowp[icolor], brow_end = color_rowp[icolor + 1];
    //         int bnz_start = h_kmat_rowp[brow_start], bnz_end = h_kmat_rowp[brow_end];

    //         int nnzb_color = bnz_end - bnz_start;
    //         h_color_bnz_ptr[icolor + 1] = h_color_bnz_ptr[icolor] + nnzb_color;

    //         // now set the local rowp arrays for this color
    //         int nbrows_color = brow_end - brow_start;
    //         h_color_local_rowp_ptr[icolor + 1] = h_color_local_rowp_ptr[icolor] + nbrows_color + 1;
    //         h_color_local_rowps[offset] = 0;
    //         for (int local_row = 0; local_row < nbrows_color; local_row++) {
    //             int row_diff =
    //                 h_kmat_rowp[brow_start + local_row + 1] - h_kmat_rowp[brow_start + local_row];
    //             h_color_local_rowps[local_row + 1 + offset] =
    //                 h_color_local_rowps[local_row + offset] + row_diff;
    //         }
    //         offset += nbrows_color + 1;
    //     }

    //     delete[] h_kmat_rowp;
    //     delete[] h_kmat_cols;

    //     d_color_local_rowps =
    //         HostVec<int>(nnodes + num_colors, h_color_local_rowps).createDeviceVec().getPtr();
    // }

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
        } else if (smoother == MULTICOLOR_GS_FAST2) {
            multicolorBlockGaussSeidel_fast2(n_iters, print, print_freq, omega, rev_colors);
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
                printf("\tLX-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

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

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {

                int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
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

                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, &alpha, descr_L,
                    d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2, d_resid,
                    policy_L, pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                                     diag_inv_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                                     d_diag_rowp, d_diag_cols, block_dim, info_U,
                                                     d_resid, d_temp, policy_U, pBuffer));

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

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        // printf("mc BGS-fast with # colors = %d\n", num_colors);

        bool time_debug = false;
        // bool time_debug = true;
        if (time_debug) printf("\t\tncolors = %d, #iters %d MC-BGS\n", num_colors, n_iters);

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {

                // -------------------------------------------------------------
                // prelim block (getting color sub-vectors ready)

                if (time_debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto prelim_time = std::chrono::high_resolution_clock::now();

                int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
                int icolor = rev_colors ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                int block_dim2 = block_dim * block_dim;
                int diag_inv_nnzb_color = nblock_rows_color;
                T *d_diag_LU_vals_color = &d_diag_LU_vals[start * block_dim2];

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_prelim_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> full_prelim_time = end_prelim_time - prelim_time;
                    printf("\t\tprelim time on iter %d,color %d in %.2e sec\n", iter, icolor, full_prelim_time.count());
                }
                auto start_Dinv_LU_tmie = std::chrono::high_resolution_clock::now();

                // --------------------------------------------------------------
                // apply Dinv * vec on each color sub-vecotr
                // use LU triang solves to apply Dinv * vec on each color (old method)
                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_L, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_L,
                    d_defect_color, d_resid, policy_L,
                    pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_U, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_U,
                    d_resid, d_temp_color, policy_U, pBuffer));

                // timing part
                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Dinv_LU_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> Dinv_LU_time = end_Dinv_LU_time - start_Dinv_LU_tmie;
                    printf("\t\tDinv LU time on iter %d,color %d in %.2e sec\n", iter, icolor, Dinv_LU_time.count());
                }

                // -----------------------------------------------------------------
                // color soln update => defect update for each color
                
                auto start_Bsrmv_time = std::chrono::high_resolution_clock::now();

                // 2) update soln x_color += dx_color

                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega, b = 1.0;  // so that defect := defect - mat*vec
                // this does full mat-vec product, so much slower..
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Bsrmv_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> bsrmv_time = end_Bsrmv_time - start_Bsrmv_time;
                    printf("\t\tbsrmv time on iter %d,color %d in %.2e sec\n", iter, icolor, bsrmv_time.count());
                }
                
            // -------------------------------------------------------------------------------------
            }  // next color iteration

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        // --------------------------------------------------------------------------------
        }  // next block-GS iteration
    }

    void multicolorBlockGaussSeidel_fast2(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool rev_colors = false) {
        // faster version

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        // printf("mc BGS-fast with # colors = %d\n", num_colors);

        bool time_debug = false;
        // bool time_debug = true;
        if (time_debug) printf("\t\tncolors = %d, #iters %d MC-BGS\n", num_colors, n_iters);

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {

                // -------------------------------------------------------------
                // prelim block (getting color sub-vectors ready)

                if (time_debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto prelim_time = std::chrono::high_resolution_clock::now();

                int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
                int icolor = rev_colors ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                int block_dim2 = block_dim * block_dim;
                int diag_inv_nnzb_color = nblock_rows_color;
                T *d_diag_LU_vals_color = &d_diag_LU_vals[start * block_dim2];

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_prelim_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> full_prelim_time = end_prelim_time - prelim_time;
                    printf("\t\tprelim time on iter %d,color %d in %.2e sec\n", iter, icolor, full_prelim_time.count());
                }

                // --------------------------------------------------------------
                // apply Dinv * vec on each color sub-vector
                
                auto start_Dinv_LU_tmie = std::chrono::high_resolution_clock::now();
                // use Dinv linear operator built from LU factor of D diag matrix to apply Dinv * vec on each color (new method)
                T a = 1.0, b = 0.0;
                // note in this case d_diag_LU_vals_color refers to Dinv form of LU factors on each nodal block
                CHECK_CUSPARSE(cusparseDbsrmv( // NOTE just uses descrKmat cause would be the same as descrDinv (convenience)
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nblock_rows_color, nblock_rows_color, diag_inv_nnzb_color, &a, descrKmat, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols,
                    block_dim, d_defect_color, &b, d_temp_color));

                // timing part
                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Dinv_LU_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> Dinv_LU_time = end_Dinv_LU_time - start_Dinv_LU_tmie;
                    printf("\t\tDinv LU time on iter %d,color %d in %.2e sec\n", iter, icolor, Dinv_LU_time.count());
                }

                // -----------------------------------------------------------------
                // color soln update => defect update for each color
                
                auto start_Bsrmv_time = std::chrono::high_resolution_clock::now();

                // 2) update soln x_color += dx_color

                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega, b = -1.0;  // so that defect := defect - mat*vec

                // TODO : need sliced products here by color (maybe Bsrmv transpose product)
                // transposes here..

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Bsrmv_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> bsrmv_time = end_Bsrmv_time - start_Bsrmv_time;
                    printf("\t\tbsrmv time on iter %d,color %d in %.2e sec\n", iter, icolor, bsrmv_time.count());
                }
                
            // -------------------------------------------------------------------------------------
            }  // next color iteration

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        // --------------------------------------------------------------------------------
        }  // next block-GS iteration
    }

    void prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in) {
        // prolongate from coarser grid to this fine grid
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        if constexpr (Prolongation::structured) {
            Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                     d_weights);
        } else {
            if constexpr (Prolongation::assembly) {
                Prolongation::prolongate(cusparseHandle, descrP, P_mat, coarse_soln_in, d_temp_vec);
            } else {
                // slower version
                Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                     d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
            }
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
        // printf("start restrict defect\n");

        if constexpr (Prolongation::structured) {
            Prolongation::restrict_defect(nelems_fine, d_iperm, d_iperm_fine, fine_defect_in,
                                          d_defect, d_weights);
        } else {
            if constexpr (Prolongation::assembly) {
                Prolongation::restrict_defect(cusparseHandle, restrict_descrPT, restrict_PT_mat, fine_defect_in, d_defect);
            } else {
                // slower restriction operator (not using sparse matrix-vec kernels)
                Prolongation::restrict_defect(restrict_nnodes_fine, d_elem_conn, restrict_d_n2e_ptr,
                                          restrict_d_n2e_elems, restrict_d_n2e_xis, d_iperm,
                                          d_iperm_fine, fine_defect_in, d_defect, d_weights);
            }
        }

        // apply bcs to the defect again (cause it will accumulate on the boundary by backprop)
        // apply bcs is on un-permuted data
        d_defect.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_defect);
        d_defect.permuteData(block_dim, d_iperm);

        // printf("end restrict defect\n");

        // reset soln (with bcs zero here, TBD others later)
        cudaMemset(d_soln.getPtr(), 0.0, N * sizeof(T));
    }

    void prolongate_debug(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in,
                          std::string file_prefix = "", std::string file_suffix = "",
                          int n_smooth = 0, T y_offset = -1.5) {
        // call main prolongate
        if (n_smooth == 0) {
            prolongate(d_coarse_iperm, coarse_soln_in);
        } else {
            smoothed_prolongate(d_coarse_iperm, coarse_soln_in, n_smooth);
        }

        // DEBUG : write out the cf update, defect update and before and after defects
        auto h_cf_update = d_temp_vec.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift[3] = {0.0, y_offset, 1.5};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_update, file_prefix + "post2_cf_soln" + file_suffix, xpts_shift);

        auto h_cf_loads = DeviceVec<T>(N, d_temp2).createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift2[3] = {0.0, y_offset, 3.0};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_loads, file_prefix + "post3_cf_loads" + file_suffix, xpts_shift2);

        auto h_defect2 = d_defect.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift3[3] = {0.0, y_offset, 4.5};
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
            if constexpr (Prolongation::assembly) {
                // permute coarse soln in
                coarse_soln_in.permuteData(block_dim, d_coarse_iperm);
                Prolongation::prolongate(cusparseHandle, descrP, P_mat, coarse_soln_in, d_temp_vec);
            } else {
                // slower version
                Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                     d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
            }
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
    cusparseMatDescr_t descrP = 0;
    BsrMat<DeviceVec<T>> P_mat;

    // unstruct prolong at the coarser mesh level (for restrict)
    int *restrict_d_n2e_ptr, *restrict_d_n2e_elems, restrict_nnodes_fine, restrict_n2e_nnz;
    T *restrict_d_n2e_xis;
    cusparseMatDescr_t restrict_descrPT = 0;
    BsrMat<DeviceVec<T>> restrict_PT_mat;

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
    bool build_lu_inv_operator;

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
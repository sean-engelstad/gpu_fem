#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_fetidp.cuh"
#include "cuda_utils.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "multigrid/solvers/solve_utils.h"

enum NodeClass { INTERIOR = 0, DIRICHLET_EDGE = 1, EDGE = 2, VERTEX = 3 };

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_>
class FetidpSolver : public BaseSolver {
   public:
    using ShellAssembler = ShellAssembler_;
    using Director = typename ShellAssembler::Director;
    using Basis = typename ShellAssembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename ShellAssembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Vec = Vec_<T>;
    using Mat = Mat_<Vec_<T>>;
    using BsrMatType = BsrMat<DeviceVec<T>>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    FetidpSolver() = default;

    FetidpSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                 ShellAssembler &assembler_, BsrMatType &kmat_)
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          assembler(assembler_),
          subdomainISolver(nullptr),
          subdomainIESolver(nullptr),
          coarseSolver(nullptr),
          elem_sd_ind(nullptr),
          elem_conn(nullptr),
          node_sd_cols(nullptr),
          node_elem_ct(nullptr),
          node_elem_rowp(nullptr),
          kmat(&kmat_),
          node_class_ind(nullptr) {
        num_elements = assembler.get_num_elements();
        num_nodes = assembler.get_num_nodes();
        N = num_nodes * vars_per_node;

        elem_conn = assembler.getConn().createHostVec().getPtr();

        auto kmat_bsr_data = kmat->getBsrData();
        kmat_nnzb = kmat_bsr_data.nnzb;
        block_dim = kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;

        d_xpts = assembler.getXpts();
        d_vars = assembler.getVars();
        d_elem_components = assembler.getElemComponents();
        d_compData = assembler.getCompData();

        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));
    }

    // required routines to be a BaseSolver (needed to be preconditioner)
    void update_after_assembly(DeviceVec<T> &vars) {  // TODO
    }
    void factor() override {  // TODO
    }
    void set_print(bool print) {}
    void set_rel_tol(T rtol) {}
    void set_abs_tol(T atol) {}
    int get_num_iterations() { return 1; }
    void set_cycle_type(std::string cycle_) {}
    void free() {  // TODO
    }

    ~FetidpSolver() { clear_host_data(); }
    int getLambdaSize() const { return lam_nnodes * block_dim; }

    void setup_structured_subdomains(int nxe_, int nye_, int nxs_, int nys_,
                                     bool close_hoop = false) {
        clear_structured_host_data();

        nxe = nxe_;
        nye = nye_;
        nxs = nxs_;
        nys = nys_;

        nx = nxe + 1;
        ny = close_hoop ? nye : (nye + 1);

        num_subdomains = nxs * nys;

        const int nxse = (nxe + nxs - 1) / nxs;
        const int nyse = (nye + nys - 1) / nys;

        // -----------------------------------------
        // classify elements into subdomains
        // -----------------------------------------
        elem_sd_ind = new int[num_elements];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int ixe = ielem % nxe;
            int iye = ielem / nxe;
            int ixs = ixe / nxse;
            int iys = iye / nyse;

            if (ixs >= nxs) ixs = nxs - 1;
            if (iys >= nys) iys = nys - 1;

            elem_sd_ind[ielem] = ixs + iys * nxs;
        }

        // -----------------------------------------
        // node -> subdomain incidence
        // -----------------------------------------
        node_elem_nnz = 0;
        node_elem_rowp = new int[num_nodes + 1];
        node_elem_ct = new int[num_nodes];
        std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
        std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        int *temp_node_elem = new int[num_nodes];
        node_sd_cols = new int[node_elem_nnz];
        std::memset(temp_node_elem, 0, num_nodes * sizeof(int));
        std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int subdomain_ind = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;

        // -----------------------------------------
        // classify nodes
        // -----------------------------------------
        node_class_ind = new int[num_nodes];
        std::memset(node_class_ind, 0, num_nodes * sizeof(int));

        nnodes_interior = 0;
        nnodes_edge = 0;
        nnodes_vertex = 0;
        nnodes_dirichlet_edge = 0;

        printf(
            "WARNING: for unstructured meshes, this node classification may fail at junctions.\n");

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int ix = inode % nx;
            int iy = inode / nx;
            bool on_x = (ix == 0) || (ix == nx - 1);
            bool on_y = ((iy == 0) || (iy == ny - 1)) && !close_hoop;
            bool on_bndry = on_x || on_y;

            int nsd = static_cast<int>(node_sds.size());

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;
                nnodes_interior++;
            } else if (nsd == 2) {
                if (on_bndry) {
                    node_class_ind[inode] = DIRICHLET_EDGE;
                    nnodes_dirichlet_edge++;
                } else {
                    node_class_ind[inode] = EDGE;
                    nnodes_edge++;
                }
            } else {
                node_class_ind[inode] = VERTEX;
                nnodes_vertex++;
            }
        }

        // TODO:
        // This count assumes structured duplication counts:
        //   interior            -> 1 copy
        //   dirichlet-edge      -> 2 copies
        //   edge                -> 2 copies
        //   vertex              -> 4 copies
        // Revisit for more general cases.
        I_nnodes = nnodes_interior + 2 * nnodes_dirichlet_edge;
        IE_nnodes = I_nnodes + 2 * nnodes_edge;
        IEV_nnodes = IE_nnodes + 4 * nnodes_vertex;
        Vc_nnodes = nnodes_vertex;     // vertex non-repeated (here for coarse system)
        V_nnodes = 4 * nnodes_vertex;  // vertex repeated
        lam_nnodes = nnodes_edge;      // FETI lagrange multipliers

        // -----------------------------------------
        // build duplicated IEV nodal layout
        // -----------------------------------------
        IEV_sd_ptr = new int[num_subdomains + 1];
        IEV_sd_ind = new int[IEV_nnodes];
        IEV_nodes = new int[IEV_nnodes];

        std::memset(IEV_sd_ptr, 0, (num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[num_nodes];

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::memset(temp_completion, 0, num_nodes * sizeof(int));
            IEV_sd_ptr[i_subdomain + 1] = IEV_sd_ptr[i_subdomain];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != i_subdomain) continue;

                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    if (temp_completion[gnode]) continue;

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = i_subdomain;
                    IEV_sd_ptr[i_subdomain + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;

        // printf("IEV_sd_ptr: ");
        // printVec<int>(num_subdomains + 1, IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_nnodes, IEV_sd_ind);

        // -----------------------------------------
        // build IEV element connectivity
        // -----------------------------------------
        IEV_elem_conn = new int[num_elements * nodes_per_elem];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int i_subdomain = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[i_subdomain]; jp < IEV_sd_ptr[i_subdomain + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                }
                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }

        // printf("IEV_ind %d\n", IEV_ind);
        // printf("IEV_nodes %d: ", IEV_nnodes);
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("IEV_conn: ");
        // printVec<int>(nodes_per_elem * num_elements, IEV_elem_conn);

        // -----------------------------------------
        // IE and I nodal lists
        // -----------------------------------------
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];
        std::memset(IE_nodes, 0, IE_nnodes * sizeof(int));
        std::memset(I_nodes, 0, I_nnodes * sizeof(int));
        IE_interior = new bool[IE_nnodes];
        IE_general_edge = new bool[IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                I_nodes[I_ind++] = gnode;
            }
        }
        d_IE_interior = HostVec<bool>(IE_nnodes, IE_interior).createDeviceVec().getPtr();
        d_IE_general_edge = HostVec<bool>(IE_nnodes, IE_general_edge).createDeviceVec().getPtr();
        d_IE_nodes = HostVec<int>(IE_nnodes, IE_nodes).createDeviceVec().getPtr();

        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("I_nodes %d: ", I_nnodes);
        // printVec<int>(I_nnodes, I_nodes);

        // -----------------------------------------
        // build IEV sparsity from duplicated connectivity
        // -----------------------------------------
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);
        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;
        IEV_rows = new int[IEV_nnzb];
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            for (int jp = IEV_rowp[inode]; jp < IEV_rowp[inode + 1]; jp++) {
                IEV_rows[jp] = inode;
            }
        }
        // printf("IEV_rowp with nnzb %d: ", IEV_nnzb);
        // printVec<int>(IEV_nnodes + 1, IEV_rowp);
        // printf("IEV_cols: ");
        // printVec<int>(IEV_nnzb, IEV_cols);

        // -----------------------------------------
        // reduced rowp arrays
        // -----------------------------------------
        IE_rowp = new int[IE_nnodes + 1];
        I_rowp = new int[I_nnodes + 1];
        std::memset(IE_rowp, 0, (IE_nnodes + 1) * sizeof(int));
        std::memset(I_rowp, 0, (I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) I_rowp[I_row + 1] = I_rowp[I_row];
            if (typeIE_row) IE_rowp[IE_row + 1] = IE_rowp[IE_row];

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        I_nnzb = I_rowp[I_nnodes];
        IE_nnzb = IE_rowp[IE_nnodes];

        // printf("I_rowp nnzb %d: ", I_nnzb);
        // printVec<int>(I_nnodes + 1, I_rowp);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_rowp nnzb %d: ", IE_nnzb);
        // printVec<int>(IE_nnodes + 1, IE_rowp);

        IE_rows = new int[IE_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }

        // -----------------------------------------
        // IEV -> IE map
        // -----------------------------------------
        IEVtoIE_map = new int[IEV_nnodes];
        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        IEVtoIE_imap = new int[IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IEVtoIE_imap[IE_ind] = inode;
                IEVtoIE_map[inode] = IE_ind++;
            }
        }

        // printf("IEVtoIE_map: ");
        // printVec<int>(IEV_nnodes, IEVtoIE_map);

        // put on device
        d_IEVtoIE_imap = HostVec<int>(IE_nnodes, IEVtoIE_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV -> I map
        // -----------------------------------------
        IEVtoI_map = new int[IEV_nnodes];
        IEVtoI_imap = new int[I_nnodes];
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IEVtoI_imap[I_ind] = inode;
                IEVtoI_map[inode] = I_ind++;
            }
        }

        // printf("IEVtoI_map: ");
        // printVec<int>(IEV_nnodes, IEVtoI_map);

        // put on device
        d_IEVtoI_imap = HostVec<int>(I_nnodes, IEVtoI_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // reduced column arrays
        // -----------------------------------------
        IE_cols = new int[IE_nnzb];
        I_cols = new int[I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    I_cols[I_ind++] = IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    IE_cols[IE_ind++] = IEVtoIE_map[col];
                }
            }
        }

        // printf("I_cols: ");
        // printVec<int>(I_nnzb, I_cols);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();

        // -----------------------------------------
        // IEV -> Vc map (coarse non-repeated vertices)
        // -----------------------------------------

        // make a list of the reduced Vc_ind (takes global node and figures out it's Vc_ind)
        std::unordered_set<int> Vc_nodeset;
        for (int i = 0; i < IEV_nnodes; i++) {
            int gnode = IEV_nodes[i];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());
        // printf("Vc_nodes %d: ", Vc_nodes_vec.size());
        // printVec<int>(Vc_nodes_vec.size(), Vc_nodes_vec.data());

        int *Vc_inodes = new int[num_nodes];  // takes Vc global node => Vc red node
        memset(Vc_inodes, -1, num_nodes * sizeof(int));
        for (int i = 0; i < Vc_nodes_vec.size(); i++) {
            int j = Vc_nodes_vec[i];
            Vc_inodes[j] = i;
        }

        // printf("Vc_inodes: ");
        // printVec<int>(num_nodes, Vc_inodes);

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();

        // set VcToV_imap and IEVtoV_imap now
        VctoV_imap = new int[V_nnodes];
        std::memset(VctoV_imap, -1, V_nnodes * sizeof(int));
        IEVtoV_imap = new int[V_nnodes];
        std::memset(IEVtoV_imap, -1, V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                VctoV_imap[V_ind] = Vc_ind;
                IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }

        // printf("IEVtoV_imap %d: ", V_nnodes);
        // printVec<int>(V_nnodes, IEVtoV_imap);

        // put on device
        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();

        // Build all remaining maps/permutations:
        //  - jump operator ownership/sign maps

        _compute_jump_operators();
        allocate_workspace();

        // ---------------------------------------------------
        // Save ORIGINAL nofill sparsity for IE and I
        // before fill-in / AMD reordering
        // ---------------------------------------------------
        IE_bsr_data = BsrData(IE_nnodes, block_dim, IE_nnzb, IE_rowp, IE_cols);
        IE_bsr_data.rows = IE_rows;
        IE_nofill_nnzb = IE_nnzb;

        I_bsr_data = BsrData(I_nnodes, block_dim, I_nnzb, I_rowp, I_cols);
        I_bsr_data.rows = I_rows;
        I_nofill_nnzb = I_nnzb;
    }

    void setup_matrix_sparsity() {
        // USER must call this routine..
        printf(
            "NOTE : FETI-DP doesn't support permutations yet on subdomains.. TBD later on that\n");
        printf("\tJust does full fillin currently of each matrix used for inner linear solves\n");

        // do fillin of IE and I matrices (later also do coarse matrix)
        IE_rowp = IE_bsr_data.rowp, IE_cols = IE_bsr_data.cols, IE_nnzb = IE_bsr_data.nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        // host
        I_rowp = I_bsr_data.rowp, I_cols = I_bsr_data.cols, I_nnzb = I_bsr_data.nnzb;
        I_perm = I_bsr_data.perm, I_iperm = I_bsr_data.iperm;

        // recompute rows after potential fillin
        delete[] IE_rows, I_rows;
        IE_rows = new int[IE_nnzb];
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }
        IE_bsr_data.rows = IE_rows;
        I_bsr_data.rows = I_rows;

        // now move sparsity to the device
        // and no fillin for IEV matrix cause it isn't used for linear solves
        IEV_bsr_data.rows = IEV_rows;
        d_IEV_bsr_data = IEV_bsr_data.createDeviceBsrData();
        d_IEV_vals = DeviceVec<T>(block_dim2 * IEV_nnzb);
        kmat_IEV = new BsrMatType(d_IEV_bsr_data, d_IEV_vals);

        d_IE_bsr_data = IE_bsr_data.createDeviceBsrData();
        d_IE_vals = DeviceVec<T>(block_dim2 * IE_nnzb);
        kmat_IE = new BsrMatType(d_IE_bsr_data, d_IE_vals);

        d_I_bsr_data = I_bsr_data.createDeviceBsrData();
        d_I_vals = DeviceVec<T>(block_dim2 * I_nnzb);
        kmat_I = new BsrMatType(d_I_bsr_data, d_I_vals);

        d_IE_perm = d_IE_bsr_data.perm, d_IE_iperm = d_IE_bsr_data.iperm;
        d_I_perm = d_I_bsr_data.perm, d_I_iperm = d_I_bsr_data.iperm;

        // -----------------------------------------
        // IEV => IE kmat block copy map
        // -----------------------------------------

        kmat_IEnofill_map = new int[IE_nofill_nnzb];
        kmat_IEtoIEV_map = new int[IE_nofill_nnzb];
        memset(kmat_IEnofill_map, -1, IE_nofill_nnzb * sizeof(int));
        memset(kmat_IEtoIEV_map, -1, IE_nofill_nnzb * sizeof(int));
        int nofill_ind = 0;
        for (int i = 0; i < IE_nnodes; i++) {
            int i_perm = IE_iperm[i];  // VIS to solve order
            for (int jp = IE_rowp[i_perm]; jp < IE_rowp[i_perm + 1]; jp++) {
                int j_perm = IE_cols[jp];
                int j = IE_perm[j_perm];  // solve to VIS order
                // find equivalent nz block of IEV rowp
                int i_IEV = IEVtoIE_imap[i];
                int j_IEV = IEVtoIE_imap[j];
                bool found = false;
                for (int kp = IEV_rowp[i_IEV]; kp < IEV_rowp[i_IEV + 1]; kp++) {
                    int k = IEV_cols[kp];
                    if (k == j_IEV) {
                        kmat_IEnofill_map[nofill_ind] = jp;
                        kmat_IEtoIEV_map[nofill_ind] = kp;

                        int gr_IE = IE_nodes[i], gc_IE = IE_nodes[j];
                        int gr_IEV = IEV_nodes[i_IEV], gc_IEV = IEV_nodes[j_IEV];
                        // printf("nofill ind %d, IE (%d,%d) block %d, IEV (%d,%d) block %d\n",
                        //        nofill_ind, gr_IE, gc_IE, jp, gr_IEV, gc_IEV, kp);
                        nofill_ind++;
                        found = true;
                    }
                }
                // would need more advanced check here since fillin means some blocks in IE
                // won't exist in IEV
                // if (!found) {
                //     printf("IE block %d not found in IEV matrix\n", jp);
                // }
            }
        }
        d_kmat_IEtoIEV_map =
            HostVec<int>(IE_nofill_nnzb, kmat_IEtoIEV_map).createDeviceVec().getPtr();
        d_kmat_IEnofill_map =
            HostVec<int>(IE_nofill_nnzb, kmat_IEnofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV => I kmat block copy map
        // -----------------------------------------

        kmat_Inofill_map = new int[I_nofill_nnzb];
        kmat_ItoIEV_map = new int[I_nofill_nnzb];
        memset(kmat_Inofill_map, -1, I_nofill_nnzb * sizeof(int));
        memset(kmat_ItoIEV_map, -1, I_nofill_nnzb * sizeof(int));
        nofill_ind = 0;
        for (int i = 0; i < I_nnodes; i++) {
            int i_perm = I_iperm[i];  // VIS to solve order
            for (int jp = I_rowp[i_perm]; jp < I_rowp[i_perm + 1]; jp++) {
                int j_perm = I_cols[jp];
                int j = I_perm[j_perm];
                // find equivalent nz block of IEV rowp
                int i_IEV = IEVtoI_imap[i];
                int j_IEV = IEVtoI_imap[j];
                bool found = false;
                for (int kp = IEV_rowp[i_IEV]; kp < IEV_rowp[i_IEV + 1]; kp++) {
                    int k = IEV_cols[kp];
                    if (k == j_IEV) {
                        kmat_Inofill_map[nofill_ind] = jp;
                        kmat_ItoIEV_map[nofill_ind] = kp;
                        nofill_ind++;
                        found = true;
                    }
                }
                // would need more advanced check here since fillin means some blocks in IE
                // won't exist in IEV
                // if (!found) {
                //     printf("IE block %d not found in IEV matrix\n", jp);
                // }
            }
        }
        d_kmat_ItoIEV_map = HostVec<int>(I_nofill_nnzb, kmat_ItoIEV_map).createDeviceVec().getPtr();
        d_kmat_Inofill_map =
            HostVec<int>(I_nofill_nnzb, kmat_Inofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // get S_VV matrix sparsity / nonzero pattern (nofill first)
        // -----------------------------------------

        // -----------------------------------------
        // get S_VV matrix sparsity / nonzero pattern (unique structure only)
        // -----------------------------------------

        // reverse map of global => reduced Vc nodes
        Vc_node_imap = new int[num_nodes];
        memset(Vc_node_imap, -1, num_nodes * sizeof(int));
        for (int vnode = 0; vnode < Vc_nnodes; vnode++) {
            int glob_node = Vc_nodes[vnode];
            Vc_node_imap[glob_node] = vnode;
        }

        // printf("Vc_node_imap: ");
        // printVec<int>(num_nodes, Vc_node_imap);

        // build unique adjacency per coarse row
        std::vector<std::unordered_set<int>> Svv_adj(Vc_nnodes);

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::unordered_set<int> sd_Vc_nodeset;

            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_nodes = &elem_conn[nodes_per_elem * ielem];
                int j_subdomain = elem_sd_ind[ielem];
                if (i_subdomain != j_subdomain) continue;

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_nodes[lnode];
                    int node_class = node_class_ind[gnode];
                    if (node_class == VERTEX) {
                        int vnode = Vc_node_imap[gnode];
                        if (vnode >= 0) {
                            sd_Vc_nodeset.insert(vnode);
                        }
                    }
                }
            }

            std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());

            // add unique couplings for this subdomain
            for (int i : sd_Vc_nodes) {
                for (int j : sd_Vc_nodes) {
                    Svv_adj[i].insert(j);
                }
            }
        }

        // row counts from unique adjacency
        int *Svv_rowcts = new int[Vc_nnodes];
        memset(Svv_rowcts, 0, Vc_nnodes * sizeof(int));
        for (int i = 0; i < Vc_nnodes; i++) {
            Svv_rowcts[i] = static_cast<int>(Svv_adj[i].size());
        }

        // fill rowp
        Svv_rowp = new int[Vc_nnodes + 1];
        memset(Svv_rowp, 0, (Vc_nnodes + 1) * sizeof(int));
        for (int i = 0; i < Vc_nnodes; i++) {
            Svv_rowp[i + 1] = Svv_rowp[i] + Svv_rowcts[i];
        }
        Svv_nnzb = Svv_rowp[Vc_nnodes];

        // fill cols
        Svv_cols = new int[Svv_nnzb];
        memset(Svv_cols, 0, Svv_nnzb * sizeof(int));

        for (int i = 0; i < Vc_nnodes; i++) {
            int jp = Svv_rowp[i];
            for (int j : Svv_adj[i]) {
                Svv_cols[jp++] = j;
            }

            // optional but recommended: sort each row
            std::sort(&Svv_cols[Svv_rowp[i]], &Svv_cols[Svv_rowp[i + 1]]);
        }

        // printf("Svv_rowp with nnzb %d: ", Svv_nnzb);
        // printVec<int>(Vc_nnodes + 1, Svv_rowp);
        // printf("Svv_cols: ");
        // printVec<int>(Svv_nnzb, Svv_cols);

        // now go back through and put cols in
        // use temp_Svv_fill to help keep track of putting nonzero entries in the Svv sparsity
        // int *temp_Svv_fill = new int[Vc_nnodes];
        // memset(temp_Svv_fill, 0, Vc_nnodes * sizeof(int));
        // Svv_cols = new int[Svv_nnzb];
        // memset(Svv_cols, 0, Svv_nnzb * sizeof(int));
        // for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
        //     std::unordered_set<int> sd_Vc_nodeset;

        //     for (int ielem = 0; ielem < num_elements; ielem++) {
        //         int *local_nodes = &elem_conn[nodes_per_elem * ielem];
        //         int j_subdomain = elem_sd_ind[ielem];
        //         if (i_subdomain != j_subdomain) continue;

        //         for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
        //             int gnode = local_nodes[lnode];
        //             int node_class = node_class_ind[gnode];
        //             if (node_class == VERTEX) {
        //                 int vnode = Vc_node_imap[gnode];
        //                 sd_Vc_nodeset.insert(vnode);
        //             }
        //         }
        //     }

        //     // convert to vector
        //     std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());
        //     // see prevoius block on why each subdomain "macro-element" is fully dense in
        //     macro-elem
        //     // but no connections outside macro-elems
        //     for (int i : sd_Vc_nodes) {
        //         for (int j : sd_Vc_nodes) {
        //             Svv_cols[temp_Svv_fill[i]++] = j;
        //         }
        //     }
        // }

        // printf("Svv_rowp with nnzb %d: ", Svv_nnzb);
        // printVec<int>(Vc_nnodes + 1, Svv_rowp);
        // printf("Svv_cols: ");
        // printVec<int>(Svv_nnzb, Svv_cols);

        Svv_rows = new int[Svv_nnzb];
        for (int i = 0; i < Vc_nnodes; i++) {
            for (int jp = Svv_rowp[i]; jp < Svv_rowp[i + 1]; jp++) {
                Svv_rows[jp] = i;
            }
        }

        // -----------------------------------------
        // build Svv matrix sparsity and do fillin
        // -----------------------------------------
        // small matrix typically

        // Svv_bsr_data = BsrData(Svv_nodes, block_dim, Vc_nnzb, Svv_rowp, Svv_cols);
        Svv_bsr_data = BsrData(Vc_nnodes, block_dim, Svv_nnzb, Svv_rowp, Svv_cols);
        Svv_bsr_data.rows = Svv_rows;
        Svv_nofill_nnzb = Svv_nnzb;
    }

    void setup_coarse_matrix_sparsity() {
        Svv_rowp = Svv_bsr_data.rowp, Svv_cols = Svv_bsr_data.cols, Svv_nnzb = Svv_bsr_data.nnzb;
        SVV_perm = Svv_bsr_data.perm, SVV_iperm = Svv_bsr_data.iperm;

        // recompute rows after potential fillin
        delete[] Svv_rows;
        Svv_rows = new int[Svv_nnzb];
        for (int i = 0; i < Vc_nnodes; i++) {
            for (int jp = Svv_rowp[i]; jp < Svv_rowp[i + 1]; jp++) {
                Svv_rows[jp] = i;
            }
        }
        Svv_bsr_data.rows = Svv_rows;

        d_Svv_bsr_data = Svv_bsr_data.createDeviceBsrData();
        d_Svv_vals = DeviceVec<T>(block_dim2 * Svv_nnzb);
        S_VV = new BsrMatType(d_Svv_bsr_data, d_Svv_vals);
        d_SVV_perm = d_Svv_bsr_data.perm, d_SVV_iperm = d_Svv_bsr_data.iperm;

        // -----------------------------------------
        // IEV => V kmat block copy map (for A_{VV} copy in S_{VV})
        // -----------------------------------------

        Svv_copy_nnzb = 0;
        std::vector<int> Svv_IEV_copyBlocks;
        std::vector<int> Svv_Vc_copyBlocks;
        for (int IEV_row = 0; IEV_row < IEV_nnodes; IEV_row++) {
            int glob_row = IEV_nodes[IEV_row];
            int row_class = node_class_ind[glob_row];
            if (row_class != VERTEX) continue;
            int Vc_row = Vc_node_imap[glob_row];
            int Vc_rowperm = SVV_iperm[Vc_row];

            for (int jp = IEV_rowp[IEV_row]; jp < IEV_rowp[IEV_row + 1]; jp++) {
                int IEV_col = IEV_cols[jp];
                int glob_col = IEV_nodes[IEV_col];
                int col_class = node_class_ind[glob_col];
                if (col_class == VERTEX) {
                    int Vc_col = Vc_node_imap[glob_col];
                    // find matching Vc block ind (will exist)
                    for (int kp = Svv_rowp[Vc_rowperm]; kp < Svv_rowp[Vc_rowperm + 1]; kp++) {
                        int k_perm = Svv_cols[kp];
                        int k = SVV_perm[k_perm];
                        if (k == Vc_col) {
                            Svv_IEV_copyBlocks.push_back(jp);
                            Svv_Vc_copyBlocks.push_back(kp);
                            Svv_copy_nnzb++;
                        }
                    }
                }
            }
        }
        d_Svv_IEV_copyBlocks =
            HostVec<int>(Svv_copy_nnzb, Svv_IEV_copyBlocks.data()).createDeviceVec().getPtr();
        d_Svv_Vc_copyBlocks =
            HostVec<int>(Svv_copy_nnzb, Svv_Vc_copyBlocks.data()).createDeviceVec().getPtr();

        // CONSTRUCT coarse Schur complement mat-invmat-mat maps
        for (int k = 0; k < 4; k++) {
            IEVset_nnzb[k] = 0;
            IEVtoSVV_nnzb[k] = 0;
            d_IEVset_blocks[k] = nullptr;
            d_IEVout_blocks[k] = nullptr;
            d_IEVtoSVV_blocks[k] = nullptr;
        }

        // printf("Vc_nodes: ");
        // printVec<int>(Vc_nnodes, Vc_nodes);

        std::vector<int> IEVset_blocks_host[4];
        std::vector<int> IEVout_blocks_host[4];
        std::vector<int> IEVtoSVV_blocks_host[4];

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::vector<int> sd_iev_vertex_blocks;  // repeated IEV block ids for THIS subdomain
            std::vector<int> sd_vc_nodes;           // coarse vertex ids for THIS subdomain

            // collect repeated vertex nodes on this subdomain in local IEV order
            for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                int gnode = IEV_nodes[jp];
                if (node_class_ind[gnode] == VERTEX) {
                    sd_iev_vertex_blocks.push_back(jp);

                    int vc_node = -1;
                    for (int j = 0; j < Vc_nnodes; j++) {
                        if (Vc_nodes[j] == gnode) {
                            vc_node = j;
                            break;
                        }
                    }

                    if (vc_node < 0) {
                        printf("ERROR: vertex gnode %d on subdomain %d not found in Vc_nodes\n",
                               gnode, isd);
                        exit(-1);
                    }

                    sd_vc_nodes.push_back(vc_node);
                }
            }

            // printf("i_sd %d, sd_iev_vertex_blocks: ", isd);
            // printVec<int>(sd_iev_vertex_blocks.size(), sd_iev_vertex_blocks.data());
            // printf("i_sd %d, sd_vc_nodes: ", isd);
            // printVec<int>(sd_vc_nodes.size(), sd_vc_nodes.data());

            const int nsv = static_cast<int>(sd_iev_vertex_blocks.size());
            if (nsv == 0) continue;

            // optional sanity check for structured quad subdomains
            if (nsv > 4) {
                printf("ERROR: subdomain %d has %d local vertex slots (>4)\n", isd, nsv);
                exit(-1);
            }

            for (int k = 0; k < nsv; k++) {
                const int iev_block =
                    sd_iev_vertex_blocks[k];        // repeated IEV block on THIS subdomain
                const int vc_row = sd_vc_nodes[k];  // global coarse row

                const int vc_row_perm = SVV_iperm[vc_row];

                // set-list: one basis injection per subdomain-local slot
                IEVset_blocks_host[k].push_back(iev_block);
                // printf("IEVset sd %d, k %d, iev_block %d, gnode %d\n", isd, k, iev_block,
                //        IEV_nodes[iev_block]);

                // output/matrix map:
                // only couple to other local vertex slots on THIS SAME subdomain
                for (int kk = 0; kk < nsv; kk++) {
                    const int iev_block2 =
                        sd_iev_vertex_blocks[kk];        // repeated IEV block on THIS subdomain
                    const int vc_col = sd_vc_nodes[kk];  // still THIS SAME subdomain

                    int svv_block = -1;
                    for (int jp = Svv_rowp[vc_row_perm]; jp < Svv_rowp[vc_row_perm + 1]; jp++) {
                        int m_perm = Svv_cols[jp];
                        int m = SVV_perm[m_perm];
                        if (m == vc_col) {
                            svv_block = jp;
                            break;
                        }
                    }

                    if (svv_block < 0) {
                        printf(
                            "ERROR: could not find global Svv block for subdomain %d, row %d, col "
                            "%d\n",
                            isd, vc_row, vc_col);
                        exit(-1);
                    }

                    // duplicate iev_block once for each local coarse destination in THIS subdomain
                    // row
                    IEVout_blocks_host[k].push_back(iev_block2);
                    IEVtoSVV_blocks_host[k].push_back(svv_block);

                    // printf("IEVout sd %d, k %d, iev_block %d, gnode %d\n", isd, k, iev_block2,
                    //        IEV_nodes[iev_block2]);

                    // debug print if needed
                    // printf("isd=%d slot k=%d kk=%d iev_block=%d -> svv_block=%d (row=%d
                    // col=%d)\n",
                    //        isd, k, kk, iev_block, svv_block, vc_row, vc_col);
                }
            }
        }

        for (int k = 0; k < 4; k++) {
            IEVset_nnzb[k] = static_cast<int>(IEVset_blocks_host[k].size());
            IEVtoSVV_nnzb[k] = static_cast<int>(IEVtoSVV_blocks_host[k].size());

            if ((int)IEVout_blocks_host[k].size() != IEVtoSVV_nnzb[k]) {
                printf("ERROR: slot %d mismatch: IEVout size %d != IEVtoSVV size %d\n", k,
                       (int)IEVout_blocks_host[k].size(), IEVtoSVV_nnzb[k]);
                exit(-1);
            }

            if (IEVset_nnzb[k] > 0) {
                d_IEVset_blocks[k] = HostVec<int>(IEVset_nnzb[k], IEVset_blocks_host[k].data())
                                         .createDeviceVec()
                                         .getPtr();
            }

            if (IEVtoSVV_nnzb[k] > 0) {
                d_IEVout_blocks[k] = HostVec<int>(IEVtoSVV_nnzb[k], IEVout_blocks_host[k].data())
                                         .createDeviceVec()
                                         .getPtr();

                d_IEVtoSVV_blocks[k] =
                    HostVec<int>(IEVtoSVV_nnzb[k], IEVtoSVV_blocks_host[k].data())
                        .createDeviceVec()
                        .getPtr();
            }

            // printf("IEVset_nnzb[%d] = %d\n", k, IEVset_nnzb[k]);
            // if (IEVset_nnzb[k] > 0) {
            //     printVec<int>(IEVset_nnzb[k], IEVset_blocks_host[k].data());
            // }

            // printf("IEVtoSVV_nnzb[%d] = %d\n", k, IEVtoSVV_nnzb[k]);
            // if (IEVtoSVV_nnzb[k] > 0) {
            //     printf("IEVout_blocks_host[%d]:\n", k);
            //     printVec<int>(IEVtoSVV_nnzb[k], IEVout_blocks_host[k].data());

            //     printf("IEVtoSVV_blocks_host[%d]:\n", k);
            //     printVec<int>(IEVtoSVV_nnzb[k], IEVtoSVV_blocks_host[k].data());
            // }
        }

        // compute the BC indices needed for kmat_IEV
        auto d_bcs = assembler.getBCs();
        int n_orig_bcs = d_bcs.getSize();
        int *h_bcs = d_bcs.createHostVec().getPtr();
        // printf("h_bcs: ");
        // printVec<int>(n_orig_bcs, h_bcs);

        std::vector<int> IEV_bcs_vec;
        // get new num bcs
        for (int IEV_node = 0; IEV_node < IEV_nnodes; IEV_node++) {
            int gnode = IEV_nodes[IEV_node];
            for (int ibc = 0; ibc < n_orig_bcs; ibc++) {
                int bc = h_bcs[ibc];
                int bc_node = bc / block_dim;
                int bc_dof = bc % block_dim;
                if (bc_node == gnode) {
                    int IEV_dof = block_dim * IEV_node + bc_dof;
                    IEV_bcs_vec.push_back(IEV_dof);
                }
            }
        }

        // printf("IEV_bcs_vec:\n");
        // printVec<int>(IEV_bcs_vec.size(), IEV_bcs_vec.data());
        // for (int ibc = 0; ibc < IEV_bcs_vec.size(); ibc++) {
        //     int bc = IEV_bcs_vec[ibc];
        //     if (ibc % 6 == 0) {
        //         int bc_node = bc / 6;
        //         int glob_node = IEV_nodes[bc_node];
        //         printf("IEV bc node %d, glob node %d\n", bc_node, glob_node);
        //     }
        // }
        d_IEV_bcs = HostVec<int>(IEV_bcs_vec.size(), IEV_bcs_vec.data()).createDeviceVec();
    }

    void assemble_subdomains() {
        // TODO:
        // build workspace vectors / allocate reduced matrices if needed

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(d_vars, d_IEV_vars, block_dim, 1.0, 0.0);

        // // DEBUG
        // T *h_xpts = d_xpts.createHostVec().getPtr();
        // printf("orig_xpts: ");
        // printVec<T>(3 * num_nodes, h_xpts);
        // T *h_IEV_xpts = d_IEV_xpts.createHostVec().getPtr();
        // // printf("IEV_xpts: ");
        // // printVec<T>(3 * IEV_nnodes, h_IEV_xpts);
        // printf("\nIEV_xpts\n");
        // for (int inode = 0; inode < IEV_nnodes; inode++) {
        //     printf("IEV %d, gnode %d, ", inode, IEV_nodes[inode]);
        //     printVec<T>(3, &h_IEV_xpts[3 * inode]);
        // }

        // TODO: kmat_IEV must be allocated before this
        kmat_IEV->zeroValues();

        // int cols_per_elem = (Quadrature::num_quad_pts <= 4) ? 24 : 9;
        const int elems_per_block = 1;
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);
        dim3 block(num_quad_pts, cols_per_elem, 1);
        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nblocks = (num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        k_add_jacobian_fast<T, elems_per_block, ShellAssembler, Data, Vec_, BsrMatType>
            <<<grid, block>>>(IEV_nnodes, num_elements, cols_per_elem, d_elem_components,
                              d_IEV_elem_conn, d_IEV_elem_conn, d_IEV_xpts, d_IEV_vars, d_compData,
                              *kmat_IEV);

        // apply bcs to the kmat_IEV (before copying to IE and I subdomain matrices)
        kmat_IEV->apply_bcs(d_IEV_bcs);

        // copy entries from kmat_IEV to kmat_IE, kmat_I and S_VV matrices
        kmat_IE->zeroValues();
        kmat_I->zeroValues();
        copyKmat_IEVtoIE();
        copyKmat_IEVtoI();
    }

    void assemble_coarse_problem() {
        // now also compute the Schur complement inverse term in S_VV
        copyKmat_IEVtoSvv();
        computeSvvInverseTerm();
    }

    template <class LoadMagnitude, int elems_per_block = 8>
    void add_subdomain_fext(const LoadMagnitude &load) {
        fext_IEV.zeroValues();

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);

        dim3 block(num_quad_pts, elems_per_block);
        dim3 grid(num_elements);

        k_add_fext_fast<T, elems_per_block, ShellAssembler, Data, LoadMagnitude, Vec_>
            <<<grid, block>>>(num_elements, load, d_elem_components, d_IEV_elem_conn,
                              d_IEV_elem_conn, d_IEV_xpts, d_compData, fext_IEV);

        // CHECK_CUDA(cudaDeviceSynchronize());

        fext_IEV.apply_bcs(d_IEV_bcs);

        // T *h_fext_IEV = fext_IEV.createHostVec().getPtr();
        // printf("h_fext_IEV: \n");
        // for (int iIEV = 0; iIEV < IEV_nnodes; iIEV++) {
        //     int iglob = IEV_nodes[iIEV];
        //     printf("iIEV %d, glob node %d: ", iIEV, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int IEV_dof = block_dim * iIEV + idof;
        //         printf("%.6e,", h_fext_IEV[IEV_dof]);
        //     }
        //     printf("\n");
        // }
    }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           BaseSolver *coarseSolver_) {
        subdomainIESolver = subdomainIESolver_;
        subdomainISolver = subdomainISolver_;
        coarseSolver = coarseSolver_;
    }

    void get_lam_rhs(DeviceVec<T> &lam_rhs) {
        // rhs = + fine term - coarse correction
        lam_rhs.zeroValues();

        // ---------------------------------
        // fine term:
        //   f_IE = restricted external load on IE
        //   u_IE = A_IE^{-1} f_IE
        //   lam_rhs += B * u_E
        //   also build repeated IEV copy of u_IE for coarse rhs
        // ---------------------------------
        addVecIEVtoIE(fext_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        // build repeated IE/V representation from full IE solution
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        // only edge/interface part contributes to lambda rhs
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_rhs, 1.0, 0.0);

        // ---------------------------------
        // coarse rhs:
        //   g_V = f_V - A_{V,IE} u_IE
        // using repeated IEV representation
        // ---------------------------------
        addVecIEVtoVc(fext_IEV, f_V, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        // T *h_fc = f_V.createHostVec().getPtr();
        // printf("h_fc:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fc[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // ---------------------------------
        // coarse correction:
        //   u_V = S_VV^{-1} g_V
        //   rhs_IE = A_{IE,V} u_V
        //   corr_IE = A_IE^{-1} rhs_IE
        //   lam_rhs -= B * corr_E
        // ---------------------------------
        // // temp debug
        // f_V.zeroValues();
        // f_V.add_value(2, 1.0);

        // T *h_fc2 = f_V.createHostVec().getPtr();
        // printf("h_fc (ei,i=2):\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fc2[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // // check matrix-vec multiply
        // sparseMatVec(*S_VV, f_V, 1.0, 0.0, u_V);
        // T *h_rV = u_V.createHostVec().getPtr();
        // u_V.zeroValues();
        // printf("h_rV (S_VV*ei):\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_rV[vc_dof]);
        //     }
        //     printf("\n");
        // }

        solveCoarse(f_V, u_V);

        // T *h_uc = u_V.createHostVec().getPtr();
        // printf("h_uc:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_uc[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // // check residual
        // sparseMatVec(*S_VV, u_V, -1.0, 1.0, f_V);
        // T *h_fv_res = f_V.createHostVec().getPtr();
        // printf("h_fv_res:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fv_res[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // addVecVctoIEV only writes selected entries, so clear first
        // u_IEV.zeroValues();
        addVecVctoIEV(u_V, u_IEV, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_rhs, -1.0, 1.0);

        // print lambda rhs
        // T *h_lam_rhs = lam_rhs.createHostVec().getPtr();
        // printf("h_lam_rhs:\n");
        // for (int ilam = 0; ilam < lam_nnodes; ilam++) {
        //     int iglob = lam_nodes[ilam];
        //     printf("ilam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int lam_dof = block_dim * ilam + idof;
        //         printf("%.6e,", h_lam_rhs[lam_dof]);
        //     }
        //     printf("\n");
        // }
    }

    void mat_vec(const DeviceVec<T> &lam_in, DeviceVec<T> &lam_out) {
        lam_out.zeroValues();

        addVecLamtoIE(lam_in, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        u_IEV.zeroValues();
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 1.0);
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_out, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 0.0);

        solveCoarse(f_V, u_V);

        u_IEV.zeroValues();
        addVecVctoIEV(u_V, u_IEV, 1.0, 1.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_out, 1.0, 1.0);
    }

    bool solve(DeviceVec<T> lam_rhs, DeviceVec<T> lam, bool check_conv = false) {
        lam.zeroValues();

        addVecLamtoIE(lam_rhs, u_IE, 0.5, 0.0);  // 1/2 cause B_Ddelta scaled operators
        zeroInteriorIE(u_IE);
        sparseMatVec(*kmat_IE, u_IE, -1.0, 0.0, f_IE);
        // sparseMatVec(*kmat_IE, u_IE, 1.0, 0.0, f_IE);

        addVecIEtoI(f_IE, f_I, 1.0, 0.0);
        solveSubdomainI(f_I, u_I);

        addVecItoIE(u_I, u_IE, 1.0, 1.0);

        sparseMatVec(*kmat_IE, u_IE, 1.0, 0.0, f_IE);
        zeroInteriorIE(f_IE);

        addVecIEtoLam(f_IE, lam, 0.5, 0.0);  // 1/2 cause B_Ddelta scaled operators
        return false;                        // fail = false
    }

    void get_global_soln(const DeviceVec<T> &lam, DeviceVec<T> &soln) {
        // 1) compute coarse grid rhs g_V
        addVecIEVtoIE(fext_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        addVecIEVtoVc(fext_IEV, f_V, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        addVecLamtoIE(lam, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        // 2) solve coarse problem
        solveCoarse(f_V, u_V);

        // 3) solve interior problems
        addVecIEVtoIE(fext_IEV, f_IE, 1.0, 0.0);

        addVecVctoIEV(u_V, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, -1.0, 1.0);

        addVecLamtoIE(lam, f_IE, -1.0, 1.0);

        solveSubdomainIE(f_IE, u_IE);

        soln.zeroValues();
        addGlobalSoln(u_IE, u_V, soln);
    }

    void debug_IEV_matrices(bool print_IEV = true, bool print_IE = true, bool print_I = true) {
        // check matrices
        if (print_IEV) {
            auto kmat_IEV_vec = kmat_IEV->getVec().createHostVec();
            T *h_kmat_IEV = kmat_IEV_vec.getPtr();
            int IEV_nvals = kmat_IEV_vec.getSize();
            auto _IEV_bsr_data = kmat_IEV->getBsrData();
            int *h_IEV_rows = DeviceVec<int>(IEV_nnzb, _IEV_bsr_data.rows).createHostVec().getPtr();
            int *h_IEV_cols = DeviceVec<int>(IEV_nnzb, _IEV_bsr_data.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_IEV %d\n", IEV_nvals);
            for (int iblock = 0; iblock < IEV_nnzb; iblock++) {
                T *IEV_block = &h_kmat_IEV[36 * iblock];
                int row = h_IEV_rows[iblock], col = h_IEV_cols[iblock];
                int grow = IEV_nodes[row], gcol = IEV_nodes[col];
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = IEV_block[6 * ix + iy];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }

        if (print_IE) {
            // prelim check, assumes not perm here
            for (int i = 0; i < IE_nnodes; i++) {
                for (int jp = IE_rowp[i]; jp < IE_rowp[i + 1]; jp++) {
                    int j = IE_cols[jp];
                    int gr = IE_nodes[i], gc = IE_nodes[j];
                    printf("IE block %d, glob (%d,%d)\n", jp, gr, gc);
                }
            }

            auto kmat_IE_vec = kmat_IE->getVec().createHostVec();
            T *h_kmat_IE = kmat_IE_vec.getPtr();
            int IE_nvals = kmat_IE_vec.getSize();
            auto _IEV_bsr_data2 = kmat_IE->getBsrData();
            int *h_IE_rows = DeviceVec<int>(IE_nnzb, _IEV_bsr_data2.rows).createHostVec().getPtr();
            int *h_IE_cols = DeviceVec<int>(IE_nnzb, _IEV_bsr_data2.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_IE %d\n", IE_nvals);
            for (int iblock = 0; iblock < IE_nnzb; iblock++) {
                T *IE_block = &h_kmat_IE[36 * iblock];
                int row = h_IE_rows[iblock], col = h_IE_cols[iblock];
                int grow = IE_nodes[row], gcol = IE_nodes[col];
                // printf("row %d, col %d\n", row, col);
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = IE_block[6 * ix + iy];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }

        if (print_I) {
            // prelim check, assumes not perm here
            for (int i = 0; i < I_nnodes; i++) {
                for (int jp = I_rowp[i]; jp < I_rowp[i + 1]; jp++) {
                    int j = I_cols[jp];
                    int gr = I_nodes[i], gc = I_nodes[j];
                    printf("I block %d, glob (%d,%d)\n", jp, gr, gc);
                }
            }

            auto kmat_I_vec = kmat_I->getVec().createHostVec();
            T *h_kmat_I = kmat_I_vec.getPtr();
            int I_nvals = kmat_I_vec.getSize();
            auto _IV_bsr_data2 = kmat_I->getBsrData();
            int *h_I_rows = DeviceVec<int>(I_nnzb, _IV_bsr_data2.rows).createHostVec().getPtr();
            int *h_I_cols = DeviceVec<int>(I_nnzb, _IV_bsr_data2.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_I %d\n", I_nvals);
            for (int iblock = 0; iblock < I_nnzb; iblock++) {
                T *I_block = &h_kmat_I[36 * iblock];
                int row = h_I_rows[iblock], col = h_I_cols[iblock];
                int grow = I_nodes[row], gcol = I_nodes[col];
                // printf("row %d, col %d\n", row, col);
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = I_block[6 * ix + iy];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }
    }

    void debug_SVV_matrix() {
        // check matrices
        auto temp_vec = S_VV->getVec().createHostVec();
        T *h_Svv_vals = temp_vec.getPtr();
        int Svv_nvals = temp_vec.getSize();
        auto _SVV_bsr_data = S_VV->getBsrData();
        // printf("h_Svv_vals: ");
        // printVec<T>(36, h_Svv_vals);
        int *h_SVV_rows = DeviceVec<int>(Svv_nnzb, _SVV_bsr_data.rows).createHostVec().getPtr();
        int *h_SVV_cols = DeviceVec<int>(Svv_nnzb, _SVV_bsr_data.cols).createHostVec().getPtr();
        printf("\n\nSVV_mat %d\n", Svv_nvals);
        for (int iblock = 0; iblock < Svv_nnzb; iblock++) {
            T *SVV_block = &h_Svv_vals[36 * iblock];
            int row = h_SVV_rows[iblock], col = h_SVV_cols[iblock];
            int grow = Vc_nodes[row], gcol = Vc_nodes[col];
            printf("block %d (%d,%d)\n", iblock, grow, gcol);
            for (int i = 0; i < 9; i++) {
                int ix = 2 + i % 3;
                int iy = 2 + i / 3;
                T val = SVV_block[6 * iy + ix];
                printf("%.6e,", val);
                if (i % 3 == 2) printf("\n");
            }
            // for (int i = 0; i < 36; i++) {
            //     int ix = i % 6;
            //     int iy = i / 6;
            //     T val = SVV_block[6 * ix + iy];
            //     printf("%.6e,", val);
            //     if (i % 6 == 5) printf("\n");
            // }
            printf("\n");
        }
    }

   private:
    void clear_host_data() { clear_structured_host_data(); }

    void clear_structured_host_data() {
        delete[] elem_sd_ind;
        delete[] node_sd_cols;
        delete[] node_elem_ct;
        delete[] node_elem_rowp;
        delete[] node_class_ind;

        delete[] IEV_nodes;
        delete[] IE_nodes;
        delete[] I_nodes;
        delete[] IE_rowp;
        delete[] I_rowp;
        delete[] IE_cols;
        delete[] I_cols;
        delete[] IEV_elem_conn;
        delete[] IEV_sd_ptr;
        delete[] IEV_sd_ind;
        delete[] IEVtoIE_map;
        delete[] IEVtoI_map;

        elem_sd_ind = nullptr;
        node_sd_cols = nullptr;
        node_elem_ct = nullptr;
        node_elem_rowp = nullptr;
        node_class_ind = nullptr;

        IEV_nodes = nullptr;
        IE_nodes = nullptr;
        I_nodes = nullptr;
        IE_rowp = nullptr;
        I_rowp = nullptr;
        IE_cols = nullptr;
        I_cols = nullptr;
        IEV_elem_conn = nullptr;
        IEV_sd_ptr = nullptr;
        IEV_sd_ind = nullptr;
        IEVtoIE_map = nullptr;
        IEVtoI_map = nullptr;
    }

    void copyKmat_IEVtoIE() {
        // Copy / restrict kmat_IEV to kmat_IE using:
        //   IEVtoIE_map, IE_rowp, IE_cols
        int n_IE_vals = IE_nofill_nnzb * block_dim2;
        dim3 block(32), grid((n_IE_vals + 31) / 32);
        k_copyMatToMat_restrict<T><<<grid, block>>>(IE_nofill_nnzb, block_dim, d_kmat_IEtoIEV_map,
                                                    d_kmat_IEnofill_map, d_IEV_vals.getPtr(),
                                                    d_IE_vals.getPtr());

        // printf("IEV_nnzb %d, IE_nnzb %d, IE_nofill_nnzb %d\n", IEV_nnzb, IE_nnzb,
        // IE_nofill_nnzb);

        // printf("h_kmat_IEtoIEV_map: ");
        // printVec<int>(IE_nofill_nnzb, IEVtoIE_imap);
        // printf("kmat_IEnofill_map: ");
        // printVec<int>(IE_nofill_nnzb, kmat_IEnofill_map);
    }

    void copyKmat_IEVtoI() {
        // Copy / restrict kmat_IEV to kmat_I using:
        //   IEVtoI_map, I_rowp, I_cols
        int n_I_vals = I_nofill_nnzb * block_dim2;
        dim3 block(32), grid((n_I_vals + 31) / 32);
        k_copyMatToMat_restrict<T><<<grid, block>>>(I_nofill_nnzb, block_dim, d_kmat_ItoIEV_map,
                                                    d_kmat_Inofill_map, d_IEV_vals.getPtr(),
                                                    d_I_vals.getPtr());
    }

    void copyKmat_IEVtoSvv() {
        // copy Avv part from kmat_IEV to Svv
        // note need Svv_copy_nnzb (not Svv_nnzb) cause Vc are duplicate nodes in IEV
        // multiple blocks copied into the same node in S_VV
        int n_Svv_vals = Svv_copy_nnzb * block_dim2;
        dim3 block(32), grid((n_Svv_vals + 31) / 32);
        k_copyMatToMat_restrict<T, true><<<grid, block>>>(Svv_copy_nnzb, block_dim,
                                                          d_Svv_IEV_copyBlocks, d_Svv_Vc_copyBlocks,
                                                          d_IEV_vals.getPtr(), d_Svv_vals.getPtr());
    }

    void computeSvvInverseTerm() {
        // need 24 IE subdomain solves (tops) for quad-macro elements of struct mesh
        // to compute the -A_{V,IE} * A_{IE,IE}^{-1} * A_{IE,V} += > S_{VV} second Schur complement
        // inverse term as part of coarse matrix assembly for vertices

        // seems quite expensive but remember we do 2 IE solves and 1 I subdomain solve per Krylov
        // step so this is similar expense to like 8 Krylov steps (not too bad, but not trivial)
        // TODO : maybe I can find faster way to do sparse mat-mat triangular solves instead later?
        int ncols = 4 * block_dim;
        for (int icol = 0; icol < ncols; icol++) {
            u_IEV.zeroValues();
            setVec_IEVtoV_vals(u_IEV, icol, 1.0);  // set these vals to 1.0 and all else 0

            sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
            addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
            solveSubdomainIE(f_IE, u_IE);
            addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
            sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);

            addMat_IEVtoV_vals(icol, f_IEV);
        }
    }

    void setVec_IEVtoV_vals(DeviceVec<T> &vec_IEV, int irow, T val) {
        int block_row = irow / block_dim;
        int set_nnzb = IEVset_nnzb[block_row];
        int *d_blocks = d_IEVset_blocks[block_row];

        dim3 block(32);
        dim3 grid((set_nnzb + 31) / 32);
        k_setVec_IEVtoV_vals<T>
            <<<grid, block>>>(set_nnzb, block_dim, irow, d_blocks, vec_IEV.getPtr(), val);
    }

    void addMat_IEVtoV_vals(const int icol, DeviceVec<T> hvec) {
        int block_col = icol / block_dim;
        int set_nnzb = IEVtoSVV_nnzb[block_col];
        int *d_svv_blocks = d_IEVtoSVV_blocks[block_col];
        int *d_iev_blocks = d_IEVout_blocks[block_col];

        dim3 block(32);
        dim3 grid((set_nnzb * block_dim + 31) / 32);
        k_addMat_IEVtoV_vals<T><<<grid, block>>>(set_nnzb, block_dim, icol, d_iev_blocks,
                                                 d_svv_blocks, hvec.getPtr(), d_Svv_vals.getPtr());
    }

    template <bool scaled = false>
    void addVec_globalToIEV(Vec &x_global, Vec &y_iev, int vars_per_node_in, T a, T b) {
        // Scatter a global nodal vector into duplicated IEV layout.
        // Likely based on IEV_nodes and vars_per_node_in.

        CHECK_CUBLAS(cublasDscal(cublasHandle, y_iev.getSize(), &b, y_iev.getPtr(), 1));

        // // DEBUG:
        // int *h_IE_nodes = DeviceVec<int>(IE_nnodes, d_IE_nodes).createHostVec().getPtr();
        // printf("h_IE_nodes: ");
        // printVec<int>(IE_nnodes, h_IE_nodes);
        // int *h_Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();
        // printf("h_Vc_nodes: ");
        // printVec<int>(Vc_nnodes, h_Vc_nodes);
        // T *h_x_glob = x_global.createHostVec().getPtr();
        // printf("h_x_glob: ");
        // printVec<T>(x_global.getSize(), h_x_glob);

        u_IE.zeroValues();
        u_V.zeroValues();

        int nvals = IE_nnodes * vars_per_node_in;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_GlobalToIE<T, scaled><<<grid, block>>>(IE_nnodes, vars_per_node_in, d_IE_nodes,
                                                        d_IE_general_edge, x_global.getPtr(),
                                                        u_IE.getPtr(), a);

        int nvals2 = Vc_nnodes * vars_per_node_in;
        dim3 grid2((nvals2 + 31) / 32);
        k_addVec_GlobaltoVc<T><<<grid2, block>>>(Vc_nnodes, vars_per_node_in, d_Vc_nodes,
                                                 x_global.getPtr(), u_V.getPtr(), a);
        // CHECK_CUBLAS(cublasDscal(cublasHandle, u_IE.getSize(), &a, u_IE.getPtr(), 1));

        // T *h_u_IE = u_IE.createHostVec().getPtr();
        // printf("\nh_u_IE\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("IE %d, gnode %d, ", inode, IE_nodes[inode]);
        //     printVec<T>(3, &h_u_IE[3 * inode]);
        // }

        // T *h_u_V = u_V.createHostVec().getPtr();
        // printf("\nh_u_V\n");
        // for (int inode = 0; inode < Vc_nnodes; inode++) {
        //     printf("IE %d, gnode %d, ", inode, Vc_nodes[inode]);
        //     printVec<T>(3, &h_u_V[3 * inode]);
        // }

        // helper routines
        addVecIEtoIEV(u_IE, y_iev, 1.0, 0.0, vars_per_node_in);
        addVecVctoIEV(u_V, y_iev, 1.0, 1.0, vars_per_node_in);
    }

    void addVecIEVtoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict I/E entries from IEV into IE
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerOut<T>
            <<<grid, block>>>(IE_nnodes, block_dim, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecItoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict interior part from IE into I
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = I_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerIn<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecIEVtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict interior part from IE into I
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = I_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerOut<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecIEVtoVc(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // gather/scatter assembled primal V entries from IEV into Vc (coarse vertex,
        // non-repeated) map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = V_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_IEVtoVc<T><<<grid, block>>>(V_nnodes, block_dim, d_IEVtoV_imap, d_VctoV_imap,
                                             x.getPtr(), y.getPtr(), a);
    }

    void addVecIEtoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        // scatter IE entries into IEV slots
        // map(a * x) + b * y => y
        if (vars_per_node == -1) vars_per_node = block_dim;
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = IE_nnodes * vars_per_node;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerIn<T>
            <<<grid, block>>>(IE_nnodes, vars_per_node, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);

        // int *h_IEVtoIE_imap = DeviceVec<int>(IE_nnodes,
        // d_IEVtoIE_imap).createHostVec().getPtr(); printf("h_IEVtoIE_imap: ");
        // printVec<int>(IE_nnodes, h_IEVtoIE_imap);
        // printf("Check global nodes match\n");
        // for (int IE_node = 0; IE_node < IE_nnodes; IE_node++) {
        //     int gnode1 = IE_nodes[IE_node];
        //     int IEV_node = h_IEVtoIE_imap[IE_node];
        //     int gnode2 = IEV_nodes[IEV_node];
        //     printf("ind %d, IEV_node %d, gnode1 %d, gnode2 %d\n", IE_node, IEV_node, gnode1,
        //            gnode2);
        // }
    }

    void addVecVctoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        // inject coarse/global V entries into local IEV primal slots
        // map(a * x) + b * y => y
        if (vars_per_node == -1) vars_per_node = block_dim;
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = V_nnodes * vars_per_node;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_VctoIEV<T><<<grid, block>>>(V_nnodes, vars_per_node, d_IEVtoV_imap, d_VctoV_imap,
                                             x.getPtr(), y.getPtr(), a);

        // int *h_IEVtoV_imap = DeviceVec<int>(V_nnodes,
        // d_IEVtoV_imap).createHostVec().getPtr(); printf("h_IEVtoV_imap: ");
        // printVec<int>(V_nnodes, h_IEVtoV_imap);
        // int *h_VctoV_imap = DeviceVec<int>(V_nnodes, d_VctoV_imap).createHostVec().getPtr();
        // printf("h_VctoV_imap: ");
        // printVec<int>(V_nnodes, h_VctoV_imap);
    }

    void addVecItoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // scatter interior part from I into IE
        // map(a * x) + b * y => y

        // reuse previous routines instead of adding a new map..
        // don't worry this routine is only called once in main solve so not too much extra
        // overhead
        addVecItoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoIE(temp_IEV, y, 1.0, b);
    }

    void addVecIEtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // scatter interior part from I into IE
        // map(a * x) + b * y => y

        // reuse previous routines instead of adding a new map..
        // don't worry this routine is only called once in main solve so not too much extra
        // overhead
        addVecIEtoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoI(temp_IEV, y, 1.0, b);
    }

    void zeroInteriorIE(DeviceVec<T> &x) {
        // keep edge part, zero interior part
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_zeroInterior<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_interior, x.getPtr());
    }

    void addVecLamtoIE(const DeviceVec<T> &lam, DeviceVec<T> &vec_IE, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecLamtoIE<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            lam.getPtr(), vec_IE.getPtr(), a);
    }

    void addVecIEtoLam(const DeviceVec<T> &vec_IE, DeviceVec<T> &lam, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, lam.getSize(), &b, lam.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecIEtoLam<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            vec_IE.getPtr(), lam.getPtr(), a);
    }

    void addGlobalSoln(const DeviceVec<T> &u_IE, const DeviceVec<T> &u_V, DeviceVec<T> &soln) {
        //  - scatter interior directly to global
        //  - average duplicated edge values into global edge dofs
        //  - inject global primal values

        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_IEtoGlobal<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_nodes, d_IE_general_edge,
                                                u_IE.getPtr(), soln.getPtr(), 1.0);

        int nvals2 = Vc_nnodes * block_dim;
        dim3 grid2((nvals2 + 31) / 32);
        k_addVec_VctoGlobal<T>
            <<<grid2, block>>>(Vc_nnodes, block_dim, d_Vc_nodes, u_V.getPtr(), soln.getPtr(), 1.0);
    }

    void sparseMatVec(BsrMatType &A, Vec &x, T alpha, T beta, Vec &y) {
        //     y <- alpha * A * x + beta * y
        auto bsr_data = A.getBsrData();
        int mb = bsr_data.mb;
        int nb = bsr_data.nb;
        int nnzb = bsr_data.nnzb;
        T *d_vals = A.getVec().getPtr();
        int *d_rowp = bsr_data.rowp;
        int *d_cols = bsr_data.cols;
        int *perm = bsr_data.perm, *iperm = bsr_data.iperm;

        // permute x input vec from VIS to solve order
        x.permuteData(block_dim, iperm);
        y.permuteData(block_dim, iperm);

        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
            &alpha, descrK, d_vals, d_rowp, d_cols, block_dim, x.getPtr(), &beta, y.getPtr()));

        // permute y output vec from solve to VIS order
        x.permuteData(block_dim, perm);
        y.permuteData(block_dim, perm);
    }

    void sparseTransposeMatVec(BsrMatType &A, Vec &x, T alpha, T beta, Vec &y) {
        // BSR mat-vec operation
        //     y <- alpha * A^T * x + beta * y

        auto bsr_data = A.getBsrData();
        int mb = bsr_data.mb;
        int nb = bsr_data.nb;
        int nnzb = bsr_data.nnzb;
        T *d_vals = A.getVec().getPtr();
        int *d_rows = bsr_data.rows;
        int *d_rowp = bsr_data.rowp;
        int *d_cols = bsr_data.cols;
        int *perm = bsr_data.perm, *iperm = bsr_data.iperm;

        // permute x input vec from VIS to solve order
        x.permuteData(block_dim, iperm);
        y.permuteData(block_dim, iperm);

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &beta, y.getPtr(), 1));
        const int nprods = nnzb * block_dim * block_dim;
        dim3 block(32), grid((nprods + 31) / 32);
        k_bsrmv_transpose_ax<T><<<grid, block>>>(nnzb, block_dim, d_rows, d_cols, d_vals,
                                                 x.getPtr(), alpha, y.getPtr());

        // permute y output vec from solve to VIS order
        x.permuteData(block_dim, perm);
        y.permuteData(block_dim, perm);
    }

    void solveSubdomainIE(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainIESolver) {
            printf("ERROR: subdomain IE solver is null\n");
            return;
        }

        auto _bsr_data = kmat_IE->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;

        rhs_in.copyValuesTo(rhs_IE_perm);
        rhs_IE_perm.permuteData(block_dim, d_iperm);

        subdomainIESolver->solve(rhs_IE_perm, sol_IE_perm);

        sol_IE_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveSubdomainI(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainISolver) {
            printf("ERROR: subdomain I solver is null\n");
            return;
        }

        auto _bsr_data = kmat_I->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(rhs_I_perm);
        rhs_I_perm.permuteData(block_dim, d_iperm);

        subdomainISolver->solve(rhs_I_perm, sol_I_perm);

        sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveCoarse(Vec &rhs_in, Vec &sol_out) {
        if (!coarseSolver) {
            printf("ERROR: coarse solver is null\n");
            return;
        }

        auto _bsr_data = S_VV->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(rhs_Vc_perm);
        rhs_Vc_perm.permuteData(block_dim, d_iperm);

        coarseSolver->solve(rhs_Vc_perm, sol_Vc_perm);
        // coarseSolver->solve(rhs_in, sol_out);

        sol_Vc_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void _compute_jump_operators() {
        // compute +1/0/-1 coefficients from u_IE to lam (with 0 for I and +1/-1 for E edges)

        // compute IE to lam map
        IE_to_lam_map = new int[IE_nnodes];
        lam_nodes = new int[lam_nnodes];
        // NOTE : turn the -1 on to ensure the map is correct (nz parts basically)
        memset(IE_to_lam_map, -1, IE_nnodes * sizeof(int));
        // memset(IE_to_lam_map, 0, IE_nnodes * sizeof(int));
        int *temp_glob_node_tracker = new int[num_nodes];
        memset(temp_glob_node_tracker, -1, num_nodes * sizeof(int));
        int *temp_lam_ind = new int[num_nodes];
        memset(temp_lam_ind, -1, num_nodes * sizeof(int));
        int lam_ind = 0;
        for (int i = 0; i < IE_nnodes; i++) {
            int glob_node = IE_nodes[i];
            int node_class = node_class_ind[glob_node];

            if (node_class == EDGE) {
                if (temp_glob_node_tracker[glob_node] != -1) {
                    // then this edge node has been reached by previous subdomain
                    IE_to_lam_map[i] = temp_lam_ind[glob_node];
                } else {
                    // has not been reached by previous subomdain, fill it
                    temp_lam_ind[glob_node] = lam_ind++;
                    temp_glob_node_tracker[glob_node] = 0;
                    IE_to_lam_map[i] = temp_lam_ind[glob_node];
                }
                lam_nodes[temp_lam_ind[glob_node]] = glob_node;
            }
            //  else {
            //     IE_to_lam_map[i] = -1;
            // }
        }
        // printf("IE_to_lam_map: ");
        // printVec<int>(IE_nnodes, IE_to_lam_map);
        // printf("lam_nodes %d: ", lam_nnodes);
        // printVec<int>(lam_nnodes, lam_nodes);

        IE_to_lam_vec = new T[IE_nnodes];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            int glob_node = IE_nodes[inode];
            int IEV_node = IEVtoIE_imap[inode];
            int node_class = node_class_ind[glob_node];
            int sd_ind = IEV_sd_ind[IEV_node];
            int isx = sd_ind % nxs, isy = sd_ind / nxs;
            // use red-black coloring to determine even/odd parity and signs of edge fluxes
            int rb_color = isx + isy;
            // printf("IE_node %d, IEV_node %d, glob_node %d, node_class %d, sd_ind %d\n",
            // inode,
            //        IEV_node, glob_node, node_class, sd_ind);
            // printf("sd_ind %d, sd (%d,%d), rb_color %d\n", sd_ind, isx, isy, rb_color);

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IE_to_lam_vec[inode] = 0.0;
            } else if (node_class == EDGE) {
                IE_to_lam_vec[inode] = (rb_color % 2 == 0) ? 1.0 : -1.0;
            }
        }
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_to_lam_vec: ");
        // printVec<T>(IE_nnodes, IE_to_lam_vec);

        d_IE_to_lam_vec = HostVec<T>(IE_nnodes, IE_to_lam_vec).createDeviceVec().getPtr();
        d_IE_to_lam_map = HostVec<int>(IE_nnodes, IE_to_lam_map).createDeviceVec().getPtr();
    }

    static int find_block_index(int irow, int jcol, const int *rowp, const int *cols) {
        for (int jp = rowp[irow]; jp < rowp[irow + 1]; jp++) {
            if (cols[jp] == jcol) {
                return jp;
            }
        }
        return -1;
    }

    void allocate_workspace() {
        // duplicated IEV vectors
        d_IEV_xpts = Vec(IEV_nnodes * 3);
        d_IEV_vars = Vec(IEV_nnodes * block_dim);

        fext_IEV = Vec(IEV_nnodes * block_dim);
        f_IEV = Vec(IEV_nnodes * block_dim);
        u_IEV = Vec(IEV_nnodes * block_dim);
        temp_IEV = Vec(IEV_nnodes * block_dim);

        // IE vectors
        f_IE = Vec(IE_nnodes * block_dim);
        u_IE = Vec(IE_nnodes * block_dim);
        rhs_IE_perm = Vec(IE_nnodes * block_dim);
        sol_IE_perm = Vec(IE_nnodes * block_dim);

        // I vectors
        f_I = Vec(I_nnodes * block_dim);
        u_I = Vec(I_nnodes * block_dim);
        rhs_I_perm = Vec(I_nnodes * block_dim);
        sol_I_perm = Vec(I_nnodes * block_dim);

        // coarse vertex vectors
        f_V = Vec(Vc_nnodes * block_dim);
        u_V = Vec(Vc_nnodes * block_dim);
        rhs_Vc_perm = Vec(Vc_nnodes * block_dim);
        sol_Vc_perm = Vec(Vc_nnodes * block_dim);
    }

   public:
    int nxe, nye, nxs, nys;
    int nx, ny;
    int num_nodes, num_elements, N;
    int nnxs, nnys, num_subdomains;

    BsrMatType *kmat, *kmat_IEV, *kmat_IE, *kmat_I, *B_delta, *B_Ddelta, *S_VV;
    Vec f_IEV, f_IE, f_I, f_V;
    Vec u_IEV, u_IE, u_I, u_V;
    Vec temp_IEV;
    BsrData IEV_bsr_data, IE_bsr_data, I_bsr_data;
    BsrData d_IEV_bsr_data, d_IE_bsr_data, d_I_bsr_data;
    BsrData Svv_bsr_data, d_Svv_bsr_data;

   private:
    ShellAssembler assembler;
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;

    BaseSolver *subdomainISolver, *subdomainIESolver, *coarseSolver;

    int *elem_sd_ind, *elem_conn;
    int node_elem_nnz;
    int *node_sd_cols, *node_elem_ct, *node_elem_rowp, *node_class_ind;
    int nnodes_interior, nnodes_edge, nnodes_vertex, nnodes_dirichlet_edge;

    int *IEV_nodes, *IE_nodes, *I_nodes;
    int *Vc_nodes;
    int *d_IE_nodes, *d_Vc_nodes;
    int *IEV_rowp, *IE_rowp, *I_rowp;
    int *IEV_rows, *IE_rows, *I_rows;
    int *IEV_cols, *IE_cols, *I_cols;
    int IEV_nnzb, IE_nnzb, I_nnzb;
    int IEV_nnodes, IE_nnodes, I_nnodes;
    int *IEV_elem_conn;
    int *IEV_sd_ptr, *IEV_sd_ind;
    int *IEVtoIE_map, *IEVtoI_map;
    int *IEVtoIE_imap, *IEVtoI_imap;
    int *d_IEVtoIE_imap, *d_IEVtoI_imap;
    int Vc_nnodes, V_nnodes;
    int *VctoV_imap, *d_VctoV_imap;
    int *IEVtoV_imap, *d_IEVtoV_imap;
    bool *IE_interior, *d_IE_interior;
    bool *IE_general_edge, *d_IE_general_edge;
    int *Vc_node_imap;

    // optional cleanup of saved host-side nofill patterns
    int *IE_rowp_nofill, *IE_cols_nofill;
    int *I_rowp_nofill, *I_cols_nofill;
    int *Svv_rowp_nofill, *Svv_cols_nofill;

    // perm maps
    int *IE_perm, *IE_iperm;
    int *I_perm, *I_iperm;
    int *SVV_perm, *SVV_iperm;
    int *d_IE_perm, *d_IE_iperm;
    int *d_I_perm, *d_I_iperm;
    int *d_SVV_perm, *d_SVV_iperm;

    int *kmat_ItoIEV_map, *d_kmat_ItoIEV_map;
    int *kmat_IEtoIEV_map, *d_kmat_IEtoIEV_map;
    int IE_nofill_nnzb, I_nofill_nnzb;
    int *kmat_Inofill_map, *d_kmat_Inofill_map;
    int *kmat_IEnofill_map, *d_kmat_IEnofill_map;
    int lam_nnodes;
    T *IE_to_lam_vec, *d_IE_to_lam_vec;
    int *IE_to_lam_map, *d_IE_to_lam_map;
    int *lam_nodes;

    int Svv_nnzb, Svv_nofill_nnzb, Svv_nodes;
    int *Svv_rowp, *Svv_rows, *Svv_cols;
    int *d_Svv_rowp, *d_Svv_rows, *d_Svv_cols;
    Vec d_Svv_vals;
    int Svv_copy_nnzb;
    int *d_Svv_Vc_copyBlocks, *d_Svv_IEV_copyBlocks;

    // // up to 4 local vertex slots per subdomain (structured quad partition case)
    int *IEVset_blocks[4], *d_IEVset_blocks[4];      // for setVec_IEVtoV_vals
    int *IEVout_blocks[4], *d_IEVout_blocks[4];      // for addMat_IEVtoV_vals read side
    int *IEVtoSVV_blocks[4], *d_IEVtoSVV_blocks[4];  // for addMat_IEVtoV_vals write side

    int IEVset_nnzb[4];
    // int IEVout_nnzb[4];
    int IEVtoSVV_nnzb[4];
    // int *IEVtoV_nnzb;       // length 4
    // int **d_IEVtoV_blocks;  // length 4, each entry is device ptr to block list
    // int *IEVtoSVV_nnzb;       // length 4
    // int **d_IEVtoSVV_blocks;  // length 4, each entry is device ptr to block list

    int block_dim, block_dim2;
    Vec d_IEV_vals, d_IE_vals, d_I_vals;
    int kmat_nnzb;
    Vec d_xpts, d_vars;
    Vec d_IEV_xpts, d_IEV_vars;
    DeviceVec<int> d_IEV_elem_conn;
    DeviceVec<int> d_elem_components;
    DeviceVec<Data> d_compData;
    Vec rhs_IE_perm, sol_IE_perm;
    Vec rhs_I_perm, sol_I_perm;
    Vec rhs_Vc_perm, sol_Vc_perm;
    Vec fext_IEV;  // external loads
    DeviceVec<int> d_IEV_bcs;

    cusparseMatDescr_t descrK;

    // TODO:
    // These are assumed to exist in your codebase / future implementation:
    // int *d_IEV_elem_conn;
    // int *elem_components;
    // int *d_compData;
    // static constexpr int elems_per_block = ...;
};
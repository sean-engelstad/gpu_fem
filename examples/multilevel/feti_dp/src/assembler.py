from ._assembler import Subdomain2DAssembler
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FETIDP_Assembler(Subdomain2DAssembler):
    """
    FETI-DP demo with:
      - primal DOF = vertex DOFs
      - dual DOF   = edge/interface DOFs (shared by exactly 2 subdomains)
      - lambda space indexed one-to-one with unique global edge DOFs

    Assumptions
    -----------
    * Every non-primal edge/interface DOF belongs to exactly 2 subdomains.
    * Vertex primal variables are assembled globally.
    * Local dual edge variables remain subdomain-local and are tied by
      Lagrange multipliers.
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E: float = 70e9,
        nu: float = 0.3,
        thick: float = 1.0e-2,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        nxs: int = 1,
        nys: int = 1,
        scale_R_GE: bool = True,
        coarse_mode: str = "assembled",  # "assembled" or "local"
        geometry='plate',
        radius:float=1.0, # ignored if plate case
    ):
        Subdomain2DAssembler.__init__(
            self, ELEMENT, nxe, E, nu, thick, length, width, load_fcn, clamped, nxs, nys, geometry, radius
        )

        self.scale_R_GE = bool(scale_R_GE)
        self.coarse_mode = coarse_mode

        # Save original full interface classification from base class
        self.sd_all_interface_nodes = {
            i_sd: np.array(self.sd_interface_nodes[i_sd], dtype=int, copy=True)
            for i_sd in range(self.num_subdomains)
        }

        self.full_interface_nodes_global = np.array(
            self.interface_nodes_global, dtype=int, copy=True
        )
        self.full_interface_dof_global = np.array(
            self.interface_dof_global, dtype=int, copy=True
        )

        # global node/dof sets
        self.vertex_nodes_global = np.array([], dtype=int)
        self.edge_interface_nodes_global = np.array([], dtype=int)

        self.vertex_dof_global = np.array([], dtype=int)
        self.edge_interface_dof_global = np.array([], dtype=int)

        # local node/dof sets
        self.sd_vertex_nodes = {}
        self.sd_edge_nodes = {}

        self.sd_vertex_dofs = {}
        self.sd_edge_dofs = {}
        self.sd_G_dofs = {}

        # subset restrictions
        self.sd_R_vertex = {}
        self.sd_R_edge = {}
        self.sd_R_G = {}

        self.R_GE = None
        self.R_GV = None
        self.sd_R_Gi = {}
        self.sd_R_Ei = {}
        self.sd_R_Vi = {}

        # true jump/scaled-jump operators:
        # B_i, B_D_i : (n_lambda, nE_i)
        self.sd_B_delta = {}
        self.sd_BD_delta = {}

        # local lambda sign info / maps
        self.sd_edge_global_dofs = {}
        self.lambda_dof_global = np.array([], dtype=int)
        self.global_edge_owner_pair = {}   # gdof -> (ilo, ihi)

        # 3x3 local blocks
        self.sd_A_II = {}
        self.sd_A_IE = {}
        self.sd_A_IV = {}

        self.sd_A_EI = {}
        self.sd_A_EE = {}
        self.sd_A_EV = {}

        self.sd_A_VI = {}
        self.sd_A_VE = {}
        self.sd_A_VV = {}

        # cached local assembled blocks/factors
        self.sd_A_IE2 = {}
        self.sd_A_IEV = {}

        # coarse operator pieces
        self.sd_S_VV = {}
        self.S_VV = None
        self.S_VV_solver = None

    def _nodes_to_dof(self, node_arr: np.ndarray):
        dpn = self.dof_per_node
        return np.array(
            [dpn * inode + idof for inode in node_arr for idof in range(dpn)],
            dtype=int,
        )

    def _extract_bsr_block(
        self, A_csr: sp.csr_matrix, row_dofs: np.ndarray, col_dofs: np.ndarray
    ):
        b = self.dof_per_node
        m = len(row_dofs)
        n = len(col_dofs)

        if m == 0 or n == 0:
            return sp.bsr_matrix((m, n), blocksize=(b, b))

        return A_csr[row_dofs][:, col_dofs].tobsr(blocksize=(b, b))

    def _make_subset_restriction(
        self,
        child_dofs: np.ndarray,
        parent_dofs: np.ndarray,
        scale: float = 1.0,
    ):
        child_dofs = np.asarray(child_dofs, dtype=int)
        parent_dofs = np.asarray(parent_dofs, dtype=int)

        m = len(child_dofs)
        n = len(parent_dofs)

        if m == 0 or n == 0:
            return sp.csr_matrix((m, n))

        parent_pos = {int(dof): j for j, dof in enumerate(parent_dofs)}

        rows = np.arange(m, dtype=int)
        cols = np.empty(m, dtype=int)

        for i, dof in enumerate(child_dofs):
            dof = int(dof)
            if dof not in parent_pos:
                raise ValueError(f"DOF {dof} is not contained in parent_dofs")
            cols[i] = parent_pos[dof]

        vals = np.full(m, float(scale), dtype=np.double)
        return sp.csr_matrix((vals, (rows, cols)), shape=(m, n))

    def _classify_fetidp_nodes(self):
        """
        Split full global interface G into:
          - vertex nodes: interface nodes belonging to 3+ subdomains
          - edge nodes:   interface nodes belonging to exactly 2 subdomains
        """
        vertex_nodes_global = []
        edge_nodes_global = []

        for gnode in self.full_interface_nodes_global:
            nowners = len(self.node_to_subdomains[int(gnode)])
            if nowners >= 3:
                vertex_nodes_global.append(int(gnode))
            else:
                edge_nodes_global.append(int(gnode))

        self.vertex_nodes_global = np.array(sorted(vertex_nodes_global), dtype=int)
        self.edge_interface_nodes_global = np.array(sorted(edge_nodes_global), dtype=int)

        self.vertex_dof_global = self._node_list_to_dofs(self.vertex_nodes_global)
        self.edge_interface_dof_global = self._node_list_to_dofs(
            self.edge_interface_nodes_global
        )

        # keep full interface G in original ordering
        self.interface_nodes_global = np.array(
            self.full_interface_nodes_global, dtype=int, copy=True
        )
        self.interface_dof_global = np.array(
            self.full_interface_dof_global, dtype=int, copy=True
        )

        self.sd_vertex_nodes = {}
        self.sd_edge_nodes = {}

        for i_sd in range(self.num_subdomains):
            edge_nodes = []
            vertex_nodes = []

            for lnode in self.sd_all_interface_nodes[i_sd]:
                gnode = self.sd_node_inv_map[i_sd][int(lnode)]
                nowners = len(self.node_to_subdomains[int(gnode)])

                if nowners >= 3:
                    vertex_nodes.append(int(lnode))
                else:
                    edge_nodes.append(int(lnode))

            self.sd_vertex_nodes[i_sd] = np.array(vertex_nodes, dtype=int)
            self.sd_edge_nodes[i_sd] = np.array(edge_nodes, dtype=int)

    def _build_global_interface_restrictions(self):
        # keep these as plain subset restrictions; scaling belongs on B_D now
        self.R_GE = self._make_subset_restriction(
            self.edge_interface_dof_global,
            self.interface_dof_global,
            scale=1.0,
        )

        self.R_GV = self._make_subset_restriction(
            self.vertex_dof_global,
            self.interface_dof_global,
            scale=1.0,
        )

    def _build_subdomain_interface_restrictions(self):
        self.sd_R_Gi = {}
        self.sd_R_Ei = {}
        self.sd_R_Vi = {}

        self.sd_G_dofs = {}
        self.sd_edge_dofs = {}
        self.sd_vertex_dofs = {}
        self.sd_edge_global_dofs = {}

        for i_sd in range(self.num_subdomains):
            G_nodes_local = self.sd_all_interface_nodes[i_sd]
            E_nodes_local = self.sd_edge_nodes[i_sd]
            V_nodes_local = self.sd_vertex_nodes[i_sd]

            G_nodes_global = np.array(
                [self.sd_node_inv_map[i_sd][int(lnode)] for lnode in G_nodes_local],
                dtype=int,
            )
            E_nodes_global = np.array(
                [self.sd_node_inv_map[i_sd][int(lnode)] for lnode in E_nodes_local],
                dtype=int,
            )
            V_nodes_global = np.array(
                [self.sd_node_inv_map[i_sd][int(lnode)] for lnode in V_nodes_local],
                dtype=int,
            )

            G_dofs_local = self._node_list_to_dofs(G_nodes_local)
            E_dofs_local = self._node_list_to_dofs(E_nodes_local)
            V_dofs_local = self._node_list_to_dofs(V_nodes_local)

            G_dofs_global = self._node_list_to_dofs(G_nodes_global)
            E_dofs_global = self._node_list_to_dofs(E_nodes_global)
            V_dofs_global = self._node_list_to_dofs(V_nodes_global)

            self.sd_G_dofs[i_sd] = G_dofs_local
            self.sd_edge_dofs[i_sd] = E_dofs_local
            self.sd_vertex_dofs[i_sd] = V_dofs_local
            self.sd_edge_global_dofs[i_sd] = E_dofs_global

            # G -> G_i
            self.sd_R_Gi[i_sd] = self._make_subset_restriction(
                G_dofs_global, self.interface_dof_global, scale=1.0
            )

            # E -> E_i
            self.sd_R_Ei[i_sd] = self._make_subset_restriction(
                E_dofs_global, self.edge_interface_dof_global, scale=1.0
            )

            # V -> V_i
            self.sd_R_Vi[i_sd] = self._make_subset_restriction(
                V_dofs_global, self.vertex_dof_global, scale=1.0
            )

    def _build_jump_operators(self):
        """
        Build true signed jump operators B_delta^(i) and scaled versions B_D^(i).

        lambda space is indexed by unique global edge dofs. For each global edge
        dof there are exactly two owning subdomains (ilo, ihi). The row has:
            +1 on ilo's local edge dof
            -1 on ihi's local edge dof

        If scaling is enabled, use +/- 1/2.
        """
        self.lambda_dof_global = np.array(self.edge_interface_dof_global, dtype=int, copy=True)
        lambda_pos = {int(d): k for k, d in enumerate(self.lambda_dof_global)}
        nlam = len(self.lambda_dof_global)

        # figure out the two-owner pair for each global edge dof
        self.global_edge_owner_pair = {}
        for gdof in self.lambda_dof_global:
            gnode = int(gdof // self.dof_per_node)
            owners = sorted(self.node_to_subdomains[gnode])
            if len(owners) != 2:
                raise ValueError(
                    f"Global edge DOF {gdof} belongs to node {gnode} with owners {owners}, "
                    "but this simplified demo expects exactly 2 subdomains per edge node."
                )
            self.global_edge_owner_pair[int(gdof)] = (owners[0], owners[1])

        self.sd_B_delta = {}
        self.sd_BD_delta = {}

        row_lists = {i_sd: [] for i_sd in range(self.num_subdomains)}
        col_lists = {i_sd: [] for i_sd in range(self.num_subdomains)}
        val_lists = {i_sd: [] for i_sd in range(self.num_subdomains)}
        valD_lists = {i_sd: [] for i_sd in range(self.num_subdomains)}

        scale = 0.5 if self.scale_R_GE else 1.0

        for i_sd in range(self.num_subdomains):
            local_edge_global = self.sd_edge_global_dofs[i_sd]
            g2l = {int(gd): j for j, gd in enumerate(local_edge_global)}

            for gd in local_edge_global:
                gd = int(gd)
                k = lambda_pos[gd]
                ilo, ihi = self.global_edge_owner_pair[gd]

                if i_sd == ilo:
                    sgn = +1.0
                elif i_sd == ihi:
                    sgn = -1.0
                else:
                    continue

                j = g2l[gd]

                row_lists[i_sd].append(k)
                col_lists[i_sd].append(j)
                val_lists[i_sd].append(sgn)
                valD_lists[i_sd].append(scale * sgn)

            nEi = len(local_edge_global)
            self.sd_B_delta[i_sd] = sp.csr_matrix(
                (np.array(val_lists[i_sd], dtype=np.double),
                 (np.array(row_lists[i_sd], dtype=int), np.array(col_lists[i_sd], dtype=int))),
                shape=(nlam, nEi),
            )
            self.sd_BD_delta[i_sd] = sp.csr_matrix(
                (np.array(valD_lists[i_sd], dtype=np.double),
                 (np.array(row_lists[i_sd], dtype=int), np.array(col_lists[i_sd], dtype=int))),
                shape=(nlam, nEi),
            )

    def _build_local_factored_blocks(self):
        self.sd_A_IE2 = {}
        self.sd_A_IEV = {}

        for i_sd in range(self.num_subdomains):
            self.sd_A_IE2[i_sd] = sp.bmat(
                [
                    [self.sd_A_II[i_sd], self.sd_A_IE[i_sd]],
                    [self.sd_A_EI[i_sd], self.sd_A_EE[i_sd]],
                ],
                format="csc",
            )

            self.sd_A_IEV[i_sd] = sp.bmat(
                [
                    [self.sd_A_II[i_sd], self.sd_A_IE[i_sd], self.sd_A_IV[i_sd]],
                    [self.sd_A_EI[i_sd], self.sd_A_EE[i_sd], self.sd_A_EV[i_sd]],
                    [self.sd_A_VI[i_sd], self.sd_A_VE[i_sd], self.sd_A_VV[i_sd]],
                ],
                format="csc",
            )

    def _build_coarse_operator(self):
        """
        Build local S_VV^(i) and assembled global S_VV = sum R_Vi^T S_VV^(i) R_Vi.
        """
        self.sd_S_VV = {}
        nVg = len(self.vertex_dof_global)
        Sglob = sp.csc_matrix((nVg, nVg))

        for i_sd in range(self.num_subdomains):
            nI = self.sd_A_II[i_sd].shape[0]
            nE = self.sd_A_EE[i_sd].shape[0]
            nV = self.sd_A_VV[i_sd].shape[0]

            if nV == 0:
                self.sd_S_VV[i_sd] = sp.csc_matrix((0, 0))
                continue

            rhs = np.zeros((nI + nE, nV), dtype=np.double)

            if nI > 0:
                rhs[:nI, :] = self.sd_A_IV[i_sd].toarray()
            if nE > 0:
                rhs[nI:, :] = self.sd_A_EV[i_sd].toarray()

            X = spla.spsolve(self.sd_A_IE2[i_sd], rhs) if (nI + nE) > 0 else np.zeros((0, nV))
            X = np.atleast_2d(X)
            if X.shape[0] != (nI + nE):
                X = X.reshape((nI + nE, nV), order="F")

            X_I = X[:nI, :] if nI > 0 else np.zeros((0, nV))
            X_E = X[nI:, :] if nE > 0 else np.zeros((0, nV))

            Sii = self.sd_A_VV[i_sd].toarray().copy()
            if nI > 0:
                Sii -= self.sd_A_VI[i_sd].toarray() @ X_I
            if nE > 0:
                Sii -= self.sd_A_VE[i_sd].toarray() @ X_E

            Sii = sp.csc_matrix(Sii)
            self.sd_S_VV[i_sd] = Sii

            RVi = self.sd_R_Vi[i_sd]   # V_global -> V_i
            Sglob = Sglob + (RVi.T @ Sii @ RVi)

        self.S_VV = Sglob.tocsc()

        if self.S_VV.shape[0] > 0:
            self.S_VV_solver = spla.factorized(self.S_VV)
        else:
            self.S_VV_solver = None

    def _solve_local_IE(self, i_sd: int, rhs_I: np.ndarray, rhs_E: np.ndarray):
        """
        Solve [A_II A_IE; A_EI A_EE] [uI; uE] = [rhs_I; rhs_E]
        """
        nI = self.sd_A_II[i_sd].shape[0]
        nE = self.sd_A_EE[i_sd].shape[0]
        if nI + nE == 0:
            return np.zeros(0), np.zeros(0)

        rhs = np.zeros(nI + nE, dtype=np.double)
        if nI > 0:
            rhs[:nI] = rhs_I
        if nE > 0:
            rhs[nI:] = rhs_E

        sol = spla.spsolve(self.sd_A_IE2[i_sd], rhs)
        uI = sol[:nI]
        uE = sol[nI:]
        return uI, uE

    def _apply_local_Schur_E(self, i_sd: int, wE: np.ndarray):
        """
        Apply local Dirichlet Schur complement on edge space with primal part fixed to zero:
            S_EE wE = A_EE wE - A_EI A_II^{-1} A_IE wE
        """
        nI = self.sd_A_II[i_sd].shape[0]
        nE = self.sd_A_EE[i_sd].shape[0]

        if nE == 0:
            return np.zeros(0, dtype=np.double)

        if nI > 0:
            rhsI = -self.sd_A_IE[i_sd].dot(wE)
            yI = spla.spsolve(self.sd_A_II[i_sd].tocsc(), rhsI)
            return self.sd_A_EI[i_sd].dot(yI) + self.sd_A_EE[i_sd].dot(wE)

        return self.sd_A_EE[i_sd].dot(wE)

    def _assemble_g_global(self):
        """
        Build global coarse rhs:
            g_V = sum_i R_Vi^T (fV_i - [A_VI A_VE] K_IE^{-1} [fI_i; fE_i])
        """
        nVg = len(self.vertex_dof_global)
        gV = np.zeros(nVg, dtype=np.double)

        for i_sd in range(self.num_subdomains):
            nV = self.sd_A_VV[i_sd].shape[0]
            if nV == 0:
                continue

            sd_rhs = self.sd_force[i_sd]
            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I]
            fE = sd_rhs[E]
            fV = sd_rhs[V]

            uI_f, uE_f = self._solve_local_IE(i_sd, fI, fE)
            gi = fV.copy()
            gi -= self.sd_A_VI[i_sd].dot(uI_f)
            gi -= self.sd_A_VE[i_sd].dot(uE_f)

            gV += self.sd_R_Vi[i_sd].T.dot(gi)

        return gV

    def _solve_coarse(self, rhs_global_V: np.ndarray):
        """
        Solve coarse system in one of two modes:
          - assembled : exact global assembled S_VV^{-1}
          - local     : approximate sum_i R_i^T S_i^{-1} R_i rhs
        """
        rhs_global_V = np.asarray(rhs_global_V, dtype=np.double)
        nVg = len(self.vertex_dof_global)

        if nVg == 0:
            return np.zeros(0, dtype=np.double)

        if self.coarse_mode == "assembled":
            return np.asarray(self.S_VV_solver(rhs_global_V), dtype=np.double)

        if self.coarse_mode == "local":
            out = np.zeros(nVg, dtype=np.double)
            for i_sd in range(self.num_subdomains):
                nV = self.sd_A_VV[i_sd].shape[0]
                if nV == 0:
                    continue
                rhs_i = self.sd_R_Vi[i_sd].dot(rhs_global_V)
                ui = spla.spsolve(self.sd_S_VV[i_sd], rhs_i)
                out += self.sd_R_Vi[i_sd].T.dot(ui)
            return out

        raise ValueError(f"Unknown coarse_mode={self.coarse_mode!r}")

    def build_subdomain_interface_blocks(self):
        """
        Build local I/E/V blocks, true jump operators, and coarse operator.
        """
        if len(self.sd_kmat) != self.num_subdomains:
            self._assemble_subdomain_systems(bcs=True)

        self._classify_fetidp_nodes()
        self._build_global_interface_restrictions()
        self._build_subdomain_interface_restrictions()

        self.sd_interior_dofs = {}
        self.sd_R_interior = {}

        self.sd_R_interface = {}
        self.sd_R_edge = {}
        self.sd_R_vertex = {}
        self.sd_R_G = {}

        self.sd_A_II = {}
        self.sd_A_IE = {}
        self.sd_A_IV = {}

        self.sd_A_EI = {}
        self.sd_A_EE = {}
        self.sd_A_EV = {}

        self.sd_A_VI = {}
        self.sd_A_VE = {}
        self.sd_A_VV = {}

        for i_sd in range(self.num_subdomains):
            ndof = self.sd_ndof[i_sd]
            A_csr = self.sd_kmat[i_sd].tocsr()

            I_nodes = self.sd_interior_nodes[i_sd]
            E_nodes = self.sd_edge_nodes[i_sd]
            V_nodes = self.sd_vertex_nodes[i_sd]
            G_nodes = self.sd_all_interface_nodes[i_sd]

            I_dofs = self._node_list_to_dofs(I_nodes)
            E_dofs = self._node_list_to_dofs(E_nodes)
            V_dofs = self._node_list_to_dofs(V_nodes)
            G_dofs = self._node_list_to_dofs(G_nodes)

            self.sd_interior_dofs[i_sd] = I_dofs
            self.sd_interface_dofs[i_sd] = G_dofs
            self.sd_G_dofs[i_sd] = G_dofs
            self.sd_edge_dofs[i_sd] = E_dofs
            self.sd_vertex_dofs[i_sd] = V_dofs

            self.sd_R_interior[i_sd] = self._make_restriction(I_dofs, ndof)
            self.sd_R_G[i_sd] = self._make_restriction(G_dofs, ndof)
            self.sd_R_interface[i_sd] = self.sd_R_G[i_sd]
            self.sd_R_edge[i_sd] = self._make_restriction(E_dofs, ndof)
            self.sd_R_vertex[i_sd] = self._make_restriction(V_dofs, ndof)

            self.sd_A_II[i_sd] = self._extract_bsr_block(A_csr, I_dofs, I_dofs)
            self.sd_A_IE[i_sd] = self._extract_bsr_block(A_csr, I_dofs, E_dofs)
            self.sd_A_IV[i_sd] = self._extract_bsr_block(A_csr, I_dofs, V_dofs)

            self.sd_A_EI[i_sd] = self._extract_bsr_block(A_csr, E_dofs, I_dofs)
            self.sd_A_EE[i_sd] = self._extract_bsr_block(A_csr, E_dofs, E_dofs)
            self.sd_A_EV[i_sd] = self._extract_bsr_block(A_csr, E_dofs, V_dofs)

            self.sd_A_VI[i_sd] = self._extract_bsr_block(A_csr, V_dofs, I_dofs)
            self.sd_A_VE[i_sd] = self._extract_bsr_block(A_csr, V_dofs, E_dofs)
            self.sd_A_VV[i_sd] = self._extract_bsr_block(A_csr, V_dofs, V_dofs)

        self._build_local_factored_blocks()
        self._build_jump_operators()
        self._build_coarse_operator()

    def get_interface_rhs(self):
        """
        FETI-DP rhs in lambda-space:
            rhs = -( d_Lambda - B_{Lambda,Pi} S_VV^{-1} g_V )
                = (+ fine load term) - (coarse correction term)
        """
        nlam = len(self.lambda_dof_global)
        rhs = np.zeros(nlam, dtype=np.double)

        # fine load term: -d_Lambda
        gV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I]
            fE = sd_rhs[E]
            fV = sd_rhs[V]

            uI_f, uE_f = self._solve_local_IE(i_sd, fI, fE)

            # + fine term
            rhs += self.sd_B_delta[i_sd].dot(uE_f)

            # local contribution to g_V
            if len(V) > 0:
                gi = fV.copy()
                gi -= self.sd_A_VI[i_sd].dot(uI_f)
                gi -= self.sd_A_VE[i_sd].dot(uE_f)
                gV += self.sd_R_Vi[i_sd].T.dot(gi)

        # coarse correction term
        if len(self.vertex_dof_global) > 0:
            uV = self._solve_coarse(gV)

            for i_sd in range(self.num_subdomains):
                nV = len(self.sd_vertex_dofs[i_sd])
                if nV == 0:
                    continue

                uVi = self.sd_R_Vi[i_sd].dot(uV)

                rhsI = self.sd_A_IV[i_sd].dot(uVi)
                rhsE = self.sd_A_EV[i_sd].dot(uVi)

                _, corrE = self._solve_local_IE(i_sd, rhsI, rhsE)

                # subtract coarse term
                rhs -= self.sd_B_delta[i_sd].dot(corrE)

        return rhs

    def mat_vec(self, lam: np.ndarray):
        """
        Apply FETI-DP operator in lambda-space:
            F lam = B S_tilde^{-1} B^T lam
        """
        lam = np.asarray(lam, dtype=np.double)
        nlam = len(self.lambda_dof_global)
        if lam.shape[0] != nlam:
            raise ValueError(f"lambda has size {lam.shape[0]}, expected {nlam}")

        out = np.zeros(nlam, dtype=np.double)

        # first local fine solve pieces
        cV = np.zeros(len(self.vertex_dof_global), dtype=np.double)
        local_z = {}

        for i_sd in range(self.num_subdomains):
            qE = self.sd_B_delta[i_sd].T.dot(lam)  # local dual forcing = B_i^T lambda
            uI, uE = self._solve_local_IE(
                i_sd,
                np.zeros_like(self.sd_interior_dofs[i_sd], dtype=np.double),
                qE,
            )
            local_z[i_sd] = (uI, uE)

            out += self.sd_B_delta[i_sd].dot(uE)

            if len(self.sd_vertex_dofs[i_sd]) > 0:
                ci = self.sd_A_VI[i_sd].dot(uI) + self.sd_A_VE[i_sd].dot(uE)
                cV += self.sd_R_Vi[i_sd].T.dot(ci)

        # assembled coarse correction
        if len(self.vertex_dof_global) > 0:
            uV = self._solve_coarse(cV)

            for i_sd in range(self.num_subdomains):
                nV = len(self.sd_vertex_dofs[i_sd])
                if nV == 0:
                    continue

                uVi = self.sd_R_Vi[i_sd].dot(uV)

                rhsI = self.sd_A_IV[i_sd].dot(uVi)
                rhsE = self.sd_A_EV[i_sd].dot(uVi)

                _, corrE = self._solve_local_IE(i_sd, rhsI, rhsE)
                out += self.sd_B_delta[i_sd].dot(corrE)

        return out

    def precond_solve(self, lam_rhs: np.ndarray):
        """
        Dirichlet preconditioner action:
            M^{-1} r = sum_i B_D^{(i)} S_EE^{(i)} B_D^{(i)T} r

        Here S_EE^{(i)} is the local Dirichlet Schur complement with primal
        variables fixed to zero.
        """
        lam_rhs = np.asarray(lam_rhs, dtype=np.double)
        nlam = len(self.lambda_dof_global)
        if lam_rhs.shape[0] != nlam:
            raise ValueError(f"preconditioner rhs has size {lam_rhs.shape[0]}, expected {nlam}")

        out = np.zeros(nlam, dtype=np.double)

        for i_sd in range(self.num_subdomains):
            wE = self.sd_BD_delta[i_sd].T.dot(lam_rhs)
            zE = self._apply_local_Schur_E(i_sd, wE)
            out += self.sd_BD_delta[i_sd].dot(zE)

        return out

    def get_global_solution(self, lam: np.ndarray):
        """
        Recover physical global displacement from lambda.

        Recovery:
          1) build global coarse rhs
               gV(lambda) = gV(load) + sum_i R_i^T [A_VI A_VE] K_IE^{-1} [0; B_i^T lam]
          2) solve uV
          3) for each subdomain solve
               K_IE [uI; uE] = [fI; fE] - [A_IV; A_EV] uVi - [0; B_i^T lam]
          4) assemble global I,V directly; average E over subdomains
        """
        lam = np.asarray(lam, dtype=np.double)
        nlam = len(self.lambda_dof_global)
        if lam.shape[0] != nlam:
            raise ValueError(f"lambda has size {lam.shape[0]}, expected {nlam}")

        global_soln = np.zeros(self.N, dtype=np.double)

        # First compute load-only coarse rhs gV
        gV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I]
            fE = sd_rhs[E]
            fV = sd_rhs[V]

            uI_f, uE_f = self._solve_local_IE(i_sd, fI, fE)

            if len(V) > 0:
                gi = fV.copy()
                gi -= self.sd_A_VI[i_sd].dot(uI_f)
                gi -= self.sd_A_VE[i_sd].dot(uE_f)
                gV += self.sd_R_Vi[i_sd].T.dot(gi)

        # Add lambda-dependent coarse rhs piece: -B_{Lambda,Pi}^T lam
        for i_sd in range(self.num_subdomains):
            qE = self.sd_B_delta[i_sd].T.dot(lam)
            uI_l, uE_l = self._solve_local_IE(
                i_sd,
                np.zeros(len(self.sd_interior_dofs[i_sd]), dtype=np.double),
                qE,
            )

            if len(self.sd_vertex_dofs[i_sd]) > 0:
                ci = self.sd_A_VI[i_sd].dot(uI_l) + self.sd_A_VE[i_sd].dot(uE_l)
                gV += self.sd_R_Vi[i_sd].T.dot(ci)

        # Solve coarse problem
        uV_global = self._solve_coarse(gV) if len(self.vertex_dof_global) > 0 else np.zeros(0)

        # Average edge values into global physical solution
        edge_accum = np.zeros(len(self.edge_interface_dof_global), dtype=np.double)
        edge_count = np.zeros(len(self.edge_interface_dof_global), dtype=np.double)

        edge_pos = {int(d): j for j, d in enumerate(self.edge_interface_dof_global)}

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I].copy()
            fE = sd_rhs[E].copy()

            qE = self.sd_B_delta[i_sd].T.dot(lam)
            uVi = self.sd_R_Vi[i_sd].dot(uV_global) if len(V) > 0 else np.zeros(0)

            rhsI = fI.copy()
            rhsE = fE.copy() - qE

            if len(V) > 0:
                rhsI -= self.sd_A_IV[i_sd].dot(uVi)
                rhsE -= self.sd_A_EV[i_sd].dot(uVi)

            uI, uE = self._solve_local_IE(i_sd, rhsI, rhsE)

            # interior directly
            sd_global_dofs = self.get_subdomain_global_dofs(i_sd)
            global_I_dofs = sd_global_dofs[I]
            global_soln[global_I_dofs] = uI

            # vertices directly (assembled/global primal values)
            if len(V) > 0:
                global_V_dofs = sd_global_dofs[V]
                global_soln[global_V_dofs] = uVi

            # average recovered edge values
            E_global = self.sd_edge_global_dofs[i_sd]
            for a, gdof in enumerate(E_global):
                j = edge_pos[int(gdof)]
                edge_accum[j] += uE[a]
                edge_count[j] += 1.0

        mask = edge_count > 0
        global_soln[self.edge_interface_dof_global[mask]] = edge_accum[mask] / edge_count[mask]

        return global_soln
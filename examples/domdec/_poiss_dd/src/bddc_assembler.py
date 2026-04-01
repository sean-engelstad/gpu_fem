from .assembler import SubdomainPlateAssembler
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class BDDC_PlateAssembler(SubdomainPlateAssembler):
    """
    Reduced-edge BDDC demo:
      - primal DOF = vertex DOFs
      - interface Krylov unknown = unique global edge DOFs

    This is a clean demo of the BDDC-style preconditioner/action using the same
    I/E/V split as the FETI-DP demo.

    Assumptions
    -----------
    * Every non-primal edge/interface DOF belongs to exactly 2 subdomains.
    * Vertex primal variables are assembled globally.
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        nxs: int = 1,
        nys: int = 1,
        scale_R_GE: bool = True,
        coarse_mode: str = "assembled",  # "assembled" or "local"
    ):
        SubdomainPlateAssembler.__init__(
            self, ELEMENT, nxe, length, width, load_fcn, clamped, nxs, nys
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

        # scaled averaging operator for BDDC:
        # u_Ei = R_DGamma^(i) u_E
        self.sd_R_DGamma = {}

        # local edge-global maps
        self.sd_edge_global_dofs = {}

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

        # cached local assembled blocks
        self.sd_A_IE2 = {}
        self.sd_A_IEV = {}

        # coarse operator pieces
        self.sd_S_VV = {}
        self.S_VV = None
        self.S_VV_solver = None

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

    def _classify_bddc_nodes(self):
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

            self.sd_R_Gi[i_sd] = self._make_subset_restriction(
                G_dofs_global, self.interface_dof_global, scale=1.0
            )
            self.sd_R_Ei[i_sd] = self._make_subset_restriction(
                E_dofs_global, self.edge_interface_dof_global, scale=1.0
            )
            self.sd_R_Vi[i_sd] = self._make_subset_restriction(
                V_dofs_global, self.vertex_dof_global, scale=1.0
            )

    def _build_scaled_averaging_operators(self):
        """
        Build scaled averaging / restriction operators for BDDC:
            u_Ei = R_DGamma^(i) u_E

        Since each edge dof is shared by exactly 2 subdomains in this demo,
        the standard scaling is 1/2 on each owner when enabled.
        """
        self.sd_R_DGamma = {}
        scale = 0.5 if self.scale_R_GE else 1.0

        for i_sd in range(self.num_subdomains):
            E_nodes_local = self.sd_edge_nodes[i_sd]
            E_nodes_global = np.array(
                [self.sd_node_inv_map[i_sd][int(lnode)] for lnode in E_nodes_local],
                dtype=int,
            )

            E_dofs_global = self._node_list_to_dofs(E_nodes_global)
            self.sd_R_DGamma[i_sd] = self._make_subset_restriction(
                E_dofs_global,
                self.edge_interface_dof_global,
                scale=scale,
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
        Build local S_VV^(i) and assembled global S_VV = sum R_Vi^T S_VV^(i) R_Vi
        using elimination of (I,E).
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
            X = np.asarray(X)
            if X.ndim == 1:
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

            RVi = self.sd_R_Vi[i_sd]
            Sglob = Sglob + (RVi.T @ Sii @ RVi)

        self.S_VV = Sglob.tocsc()
        self.S_VV_solver = spla.factorized(self.S_VV) if self.S_VV.shape[0] > 0 else None

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
        return sol[:nI], sol[nI:]

    def _solve_local_IEV(self, i_sd: int, rhs_I: np.ndarray, rhs_E: np.ndarray, rhs_V: np.ndarray):
        """
        Solve full local 3x3 system.
        """
        nI = self.sd_A_II[i_sd].shape[0]
        nE = self.sd_A_EE[i_sd].shape[0]
        nV = self.sd_A_VV[i_sd].shape[0]
        if nI + nE + nV == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)

        rhs = np.zeros(nI + nE + nV, dtype=np.double)
        if nI > 0:
            rhs[:nI] = rhs_I
        if nE > 0:
            rhs[nI:nI + nE] = rhs_E
        if nV > 0:
            rhs[nI + nE:] = rhs_V

        sol = spla.spsolve(self.sd_A_IEV[i_sd], rhs)
        return sol[:nI], sol[nI:nI + nE], sol[nI + nE:]

    def _apply_local_Schur_E(self, i_sd: int, wE: np.ndarray):
        """
        Apply local Schur complement on edge space with primal part fixed to zero:
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
        gV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        for i_sd in range(self.num_subdomains):
            if self.sd_A_VV[i_sd].shape[0] == 0:
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
                if self.sd_A_VV[i_sd].shape[0] == 0:
                    continue
                rhs_i = self.sd_R_Vi[i_sd].dot(rhs_global_V)
                ui = spla.spsolve(self.sd_S_VV[i_sd], rhs_i)
                out += self.sd_R_Vi[i_sd].T.dot(ui)
            return out

        raise ValueError(f"Unknown coarse_mode={self.coarse_mode!r}")

    def build_subdomain_interface_blocks(self):
        """
        Build local I/E/V blocks and BDDC averaging/coarse operators.
        """
        if len(self.sd_kmat) != self.num_subdomains:
            self._assemble_subdomain_systems(bcs=True)

        self._classify_bddc_nodes()
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
        self._build_scaled_averaging_operators()
        self._build_coarse_operator()

    def _solve_local_I(self, i_sd: int, rhs_I: np.ndarray):
        nI = self.sd_A_II[i_sd].shape[0]
        if nI == 0:
            return np.zeros(0, dtype=np.double)
        return np.asarray(
            spla.spsolve(self.sd_A_II[i_sd].tocsc(), rhs_I),
            dtype=np.double,
        )

    def _apply_local_S_EV(self, i_sd: int, wV: np.ndarray):
        """
        Apply
            S_EV wV = A_EV wV - A_EI A_II^{-1} A_IV wV
        """
        nE = self.sd_A_EE[i_sd].shape[0]
        nV = self.sd_A_VV[i_sd].shape[0]
        if nE == 0 or nV == 0:
            return np.zeros(nE, dtype=np.double)

        nI = self.sd_A_II[i_sd].shape[0]
        if nI > 0:
            yI = self._solve_local_I(i_sd, -self.sd_A_IV[i_sd].dot(wV))
            return self.sd_A_EI[i_sd].dot(yI) + self.sd_A_EV[i_sd].dot(wV)

        return self.sd_A_EV[i_sd].dot(wV)

    def _apply_local_S_VE(self, i_sd: int, wE: np.ndarray):
        """
        Apply
            S_VE wE = A_VE wE - A_VI A_II^{-1} A_IE wE
        """
        nV = self.sd_A_VV[i_sd].shape[0]
        nE = self.sd_A_EE[i_sd].shape[0]
        if nV == 0 or nE == 0:
            return np.zeros(nV, dtype=np.double)

        nI = self.sd_A_II[i_sd].shape[0]
        if nI > 0:
            yI = self._solve_local_I(i_sd, -self.sd_A_IE[i_sd].dot(wE))
            return self.sd_A_VI[i_sd].dot(yI) + self.sd_A_VE[i_sd].dot(wE)

        return self.sd_A_VE[i_sd].dot(wE)
    
    def get_interface_rhs(self):
        """
        Reduced edge rhs:
            g_E^red = g_E - S_EV S_VV^{-1} g_V
        """
        nEglob = len(self.edge_interface_dof_global)
        rhs = np.zeros(nEglob, dtype=np.double)
        gV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        # Step 1: form Schur loads after eliminating only I
        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I]
            fE = sd_rhs[E]
            fV = sd_rhs[V]

            yI = self._solve_local_I(i_sd, fI)

            gEi = fE.copy()
            gVi = fV.copy()

            if len(yI) > 0:
                gEi -= self.sd_A_EI[i_sd].dot(yI)
                gVi -= self.sd_A_VI[i_sd].dot(yI)

            rhs += self.sd_R_Ei[i_sd].T.dot(gEi)

            if len(V) > 0:
                gV += self.sd_R_Vi[i_sd].T.dot(gVi)

        # Step 2: eliminate global primal block
        if len(self.vertex_dof_global) > 0:
            uV = self._solve_coarse(gV)

            for i_sd in range(self.num_subdomains):
                if len(self.sd_vertex_dofs[i_sd]) == 0:
                    continue

                uVi = self.sd_R_Vi[i_sd].dot(uV)
                rhs -= self.sd_R_Ei[i_sd].T.dot(self._apply_local_S_EV(i_sd, uVi))

        return rhs

    def mat_vec(self, edge_in: np.ndarray):
        """
        Reduced edge operator:
            S_red = S_EE - S_EV S_VV^{-1} S_VE
        """
        edge_in = np.asarray(edge_in, dtype=np.double)
        nEglob = len(self.edge_interface_dof_global)
        if edge_in.shape[0] != nEglob:
            raise ValueError(f"edge_in has size {edge_in.shape[0]}, expected {nEglob}")

        out = np.zeros(nEglob, dtype=np.double)
        cV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        # S_EE * uE and S_VE * uE
        for i_sd in range(self.num_subdomains):
            uEi = self.sd_R_Ei[i_sd].dot(edge_in)
            out += self.sd_R_Ei[i_sd].T.dot(self._apply_local_Schur_E(i_sd, uEi))

            if len(self.sd_vertex_dofs[i_sd]) > 0:
                cV += self.sd_R_Vi[i_sd].T.dot(self._apply_local_S_VE(i_sd, uEi))

        # subtract S_EV S_VV^{-1} S_VE * uE
        if len(self.vertex_dof_global) > 0:
            uV = self._solve_coarse(cV)

            for i_sd in range(self.num_subdomains):
                if len(self.sd_vertex_dofs[i_sd]) == 0:
                    continue

                uVi = self.sd_R_Vi[i_sd].dot(uV)
                out -= self.sd_R_Ei[i_sd].T.dot(self._apply_local_S_EV(i_sd, uVi))

        return out

    def precond_solve(self, edge_rhs: np.ndarray):
        """
        BDDC preconditioner action:
            M^{-1} r = R_D^T \tilde S^{-1} R_D r

        Implemented with the same local IE solves + global vertex coarse solve
        pattern as the working FETI-DP inverse-side operator, but with scaled
        averaging instead of the jump operator.
        """
        edge_rhs = np.asarray(edge_rhs, dtype=np.double)
        nEglob = len(self.edge_interface_dof_global)
        if edge_rhs.shape[0] != nEglob:
            raise ValueError(f"edge_rhs has size {edge_rhs.shape[0]}, expected {nEglob}")

        out = np.zeros(nEglob, dtype=np.double)
        cV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        for i_sd in range(self.num_subdomains):
            qE = self.sd_R_DGamma[i_sd].dot(edge_rhs)
            uI, uE = self._solve_local_IE(
                i_sd,
                np.zeros(len(self.sd_interior_dofs[i_sd]), dtype=np.double),
                qE,
            )

            out += self.sd_R_DGamma[i_sd].T.dot(uE)

            if len(self.sd_vertex_dofs[i_sd]) > 0:
                ci = self.sd_A_VI[i_sd].dot(uI) + self.sd_A_VE[i_sd].dot(uE)
                cV += self.sd_R_Vi[i_sd].T.dot(ci)

        if len(self.vertex_dof_global) > 0:
            uV = self._solve_coarse(cV)

            for i_sd in range(self.num_subdomains):
                if len(self.sd_vertex_dofs[i_sd]) == 0:
                    continue

                uVi = self.sd_R_Vi[i_sd].dot(uV)
                rhsI = self.sd_A_IV[i_sd].dot(uVi)
                rhsE = self.sd_A_EV[i_sd].dot(uVi)

                _, corrE = self._solve_local_IE(i_sd, rhsI, rhsE)
                out += self.sd_R_DGamma[i_sd].T.dot(corrE)

        return out

    def get_global_solution(self, edge_disp: np.ndarray):
        """
        Recover physical global displacement from solved global edge displacement.
        """
        edge_disp = np.asarray(edge_disp, dtype=np.double)
        nEglob = len(self.edge_interface_dof_global)
        if edge_disp.shape[0] != nEglob:
            raise ValueError(f"edge_disp has size {edge_disp.shape[0]}, expected {nEglob}")

        global_soln = np.zeros(self.N, dtype=np.double)
        global_soln[self.edge_interface_dof_global] = edge_disp.copy()

        # build coarse rhs from loads and edge contribution
        gV = np.zeros(len(self.vertex_dof_global), dtype=np.double)

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I]
            fE = sd_rhs[E]
            fV = sd_rhs[V]

            uEi = self.sd_R_Ei[i_sd].dot(edge_disp)
            uI_f, uE_f = self._solve_local_IE(i_sd, fI, fE - uEi)

            if len(V) > 0:
                gi = fV.copy()
                gi -= self.sd_A_VI[i_sd].dot(uI_f)
                gi -= self.sd_A_VE[i_sd].dot(uE_f)
                gV += self.sd_R_Vi[i_sd].T.dot(gi)

        uV_global = self._solve_coarse(gV) if len(self.vertex_dof_global) > 0 else np.zeros(0)

        if len(self.vertex_dof_global) > 0:
            global_soln[self.vertex_dof_global] = uV_global

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = sd_rhs[I].copy()
            uEi = self.sd_R_Ei[i_sd].dot(edge_disp)
            uVi = self.sd_R_Vi[i_sd].dot(uV_global) if len(V) > 0 else np.zeros(0)

            rhsI = fI.copy()
            if len(E) > 0:
                rhsI -= self.sd_A_IE[i_sd].dot(uEi)
            if len(V) > 0:
                rhsI -= self.sd_A_IV[i_sd].dot(uVi)

            if len(I) > 0:
                uI = spla.spsolve(self.sd_A_II[i_sd].tocsc(), rhsI)
            else:
                uI = np.zeros(0, dtype=np.double)

            sd_global_dofs = self.get_subdomain_global_dofs(i_sd)

            if len(I) > 0:
                global_I_dofs = sd_global_dofs[I]
                global_soln[global_I_dofs] = uI

            if len(V) > 0:
                global_V_dofs = sd_global_dofs[V]
                global_soln[global_V_dofs] = uVi

        return global_soln
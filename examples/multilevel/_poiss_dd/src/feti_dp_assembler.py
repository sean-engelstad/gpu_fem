from .assembler import SubdomainPlateAssembler
import numpy as np
import scipy.sparse as sp


class FETIDP_PlateAssembler(SubdomainPlateAssembler):
    def __init__(
        self,
        ELEMENT,
        nxe: int,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        nxs: int = 1,   # num subdomains in x-direc
        nys: int = 1,   # num subdomains in y-direc
        scale_R_GE: bool = True, # default we do scale R_GE
    ):
        # call main init, including nz pattern for subdomains
        SubdomainPlateAssembler.__init__(
            self, ELEMENT, nxe, length, width, load_fcn, clamped, nxs, nys
        )

        self.scale_R_GE = bool(scale_R_GE)

        # save original base-class interface-node classification
        # base interface = full non-Dirichlet shared interface = G on each subdomain
        self.sd_all_interface_nodes = {
            i_sd: np.array(self.sd_interface_nodes[i_sd], dtype=int, copy=True)
            for i_sd in range(self.num_subdomains)
        }

        # save original full global interface from base class
        self.full_interface_nodes_global = np.array(self.interface_nodes_global, dtype=int, copy=True)
        self.full_interface_dof_global = np.array(self.interface_dof_global, dtype=int, copy=True)

        # global node/dof sets
        # G = full interface, E = edge subset, V = vertex subset
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

        # old local restrictions from local full dof vector -> subset
        self.sd_R_vertex = {}
        self.sd_R_edge = {}
        self.sd_R_G = {}

        # new global/subdomain restriction operators
        # x_E  = R_GE x_G
        # x_V  = R_GV x_G
        # x_Gi = R_Gi x_G
        # x_Ei = R_Ei x_E
        # x_Vi = R_Vi x_V
        self.R_GE = None
        self.R_GV = None
        self.sd_R_Gi = {}
        self.sd_R_Ei = {}
        self.sd_R_Vi = {}

        # 3x3 subdomain blocks
        self.sd_A_II = {}
        self.sd_A_IE = {}
        self.sd_A_IV = {}

        self.sd_A_EI = {}
        self.sd_A_EE = {}
        self.sd_A_EV = {}

        self.sd_A_VI = {}
        self.sd_A_VE = {}
        self.sd_A_VV = {}

    def _nodes_to_dof(self, node_arr: np.ndarray):
        dpn = self.dof_per_node
        return np.array(
            [dpn * inode + idof for inode in node_arr for idof in range(dpn)],
            dtype=int,
        )

    def _extract_bsr_block(
        self, A_csr: sp.csr_matrix, row_dofs: np.ndarray, col_dofs: np.ndarray
    ):
        """
        Extract A[row_dofs, col_dofs] from CSR, return BSR with node blocksize.
        Handles empty slices cleanly.
        """
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
        """
        Build restriction R such that
            x_child = R @ x_parent

        where child_dofs is a subset of parent_dofs, both expressed in the same
        global DOF numbering.

        Shape:
            R : (len(child_dofs), len(parent_dofs))
        """
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

        Dirichlet boundary nodes were already excluded from the base interface.
        """
        vertex_nodes_global = []
        edge_nodes_global = []

        # IMPORTANT:
        # Use full_interface_nodes_global (base-class G), not self.interface_nodes_global
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

        # Keep full interface G in the original base-class ordering
        self.interface_nodes_global = np.array(self.full_interface_nodes_global, dtype=int, copy=True)
        self.interface_dof_global = np.array(self.full_interface_dof_global, dtype=int, copy=True)

        # local split for each subdomain
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
        """
        Build:
          R_GE : G -> E
          R_GV : G -> V

        with optional 1/2 scaling on R_GE.
        """
        scale_ge = 0.5 if self.scale_R_GE else 1.0

        self.R_GE = self._make_subset_restriction(
            self.edge_interface_dof_global,
            self.interface_dof_global,
            scale=scale_ge,
        )

        self.R_GV = self._make_subset_restriction(
            self.vertex_dof_global,
            self.interface_dof_global,
            scale=1.0,
        )

    def _build_subdomain_interface_restrictions(self):
        """
        Build:
          R_Gi : G  -> G_i
          R_Ei : E  -> E_i
          R_Vi : V  -> V_i

        where:
          - G is the full unique global interface
          - E is the unique global edge set
          - V is the unique global vertex set
          - G_i, E_i, V_i use local subdomain ordering
        """
        self.sd_R_Gi = {}
        self.sd_R_Ei = {}
        self.sd_R_Vi = {}

        self.sd_G_dofs = {}
        self.sd_edge_dofs = {}
        self.sd_vertex_dofs = {}

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

            # G -> G_i
            self.sd_R_Gi[i_sd] = self._make_subset_restriction(
                G_dofs_global,
                self.interface_dof_global,
                scale=1.0,
            )

            # E -> E_i
            self.sd_R_Ei[i_sd] = self._make_subset_restriction(
                E_dofs_global,
                self.edge_interface_dof_global,
                scale=1.0,
            )

            # V -> V_i
            self.sd_R_Vi[i_sd] = self._make_subset_restriction(
                V_dofs_global,
                self.vertex_dof_global,
                scale=1.0,
            )

    def build_subdomain_interface_blocks(self):
        """
        Override base class splitting with FETI-DP splitting.

        Build for each subdomain:
          I = interior nodes
          E = edge/interface nodes (shared by exactly 2 subdomains)
          V = vertex nodes        (shared by 3+ subdomains)

        Then build all 3x3 matrix blocks:
            [A_II A_IE A_IV
             A_EI A_EE A_EV
             A_VI A_VE A_VV]

        Also builds global/subdomain restriction operators with:
          G = full global interface = E union V
        """
        if len(self.sd_kmat) != self.num_subdomains:
            self._assemble_subdomain_systems(bcs=True)

        self._classify_fetidp_nodes()

        # build new G/E/V restrictions
        self._build_global_interface_restrictions()
        self._build_subdomain_interface_restrictions()

        self.sd_interior_dofs = {}
        self.sd_R_interior = {}

        # old local restrictions from local full subdomain vector -> subset
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
            self.sd_interface_dofs[i_sd] = G_dofs   # now use full local interface G_i
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

    def get_interface_rhs(self):
        scale = 0.5 if self.scale_R_GE else 1.0
        n_edge = self.edge_interface_dof_global.shape[0]
        interface_rhs = np.zeros(n_edge)

        # fine DOF d_Lam term
        for i_subdomain in range(self.num_subdomains):
            B_D = scale * self.sd_R_Ei[i_subdomain]
            sd_rhs = self.sd_force[i_subdomain]
            interior_dof = self.sd_interior_dofs[i_subdomain]
            edge_dof = self.sd_edge_dofs[i_subdomain]
            fI = sd_rhs[interior_dof]
            fE = sd_rhs[edge_dof]
            nI = self.sd_A_II[i_subdomain].shape[0]
            nE = self.sd_A_EE[i_subdomain].shape[0]
            if nE == 0:
                continue

            rhs_ie = np.zeros(nI + nE, dtype=np.double)
            rhs_ie[:nI] = fI
            rhs_ie[nI:] = fE

            A_IE2 = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain]],
                ],
                format="csc",
            )

            sol_ie = sp.linalg.spsolve(A_IE2, rhs_ie)
            i_edge_soln = sol_ie[nI:]
            # print(f"{B_D.shape=} {i_edge_soln.shape=} {edge_dof.shape=}")

            interface_rhs += B_D.T.dot(i_edge_soln)

        # coarse DOF (vertex), g_Pi term
        for i_subdomain in range(self.num_subdomains):
            B_D = scale * self.sd_R_Ei[i_subdomain]
            sd_rhs = self.sd_force[i_subdomain]
            interior_dof = self.sd_interior_dofs[i_subdomain]
            edge_dof = self.sd_edge_dofs[i_subdomain]
            vertex_dof = self.sd_vertex_dofs[i_subdomain]
            fI = sd_rhs[interior_dof]
            fE = sd_rhs[edge_dof]
            fV = sd_rhs[vertex_dof]
            # do coarse DOF solve to get I and E DOF
            nI = self.sd_A_II[i_subdomain].shape[0]
            nE = self.sd_A_EE[i_subdomain].shape[0]
            if nE == 0:
                continue
            rhs_ie = np.zeros(nI + nE, dtype=np.double)
            rhs_ie[:nI] = fI
            rhs_ie[nI:] = fE
            A_IE2 = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain]],
                ],
                format="csc",
            )
            sol_ie = sp.linalg.spsolve(A_IE2, rhs_ie)

            # now compute {I,E} => V product with A matrix for RHS
            temp_I = sol_ie[:nI]
            temp_E = sol_ie[nI:]
            # here is g_Pi
            g_PI = fV
            g_PI -= self.sd_A_VI[i_subdomain].dot(temp_I)
            g_PI -= self.sd_A_VE[i_subdomain].dot(temp_E)
            
            # now perform S_{VV}^{-1} solve (coarse solve of Schur complement for vertices)
            # using full subdomain {I,E,V} 3x3 system with RHS only in V entry (inverse of Schur complement)
            nV = self.sd_A_VV[i_subdomain].shape[0]
            rhs_iev = np.zeros(nI + nE + nV, dtype=np.double)
            rhs_iev[(nI+nE):] = g_PI.copy()
            A_IEV = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain], self.sd_A_IV[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain], self.sd_A_EV[i_subdomain]],
                    [self.sd_A_VI[i_subdomain], self.sd_A_VE[i_subdomain], self.sd_A_VV[i_subdomain]],
                ],
                format="csc",
            )
            sol_iev = sp.linalg.spsolve(A_IEV, rhs_iev)
            # get only the V part out of it (Schur complement inverse)
            temp_V2 = sol_iev[(nI+nE):]

            # now do V -> {I,E} then E step (transpose of above)
            temp_I2 = self.sd_A_IV[i_subdomain].dot(temp_V2)
            temp_E2 = self.sd_A_EV[i_subdomain].dot(temp_V2)
            rhs_ie2 = np.concatenate([temp_I2, temp_E2], axis=0)
            sol_ie2 = sp.linalg.spsolve(A_IE2, rhs_ie2)
            temp_E3 = sol_ie2[nI:]

            # now add into global edge solution
            # interface_rhs += B_D.T.dot(temp_E3)
            interface_rhs -= B_D.T.dot(temp_E3) # is this term right?

        return interface_rhs

    def get_global_solution(self, interface_soln: np.ndarray):
        global_soln = np.zeros(self.N)
        global_soln[self.edge_interface_dof_global] = interface_soln.copy()

        for i_subdomain in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_subdomain]

            interior_dof = self.sd_interior_dofs[i_subdomain]
            edge_dof = self.sd_edge_dofs[i_subdomain]
            vertex_dof = self.sd_vertex_dofs[i_subdomain]

            fI = sd_rhs[interior_dof].copy()
            fV = sd_rhs[vertex_dof].copy()

            R_E = self.sd_R_Ei[i_subdomain]
            uE = R_E.dot(interface_soln)

            fI -= self.sd_A_IE[i_subdomain].dot(uE)
            fV -= self.sd_A_VE[i_subdomain].dot(uE)

            nI = self.sd_A_II[i_subdomain].shape[0]
            nV = self.sd_A_VV[i_subdomain].shape[0]

            rhs_iv = np.zeros(nI + nV, dtype=np.double)
            rhs_iv[:nI] = fI
            rhs_iv[nI:] = fV

            A_IV = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IV[i_subdomain]],
                    [self.sd_A_VI[i_subdomain], self.sd_A_VV[i_subdomain]],
                ],
                format="csc",
            )

            sol_iv = sp.linalg.spsolve(A_IV, rhs_iv)
            uI = sol_iv[:nI]
            uV = sol_iv[nI:]

            sd_global_dofs = self.get_subdomain_global_dofs(i_subdomain)
            global_I_dofs = sd_global_dofs[interior_dof]
            global_V_dofs = sd_global_dofs[vertex_dof]

            global_soln[global_I_dofs] = uI
            global_soln[global_V_dofs] = uV

        return global_soln

    def precond_solve(self, edge_rhs: np.ndarray):
        # TODO : check if the precond solve part here is correct.. not sure honestly
        # this is S_{Gam} or S_{Delta} from https://cs.nyu.edu/~widlund/li_widlund_041211.pdf
        # most confusing step of FETI-DP probably (inverse of inverse matrix, cause flipped from BDDC)

        scale = 0.5 if self.scale_R_GE else 1.0
        edge_soln = np.zeros_like(edge_rhs)

        # fine DOF edge solve
        for i_subdomain in range(self.num_subdomains):
            B_D = scale * self.sd_R_Ei[i_subdomain]
            i_edge_rhs = B_D.dot(edge_rhs)

            nI = self.sd_A_II[i_subdomain].shape[0]
            nE = self.sd_A_EE[i_subdomain].shape[0]

            if nE == 0:
                continue

            rhs_ie = np.zeros(nI + nE, dtype=np.double)
            rhs_ie[nI:] = i_edge_rhs

            A_IE2 = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain]],
                ],
                format="csc",
            )

            # strange that it's not inverse matrix here, but that's because FETI-DP reversed from BDDC
            # see mat-vec has inverse
            sol_ie = A_IE2.dot(rhs_ie)
            i_edge_soln = sol_ie[nI:]

            edge_soln += B_D.T.dot(i_edge_soln)
        return edge_soln

    def mat_vec(self, edge_in: np.ndarray):
        # weird thing about FETI-DP is operators are reversed, so
        # the S^{-1} is the mat-vec.. not precond solve

        scale = 0.5 if self.scale_R_GE else 1.0
        edge_out = np.zeros_like(edge_in)

        # fine DOF edge solve
        for i_subdomain in range(self.num_subdomains):
            B_D = scale * self.sd_R_Ei[i_subdomain]
            i_edge_rhs = B_D.dot(edge_in)

            nI = self.sd_A_II[i_subdomain].shape[0]
            nE = self.sd_A_EE[i_subdomain].shape[0]

            if nE == 0:
                continue

            rhs_ie = np.zeros(nI + nE, dtype=np.double)
            rhs_ie[nI:] = i_edge_rhs

            A_IE2 = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain]],
                ],
                format="csc",
            )

            sol_ie = sp.linalg.spsolve(A_IE2, rhs_ie)
            i_edge_soln = sol_ie[nI:]

            edge_out += B_D.T.dot(i_edge_soln)

        # coarse DOF vertex solve now
        for i_subdomain in range(self.num_subdomains):
            B_D = scale * self.sd_R_Ei[i_subdomain]
            i_edge_rhs = B_D.dot(edge_in)

            # TODO : could probably parallelize this part a bit better (with other guy above)
            # but on GPU these could be done very simultaneously or in separate kernels..
            
            # just cause several extra local solves.. but this kind of works like multigrid though I guess.. 
            # extra transfer steps.. and smoothish solves

            # do coarse DOF solve to get I and E DOF
            nI = self.sd_A_II[i_subdomain].shape[0]
            nE = self.sd_A_EE[i_subdomain].shape[0]
            if nE == 0:
                continue
            rhs_ie = np.zeros(nI + nE, dtype=np.double)
            rhs_ie[nI:] = i_edge_rhs
            A_IE2 = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain]],
                ],
                format="csc",
            )
            sol_ie = sp.linalg.spsolve(A_IE2, rhs_ie)

            # now compute {I,E} => V product with A matrix for RHS
            temp_I = sol_ie[:nI]
            temp_E = sol_ie[nI:]
            temp_V = self.sd_A_VI[i_subdomain].dot(temp_I)
            temp_V += self.sd_A_VE[i_subdomain].dot(temp_E)
            
            # now perform S_{VV}^{-1} solve (coarse solve of Schur complement for vertices)
            # using full subdomain {I,E,V} 3x3 system with RHS only in V entry (inverse of Schur complement)
            nV = self.sd_A_VV[i_subdomain].shape[0]
            rhs_iev = np.zeros(nI + nE + nV, dtype=np.double)
            rhs_iev[(nI+nE):] = temp_V.copy()
            A_IEV = sp.bmat(
                [
                    [self.sd_A_II[i_subdomain], self.sd_A_IE[i_subdomain], self.sd_A_IV[i_subdomain]],
                    [self.sd_A_EI[i_subdomain], self.sd_A_EE[i_subdomain], self.sd_A_EV[i_subdomain]],
                    [self.sd_A_VI[i_subdomain], self.sd_A_VE[i_subdomain], self.sd_A_VV[i_subdomain]],
                ],
                format="csc",
            )
            sol_iev = sp.linalg.spsolve(A_IEV, rhs_iev)
            # get only the V part out of it (Schur complement inverse)
            temp_V2 = sol_iev[(nI+nE):]

            # now do V -> {I,E} then E step (transpose of above)
            temp_I2 = self.sd_A_IV[i_subdomain].dot(temp_V2)
            temp_E2 = self.sd_A_EV[i_subdomain].dot(temp_V2)
            rhs_ie2 = np.concatenate([temp_I2, temp_E2], axis=0)
            sol_ie2 = sp.linalg.spsolve(A_IE2, rhs_ie2)
            temp_E3 = sol_ie2[nI:]

            # now add into global edge solution
            edge_out += B_D.T.dot(temp_E3)

        return edge_out
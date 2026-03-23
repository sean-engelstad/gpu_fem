import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .bddc import BDDC_Assembler


class BDDC_EdgeAvg_Assembler(BDDC_Assembler):
    """
    BDDC with vertex primal DOFs + one edge-average primal DOF per
    interface edge per nodal component.

    Important:
    - The Krylov unknown is STILL the physical assembled interface vector
          uGamma = [uE_global ; uV_global]
      exactly like the existing BDDC class.

    - Only the preconditioner changes: it uses a partially assembled space
      with local transformed edge-dual variables and global primal variables
          p = [uA_global ; uV_global]
      where uA_global are the edge-average primal DOFs.

    Assumptions
    -----------
    1) Structured 2D subdomains: each neighboring subdomain pair shares
       exactly one interface edge.
    2) Local edge-node ordering inherited from self.sd_edge_nodes[i_sd]
       is used for the transform.
    3) One average is imposed per DOF component along each geometric edge.
    """

    def __init__(self, *args, edge_elim_policy="middle", **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_elim_policy = edge_elim_policy

        # per-subdomain geometric edge grouping
        self.sd_edge_groups_local_nodes = {}   # i_sd -> [local-node-array per edge]
        self.sd_edge_groups_global_nodes = {}  # i_sd -> [global-node-array per edge]
        self.sd_edge_neighbor = {}             # i_sd -> [neighbor sd per edge]

        # global edge-average indexing
        self.global_edge_keys = []             # sorted owner-pair list
        self.global_edge_key_to_id = {}        # (ilo,ihi) -> iedge
        self.edge_avg_dof_global = np.array([], dtype=int)

        # local avg dof numbering and restrictions
        self.sd_avg_dofs_local = {}            # i_sd -> local avg dof ids (purely local numbering for transformed system)
        self.sd_avg_dofs_global = {}           # i_sd -> corresponding global avg dof ids
        self.sd_R_Ai = {}                      # global avg -> local avg restriction

        # local transforms
        self.sd_Td = {}                        # physical edge -> transformed dual edge
        self.sd_Ta = {}                        # physical edge -> local edge-average primal

        # local interface Schur in transformed coordinates
        self.sd_SEE_phys = {}
        self.sd_SEV_phys = {}
        self.sd_SVE_phys = {}
        self.sd_SVV_phys = {}

        self.sd_Sdd = {}
        self.sd_Sda = {}
        self.sd_Sdv = {}

        self.sd_Sad = {}
        self.sd_Saa = {}
        self.sd_Sav = {}

        self.sd_Svd = {}
        self.sd_Sva = {}
        self.sd_Svv = {}

        self.sd_solver_Sdd = {}

        # combined primal coarse operator on [A_global ; V_global]
        self.S_P = None
        self.S_P_solver = None

    # ============================================================
    # build
    # ============================================================

    def build_subdomain_interface_blocks(self):
        """
        Reuse the parent build for I/E/V extraction and physical-interface
        operator, then build the edge-average transformed preconditioner data.
        """
        super().build_subdomain_interface_blocks()

        self._build_edge_groups()
        self._build_edge_average_indexing()
        self._build_edge_transforms()
        self._build_local_interface_schur_blocks()
        self._build_transformed_local_blocks()
        self._build_edgeavg_coarse_operator()

    # ============================================================
    # edge grouping / indexing
    # ============================================================

    def _build_edge_groups(self):
        """
        Group each subdomain's local edge nodes by neighboring subdomain.

        Because the decomposition is structured, each neighbor pair corresponds
        to one geometric interface edge.
        """
        self.sd_edge_groups_local_nodes = {}
        self.sd_edge_groups_global_nodes = {}
        self.sd_edge_neighbor = {}

        for i_sd in range(self.num_subdomains):
            local_edge_nodes = np.asarray(self.sd_edge_nodes[i_sd], dtype=int)

            by_neighbor_local = {}
            by_neighbor_global = {}

            for lnode in local_edge_nodes:
                gnode = int(self.sd_node_inv_map[i_sd][int(lnode)])
                owners = sorted(self.node_to_subdomains[gnode])

                if len(owners) != 2:
                    raise ValueError(
                        f"Expected edge node {gnode} to have exactly 2 owners, got {owners}"
                    )

                nbr = owners[1] if owners[0] == i_sd else owners[0]

                by_neighbor_local.setdefault(nbr, []).append(int(lnode))
                by_neighbor_global.setdefault(nbr, []).append(int(gnode))

            nbrs = sorted(by_neighbor_local.keys())
            self.sd_edge_neighbor[i_sd] = nbrs
            self.sd_edge_groups_local_nodes[i_sd] = [
                np.array(by_neighbor_local[nbr], dtype=int) for nbr in nbrs
            ]
            self.sd_edge_groups_global_nodes[i_sd] = [
                np.array(by_neighbor_global[nbr], dtype=int) for nbr in nbrs
            ]

    def _build_edge_average_indexing(self):
        """
        One global edge-average primal block per neighboring subdomain pair.
        Each block has size dof_per_node (one average per nodal component).
        """
        edge_keys = set()
        for i_sd in range(self.num_subdomains):
            for nbr in self.sd_edge_neighbor[i_sd]:
                edge_keys.add(tuple(sorted((i_sd, nbr))))

        self.global_edge_keys = sorted(edge_keys)
        self.global_edge_key_to_id = {k: i for i, k in enumerate(self.global_edge_keys)}

        nA_edges = len(self.global_edge_keys)
        b = self.dof_per_node
        self.edge_avg_dof_global = np.arange(nA_edges * b, dtype=int)

        self.sd_avg_dofs_local = {}
        self.sd_avg_dofs_global = {}
        self.sd_R_Ai = {}

        for i_sd in range(self.num_subdomains):
            nbrs = self.sd_edge_neighbor[i_sd]
            nlocA = len(nbrs) * b

            local_avg = np.arange(nlocA, dtype=int)
            global_avg = np.zeros(nlocA, dtype=int)

            for ie, nbr in enumerate(nbrs):
                edge_key = tuple(sorted((i_sd, nbr)))
                gid = self.global_edge_key_to_id[edge_key]
                for k in range(b):
                    global_avg[ie * b + k] = gid * b + k

            self.sd_avg_dofs_local[i_sd] = local_avg
            self.sd_avg_dofs_global[i_sd] = global_avg
            self.sd_R_Ai[i_sd] = self._make_subset_restriction(
                global_avg, self.edge_avg_dof_global, scale=1.0
            )

    # ============================================================
    # transforms
    # ============================================================

    def _choose_edge_elim_index(self, m: int):
        """
        Pick which physical node on the edge is eliminated in the transform.
        For m=4 and policy='middle', this gives 2, matching your example.
        """
        if m < 2:
            raise ValueError("Need at least 2 edge nodes to build an edge-average transform")

        if self.edge_elim_policy == "middle":
            return m // 2
        elif self.edge_elim_policy == "last":
            return m - 1
        elif self.edge_elim_policy == "first":
            return 0
        else:
            raise ValueError(f"Unknown edge_elim_policy={self.edge_elim_policy!r}")

    def _edge_group_transform(self, m_nodes: int):
        """
        Build the transform for ONE geometric edge and ONE subdomain.

        Physical edge nodal values u (size m) are represented as
            u = C_d * uhat + C_a * uavg
        where:
            - uhat has size (m-1), zero-average complement
            - uavg is the edge average scalar

        We eliminate one chosen physical node index j*.
        For every non-eliminated node j:
            u_j = uhat_j + uavg
        and the eliminated one is set so the sum of complements is zero:
            u_j* = uavg - sum(uhat_j)

        This is exactly your 4-node example if j*=2.
        """
        jstar = self._choose_edge_elim_index(m_nodes)

        keep = [j for j in range(m_nodes) if j != jstar]
        Cd = np.zeros((m_nodes, m_nodes - 1), dtype=np.double)
        Ca = np.ones((m_nodes, 1), dtype=np.double)

        for col, j in enumerate(keep):
            Cd[j, col] = 1.0
            Cd[jstar, col] = -1.0

        return Cd, Ca, jstar

    def _build_edge_transforms(self):
        """
        Build Td_i and Ta_i for each subdomain, mapping local PHYSICAL edge DOFs to:
          - transformed local dual edge DOFs
          - local edge-average primal DOFs
        """
        self.sd_Td = {}
        self.sd_Ta = {}

        b = self.dof_per_node

        for i_sd in range(self.num_subdomains):
            E_nodes_local = np.asarray(self.sd_edge_nodes[i_sd], dtype=int)
            E_dofs_local = np.asarray(self.sd_edge_dofs[i_sd], dtype=int)
            ndof_E = len(E_dofs_local)

            if ndof_E == 0:
                self.sd_Td[i_sd] = sp.csr_matrix((0, 0))
                self.sd_Ta[i_sd] = sp.csr_matrix((0, 0))
                continue

            # local node -> physical edge-node block position
            edge_node_pos = {int(ln): j for j, ln in enumerate(E_nodes_local)}

            td_rows, td_cols, td_vals = [], [], []
            ta_rows, ta_cols, ta_vals = [], [], []

            dual_col_offset = 0
            avg_col_offset = 0

            for edge_nodes in self.sd_edge_groups_local_nodes[i_sd]:
                m = len(edge_nodes)
                Cd, Ca, _ = self._edge_group_transform(m)

                # Expand node-level transform to DOF-level transform, DOF component by DOF component
                for comp in range(b):
                    # dual columns for this component on this edge
                    for rnode_local_idx, lnode in enumerate(edge_nodes):
                        prow = edge_node_pos[int(lnode)] * b + comp

                        # dual part
                        for cnode in range(m - 1):
                            val = Cd[rnode_local_idx, cnode]
                            if val != 0.0:
                                td_rows.append(prow)
                                td_cols.append(dual_col_offset + cnode)
                                td_vals.append(val)

                        # avg part
                        val = Ca[rnode_local_idx, 0]
                        if val != 0.0:
                            ta_rows.append(prow)
                            ta_cols.append(avg_col_offset + comp)
                            ta_vals.append(val)

                    dual_col_offset += (m - 1)
                avg_col_offset += b

            ndual = dual_col_offset
            navg = avg_col_offset

            self.sd_Td[i_sd] = sp.csr_matrix(
                (np.array(td_vals, dtype=np.double),
                 (np.array(td_rows, dtype=int), np.array(td_cols, dtype=int))),
                shape=(ndof_E, ndual),
            )
            self.sd_Ta[i_sd] = sp.csr_matrix(
                (np.array(ta_vals, dtype=np.double),
                 (np.array(ta_rows, dtype=int), np.array(ta_cols, dtype=int))),
                shape=(ndof_E, navg),
            )

    # ============================================================
    # physical Schur blocks on (E,V), with I eliminated
    # ============================================================

    def _dense_solve_multi_rhs(self, solver, B):
        """
        Helper for solving A X = B with multiple RHS columns using the solver API
        already used elsewhere in your code.
        """
        B = np.asarray(B, dtype=np.double)
        if B.ndim == 1:
            return np.asarray(solver.solve(B), dtype=np.double)

        X = np.zeros_like(B, dtype=np.double)
        for j in range(B.shape[1]):
            X[:, j] = np.asarray(solver.solve(B[:, j]), dtype=np.double)
        return X

    def _build_local_interface_schur_blocks(self):
        """
        Build physical local interface Schur blocks
            S_i on (E,V)
        after eliminating I only.
        """
        self.sd_SEE_phys = {}
        self.sd_SEV_phys = {}
        self.sd_SVE_phys = {}
        self.sd_SVV_phys = {}

        for i_sd in range(self.num_subdomains):
            nI = self.sd_A_II[i_sd].shape[0]
            nE = self.sd_A_EE[i_sd].shape[0]
            nV = self.sd_A_VV[i_sd].shape[0]

            if nE == 0 and nV == 0:
                self.sd_SEE_phys[i_sd] = sp.csc_matrix((0, 0))
                self.sd_SEV_phys[i_sd] = sp.csc_matrix((0, 0))
                self.sd_SVE_phys[i_sd] = sp.csc_matrix((0, 0))
                self.sd_SVV_phys[i_sd] = sp.csc_matrix((0, 0))
                continue

            AEE = self.sd_A_EE[i_sd].tocsc()
            AEV = self.sd_A_EV[i_sd].tocsc()
            AVE = self.sd_A_VE[i_sd].tocsc()
            AVV = self.sd_A_VV[i_sd].tocsc()

            if nI == 0:
                SEE = AEE
                SEV = AEV
                SVE = AVE
                SVV = AVV
            else:
                X_IE = self._dense_solve_multi_rhs(self.sd_solver_II[i_sd], self.sd_A_IE[i_sd].toarray())
                X_IV = self._dense_solve_multi_rhs(self.sd_solver_II[i_sd], self.sd_A_IV[i_sd].toarray())

                SEE = AEE - self.sd_A_EI[i_sd].dot(X_IE)
                SEV = AEV - self.sd_A_EI[i_sd].dot(X_IV)
                SVE = AVE - self.sd_A_VI[i_sd].dot(X_IE)
                SVV = AVV - self.sd_A_VI[i_sd].dot(X_IV)

            self.sd_SEE_phys[i_sd] = sp.csc_matrix(SEE)
            self.sd_SEV_phys[i_sd] = sp.csc_matrix(SEV)
            self.sd_SVE_phys[i_sd] = sp.csc_matrix(SVE)
            self.sd_SVV_phys[i_sd] = sp.csc_matrix(SVV)

    # ============================================================
    # transformed blocks
    # ============================================================

    def _build_transformed_local_blocks(self):
        """
        Build transformed local interface blocks:
            [Sdd Sda Sdv]
            [Sad Saa Sav]
            [Svd Sva Svv]
        where:
            d = transformed local dual edge variables
            a = local edge-average primal variables
            v = local vertex primal variables
        """
        self.sd_Sdd = {}
        self.sd_Sda = {}
        self.sd_Sdv = {}

        self.sd_Sad = {}
        self.sd_Saa = {}
        self.sd_Sav = {}

        self.sd_Svd = {}
        self.sd_Sva = {}
        self.sd_Svv = {}

        self.sd_solver_Sdd = {}

        for i_sd in range(self.num_subdomains):
            Td = self.sd_Td[i_sd].tocsc()
            Ta = self.sd_Ta[i_sd].tocsc()

            SEE = self.sd_SEE_phys[i_sd].tocsc()
            SEV = self.sd_SEV_phys[i_sd].tocsc()
            SVE = self.sd_SVE_phys[i_sd].tocsc()
            SVV = self.sd_SVV_phys[i_sd].tocsc()

            Sdd = Td.T @ SEE @ Td
            Sda = Td.T @ SEE @ Ta
            Sdv = Td.T @ SEV

            Sad = Ta.T @ SEE @ Td
            Saa = Ta.T @ SEE @ Ta
            Sav = Ta.T @ SEV

            Svd = SVE @ Td
            Sva = SVE @ Ta
            Svv = SVV

            self.sd_Sdd[i_sd] = Sdd.tocsc()
            self.sd_Sda[i_sd] = Sda.tocsc()
            self.sd_Sdv[i_sd] = Sdv.tocsc()

            self.sd_Sad[i_sd] = Sad.tocsc()
            self.sd_Saa[i_sd] = Saa.tocsc()
            self.sd_Sav[i_sd] = Sav.tocsc()

            self.sd_Svd[i_sd] = Svd.tocsc()
            self.sd_Sva[i_sd] = Sva.tocsc()
            self.sd_Svv[i_sd] = Svv.tocsc()

            if Sdd.shape[0] > 0:
                self.sd_solver_Sdd[i_sd] = self.subdomain_solver_cls(
                    Sdd.tocsc(),
                    **self.subdomain_solver_kwargs,
                )
            else:
                self.sd_solver_Sdd[i_sd] = None

    # ============================================================
    # coarse operator on primal [A ; V]
    # ============================================================

    def _split_primal_global_vec(self, p):
        p = np.asarray(p, dtype=np.double)
        nAg = len(self.edge_avg_dof_global)
        nVg = len(self.vertex_dof_global)
        if p.shape[0] != nAg + nVg:
            raise ValueError(f"primal vector has size {p.shape[0]}, expected {nAg+nVg}")
        return p[:nAg], p[nAg:]

    def _build_edgeavg_coarse_operator(self):
        """
        Assemble coarse operator on global primal variables
            p_global = [uA_global ; uV_global]
        using local Schur complement with transformed dual d eliminated.
        """
        nAg = len(self.edge_avg_dof_global)
        nVg = len(self.vertex_dof_global)
        nPg = nAg + nVg

        Sglob = sp.csc_matrix((nPg, nPg))

        for i_sd in range(self.num_subdomains):
            nA = self.sd_Saa[i_sd].shape[0]
            nV = self.sd_Svv[i_sd].shape[0]
            nD = self.sd_Sdd[i_sd].shape[0]

            # local primal block
            if nA == 0 and nV == 0:
                continue

            if nA > 0 and nV > 0:
                Spp = sp.bmat(
                    [[self.sd_Saa[i_sd], self.sd_Sav[i_sd]],
                     [self.sd_Sva[i_sd], self.sd_Svv[i_sd]]],
                    format="csc",
                )
                Spd = sp.vstack([self.sd_Sad[i_sd], self.sd_Svd[i_sd]], format="csc")
                Sdp = sp.hstack([self.sd_Sda[i_sd], self.sd_Sdv[i_sd]], format="csc")
            elif nA > 0:
                Spp = self.sd_Saa[i_sd].tocsc()
                Spd = self.sd_Sad[i_sd].tocsc()
                Sdp = self.sd_Sda[i_sd].tocsc()
            else:
                Spp = self.sd_Svv[i_sd].tocsc()
                Spd = self.sd_Svd[i_sd].tocsc()
                Sdp = self.sd_Sdv[i_sd].tocsc()

            if nD > 0:
                X = self._dense_solve_multi_rhs(self.sd_solver_Sdd[i_sd], Sdp.toarray())
                Scc = Spp - Spd.dot(X)
            else:
                Scc = Spp.copy()

            RAi = self.sd_R_Ai[i_sd].tocsc()
            RVi = self.sd_R_Vi[i_sd].tocsc()

            if nA > 0 and nV > 0:
                RPi = sp.block_diag((RAi, RVi), format="csc")
            elif nA > 0:
                RPi = RAi
            else:
                RPi = RVi

            Sglob = Sglob + RPi.T @ sp.csc_matrix(Scc) @ RPi

        self.S_P = Sglob.tocsc()

        if self.S_P.shape[0] > 0:
            self.S_P_solver = self.coarse_solver_cls(
                self.S_P.copy(),
                **self.coarse_solver_kwargs,
            )
        else:
            self.S_P_solver = None

    def _solve_edgeavg_coarse(self, rhsP_global):
        rhsP_global = np.asarray(rhsP_global, dtype=np.double)
        if rhsP_global.size == 0:
            return np.zeros(0, dtype=np.double)
        return np.asarray(self.S_P_solver.solve(rhsP_global), dtype=np.double)

    # ============================================================
    # preconditioner
    # ============================================================

    def _solve_local_transformed_dual(self, i_sd, rhs_d):
        rhs_d = np.asarray(rhs_d, dtype=np.double)
        if rhs_d.size == 0:
            return np.zeros(0, dtype=np.double)
        return np.asarray(self.sd_solver_Sdd[i_sd].solve(rhs_d), dtype=np.double)

    def precond_solve(self, rGamma: np.ndarray):
        """
        BDDC preconditioner with edge-average primal constraints.

        Input / output remain in the ORIGINAL physical assembled interface basis:
            rGamma = [rE_global ; rV_global]
            zGamma = [zE_global ; zV_global]

        Internally, the partially assembled inverse uses:
            local transformed dual edge vars d
            global primal vars p = [uA_global ; uV_global]
        """
        rE_g, rV_g = self._split_interface_vec(rGamma)

        nEg = len(self.edge_interface_dof_global)
        nAg = len(self.edge_avg_dof_global)
        nVg = len(self.vertex_dof_global)

        edge_weight = 0.5 if self.scale_R_GE else 1.0

        qd_dict = {}
        qP = np.zeros(nAg + nVg, dtype=np.double)
        qP[nAg:] = rV_g

        # Step 1: distribute physical edge residual to local physical edge spaces,
        # then transform to local (d,a) coordinates.
        for i_sd in range(self.num_subdomains):
            qE_phys_i = edge_weight * self.sd_R_Ei[i_sd].dot(rE_g)

            Td = self.sd_Td[i_sd]
            Ta = self.sd_Ta[i_sd]

            qd_i = Td.T.dot(qE_phys_i) if Td.shape[1] > 0 else np.zeros(0, dtype=np.double)
            qa_i = Ta.T.dot(qE_phys_i) if Ta.shape[1] > 0 else np.zeros(0, dtype=np.double)

            qd_dict[i_sd] = np.asarray(qd_i, dtype=np.double)

            if qa_i.size > 0:
                qP[:nAg] += self.sd_R_Ai[i_sd].T.dot(np.asarray(qa_i, dtype=np.double))

        # Step 2: eliminate local transformed duals from primal rhs
        cP = qP.copy()
        d0_dict = {}

        for i_sd in range(self.num_subdomains):
            d0_i = self._solve_local_transformed_dual(i_sd, qd_dict[i_sd])
            d0_dict[i_sd] = d0_i

            nA = self.sd_Saa[i_sd].shape[0]
            nV = self.sd_Svv[i_sd].shape[0]

            if nA > 0 and nV > 0:
                Spd_i = sp.vstack([self.sd_Sad[i_sd], self.sd_Svd[i_sd]], format="csc")
                RPi = sp.block_diag((self.sd_R_Ai[i_sd], self.sd_R_Vi[i_sd]), format="csc")
            elif nA > 0:
                Spd_i = self.sd_Sad[i_sd].tocsc()
                RPi = self.sd_R_Ai[i_sd].tocsc()
            elif nV > 0:
                Spd_i = self.sd_Svd[i_sd].tocsc()
                RPi = self.sd_R_Vi[i_sd].tocsc()
            else:
                continue

            cP -= RPi.T.dot(Spd_i.dot(d0_i))

        # Step 3: solve coarse primal problem on [A ; V]
        p_global = self._solve_edgeavg_coarse(cP)
        pA_g, pV_g = self._split_primal_global_vec(p_global)

        # Step 4: back-substitute local transformed duals and recover physical edge corrections
        zE = np.zeros(nEg, dtype=np.double)

        for i_sd in range(self.num_subdomains):
            a_i = self.sd_R_Ai[i_sd].dot(pA_g) if len(self.sd_avg_dofs_local[i_sd]) > 0 else np.zeros(0, dtype=np.double)
            v_i = self.sd_R_Vi[i_sd].dot(pV_g) if len(self.sd_vertex_dofs[i_sd]) > 0 else np.zeros(0, dtype=np.double)

            nA = a_i.size
            nV = v_i.size

            if nA > 0 and nV > 0:
                p_i = np.concatenate([a_i, v_i])
                Sdp_i = sp.hstack([self.sd_Sda[i_sd], self.sd_Sdv[i_sd]], format="csc")
            elif nA > 0:
                p_i = a_i
                Sdp_i = self.sd_Sda[i_sd].tocsc()
            elif nV > 0:
                p_i = v_i
                Sdp_i = self.sd_Sdv[i_sd].tocsc()
            else:
                p_i = np.zeros(0, dtype=np.double)
                Sdp_i = sp.csc_matrix((self.sd_Sdd[i_sd].shape[0], 0))

            rhs_d = qd_dict[i_sd] - Sdp_i.dot(p_i)
            d_i = self._solve_local_transformed_dual(i_sd, rhs_d)

            uE_phys_i = np.zeros(self.sd_A_EE[i_sd].shape[0], dtype=np.double)
            if self.sd_Td[i_sd].shape[1] > 0:
                uE_phys_i += self.sd_Td[i_sd].dot(d_i)
            if self.sd_Ta[i_sd].shape[1] > 0:
                uE_phys_i += self.sd_Ta[i_sd].dot(a_i)

            if uE_phys_i.size > 0:
                zE += edge_weight * self.sd_R_Ei[i_sd].T.dot(uE_phys_i)

        zV = pV_g.copy()
        return np.concatenate([zE, zV])
import sys
from .inexact import ExactSparseSolver
# from _assembler import Subdomain2DAssembler
from .assembler import FETIDP_Assembler

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class BDDC_Assembler(FETIDP_Assembler):
    """
    BDDC demo with:
      - primal DOF = vertex DOFs
      - dual DOF   = edge/interface DOFs (shared by exactly 2 subdomains)
      - lambda space indexed one-to-one with unique global edge DOFs

    FETI-DP solver was written first (this one is based off it)

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
        coarse_mode: str = "assembled",
        geometry='plate',
        radius: float = 1.0,
        nye:int=None,

        # NEW options
        subdomain_solver_cls=ExactSparseSolver,
        subdomain_solver_kwargs=None,

        coarse_solver_cls=ExactSparseSolver,
        coarse_solver_kwargs=None,
    ):
        print("WARNING: BDDC doesn't support general BCs yet, only all DOF clamped on Dirichlet nodes..")
        print("\tit can support in future, but then some DOF per node are on interface and some on interior (more tricky and probably will need duplicate nodes with Identity blocks)")
        assert clamped

        FETIDP_Assembler.__init__(
            self, ELEMENT, nxe, E, nu, thick, length, width, load_fcn, clamped, nxs, nys, 
            scale_R_GE=scale_R_GE, coarse_mode=coarse_mode,
            geometry=geometry, radius=radius, nye=nye, 
            subdomain_solver_cls=subdomain_solver_cls,
            subdomain_solver_kwargs=subdomain_solver_kwargs,
            coarse_solver_cls=coarse_solver_cls,
            coarse_solver_kwargs=coarse_solver_kwargs,
        )

    # ============================================
    # BDDC specific utils
    # ============================================

    def _solve_local_I_only(self, i_sd: int, rhs_I: np.ndarray):
        rhs_I = np.asarray(rhs_I, dtype=np.double)
        nI = self.sd_A_II[i_sd].shape[0]
        if nI == 0:
            return np.zeros(0, dtype=np.double)
        return self.sd_solver_II[i_sd].solve(rhs_I)


    def _split_interface_vec(self, uGamma: np.ndarray):
        uGamma = np.asarray(uGamma, dtype=np.double)
        nEg = len(self.edge_interface_dof_global)
        nVg = len(self.vertex_dof_global)
        if uGamma.shape[0] != nEg + nVg:
            raise ValueError(f"interface vector has size {uGamma.shape[0]}, expected {nEg+nVg}")
        return uGamma[:nEg], uGamma[nEg:]


    def _apply_local_interface_schur(self, i_sd: int, wE: np.ndarray, wV: np.ndarray):
        """
        Apply local interface Schur complement on (E,V) after eliminating I only:
            S_i [wE; wV]
        where
            S_i = [A_EE A_EV; A_VE A_VV] - [A_EI; A_VI] A_II^{-1} [A_IE A_IV]
        """
        wE = np.asarray(wE, dtype=np.double)
        wV = np.asarray(wV, dtype=np.double)

        nI = self.sd_A_II[i_sd].shape[0]

        if nI > 0:
            rhsI = -(self.sd_A_IE[i_sd].dot(wE) + self.sd_A_IV[i_sd].dot(wV))
            yI = self.sd_solver_II[i_sd].solve(rhsI)
        else:
            yI = np.zeros(0, dtype=np.double)


        # print(f"{i_sd=}")
        # I_nodes = self.sd_interior_nodes[i_sd]
        # for j in range(len(I_nodes)):
        #     I_node = I_nodes[j]
        #     glob_node = self.sd_nodes[i_sd][I_node]
        #     sub_vec = yI[3*j:(3*j+3)]
        #     print(f"\tinterior soln {j=} {glob_node=} {sub_vec}")

        zE = self.sd_A_EE[i_sd].dot(wE) + self.sd_A_EV[i_sd].dot(wV)
        zV = self.sd_A_VE[i_sd].dot(wE) + self.sd_A_VV[i_sd].dot(wV)

        if nI > 0:
            zE += self.sd_A_EI[i_sd].dot(yI)
            zV += self.sd_A_VI[i_sd].dot(yI)

        return np.asarray(zE, dtype=np.double), np.asarray(zV, dtype=np.double)


    def _apply_bddc_tilde_S_inv(self, qE_dict, qV_global):
        """
        Apply inverse of partially assembled interface operator \tilde{S}^{-1}
        to a rhs consisting of:
        - local edge pieces qE_dict[i_sd]
        - global primal/vertex piece qV_global

        Returns:
        uE_dict : local edge corrections
        uV      : global primal correction
        """
        qV_global = np.asarray(qV_global, dtype=np.double)
        cV = qV_global.copy()

        uE0 = {}
        uI0 = {}

        # First local solves with distributed edge rhs only
        for i_sd in range(self.num_subdomains):
            qEi = np.asarray(qE_dict[i_sd], dtype=np.double)

            uIi, uEi = self._solve_local_IE(
                i_sd,
                np.zeros(len(self.sd_interior_dofs[i_sd]), dtype=np.double),
                qEi,
            )

            uI0[i_sd] = uIi
            uE0[i_sd] = uEi

            if len(self.sd_vertex_dofs[i_sd]) > 0:
                cVi = self.sd_A_VI[i_sd].dot(uIi) + self.sd_A_VE[i_sd].dot(uEi)
                cV -= self.sd_R_Vi[i_sd].T.dot(cVi)

        # coarse/primal solve
        uV = self._solve_coarse(cV) if len(self.vertex_dof_global) > 0 else np.zeros(0, dtype=np.double)

        # harmonic extension of primal correction back into edge space
        uE_dict = {}
        for i_sd in range(self.num_subdomains):
            if len(self.sd_vertex_dofs[i_sd]) > 0:
                uVi = self.sd_R_Vi[i_sd].dot(uV)

                _, corrE = self._solve_local_IE(
                    i_sd,
                    -self.sd_A_IV[i_sd].dot(uVi),
                    -self.sd_A_EV[i_sd].dot(uVi),
                )
                uE_dict[i_sd] = uE0[i_sd] + corrE
            else:
                uE_dict[i_sd] = uE0[i_sd].copy()

        return uE_dict, uV
    
    # =========================================
    # main solver routines now
    # =========================================

    def get_gam_rhs(self):
        """
        For compatibility with existing Krylov driver, return the BDDC interface rhs:
            rhs = [gE_global; gV_global]
        after eliminating interior I only.
        """
        nEg = len(self.edge_interface_dof_global)
        nVg = len(self.vertex_dof_global)

        gE = np.zeros(nEg, dtype=np.double)
        gV = np.zeros(nVg, dtype=np.double)

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = np.asarray(sd_rhs[I], dtype=np.double)
            fE = np.asarray(sd_rhs[E], dtype=np.double)
            fV = np.asarray(sd_rhs[V], dtype=np.double)

            uI = self._solve_local_I_only(i_sd, fI)

            gEi = fE.copy()
            gVi = fV.copy()

            if len(I) > 0:
                gEi -= self.sd_A_EI[i_sd].dot(uI)
                gVi -= self.sd_A_VI[i_sd].dot(uI)

            if len(E) > 0:
                gE += self.sd_R_Ei[i_sd].T.dot(gEi)
            if len(V) > 0:
                gV += self.sd_R_Vi[i_sd].T.dot(gVi)

        # for j in range(nVg//3):
        #     v_node = self.vertex_nodes_global[j]
        #     sub_vec = gV[3*j:(3*j+3)]
        #     print(f"gam vertex {j=} {v_node=} {sub_vec}")

        return np.concatenate([gE, gV])
    
    def mat_vec(self, uGamma: np.ndarray):
        """
        Apply BDDC interface operator:
            y = \hat{S} uGamma
        where
            uGamma = [uE_global; uV_global]
        """
        uE_g, uV_g = self._split_interface_vec(uGamma)

        nEg = len(self.edge_interface_dof_global)
        nVg = len(self.vertex_dof_global)

        yE_g = np.zeros(nEg, dtype=np.double)
        yV_g = np.zeros(nVg, dtype=np.double)

        for i_sd in range(self.num_subdomains):
            uEi = self.sd_R_Ei[i_sd].dot(uE_g)
            uVi = self.sd_R_Vi[i_sd].dot(uV_g)

            zEi, zVi = self._apply_local_interface_schur(i_sd, uEi, uVi)

            if len(uEi) > 0:
                yE_g += self.sd_R_Ei[i_sd].T.dot(zEi)
            if len(uVi) > 0:
                yV_g += self.sd_R_Vi[i_sd].T.dot(zVi)

        # print("matvec-uE")
        # E_nodes = self.edge_interface_nodes_global
        # for j in range(len(E_nodes)):
        #     E_node = E_nodes[j]
        #     sub_vec = yE_g[3*j:(3*j+3)]
        #     print(f"\tedge soln {j=} {E_node=} {sub_vec}")

        # V_nodes = self.vertex_nodes_global
        # print("matvec-uV")
        # for j in range(len(V_nodes)):
        #     v_node = self.vertex_nodes_global[j]
        #     sub_vec = yV_g[3*j:(3*j+3)]
        #     print(f"\tuV vertex {j=} {v_node=} {sub_vec}")

        return np.concatenate([yE_g, yV_g])

    def precond_solve(self, rGamma: np.ndarray):
        """
        BDDC preconditioner:
            M^{-1} r = R_D^T \tilde{S}^{-1} R_D r

        Here the interface vector is
            rGamma = [rE_global; rV_global]
        """
        rE_g, rV_g = self._split_interface_vec(rGamma)

        nEg = len(self.edge_interface_dof_global)

        # simple multiplicity scaling for 2-subdomain edges
        edge_weight = 0.5 if self.scale_R_GE else 1.0

        # distribute global edge residual to local edge spaces
        qE_dict = {}
        for i_sd in range(self.num_subdomains):
            qE_dict[i_sd] = edge_weight * self.sd_R_Ei[i_sd].dot(rE_g)

        # primal variables are already global in this representation
        qV = rV_g.copy()

        uE_dict, uV = self._apply_bddc_tilde_S_inv(qE_dict, qV)

        # weighted averaging back to global edge space
        zE = np.zeros(nEg, dtype=np.double)
        for i_sd in range(self.num_subdomains):
            if len(uE_dict[i_sd]) > 0:
                zE += edge_weight * self.sd_R_Ei[i_sd].T.dot(uE_dict[i_sd])

        zV = uV.copy()

        return np.concatenate([zE, zV])

    def get_global_solution(self, uGamma: np.ndarray):
        """
        Recover physical global displacement from BDDC interface solution
            uGamma = [uE_global; uV_global].
        """
        uE_g, uV_g = self._split_interface_vec(uGamma)

        global_soln = np.zeros(self.N, dtype=np.double)

        # Global interface values are already continuous
        if len(self.edge_interface_dof_global) > 0:
            global_soln[self.edge_interface_dof_global] = uE_g
        if len(self.vertex_dof_global) > 0:
            global_soln[self.vertex_dof_global] = uV_g

        for i_sd in range(self.num_subdomains):
            sd_rhs = self.sd_force[i_sd]

            I = self.sd_interior_dofs[i_sd]
            E = self.sd_edge_dofs[i_sd]
            V = self.sd_vertex_dofs[i_sd]

            fI = np.asarray(sd_rhs[I], dtype=np.double)

            uEi = self.sd_R_Ei[i_sd].dot(uE_g) if len(E) > 0 else np.zeros(0, dtype=np.double)
            uVi = self.sd_R_Vi[i_sd].dot(uV_g) if len(V) > 0 else np.zeros(0, dtype=np.double)

            rhsI = fI.copy()
            if len(E) > 0:
                rhsI -= self.sd_A_IE[i_sd].dot(uEi)
            if len(V) > 0:
                rhsI -= self.sd_A_IV[i_sd].dot(uVi)

            uI = self._solve_local_I_only(i_sd, rhsI)

            sd_global_dofs = self.get_subdomain_global_dofs(i_sd)

            if len(I) > 0:
                global_soln[sd_global_dofs[I]] = uI
            if len(E) > 0:
                global_soln[sd_global_dofs[E]] = uEi
            if len(V) > 0:
                global_soln[sd_global_dofs[V]] = uVi

        return global_soln
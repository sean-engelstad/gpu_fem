import numpy as np
import scipy.sparse as sp

from .basis import (
    second_order_quadrature,
    third_order_quadrature,
    first_order_quadrature,
    get_iga2_basis,   # returns (N, dN) for p=2 on [0,1]
)

# -------------------------
# Helpers: discontinuous p=1 (C^{-1}) strain basis on each element
# -------------------------
def iga1_disc_basis(xi: float):
    """
    Discontinuous p=1 basis on element param domain xi ∈ [0,1].
    Two DOFs per element: gamma_L, gamma_R. Not shared between elements (C^{-1} globally).
    """
    Ng = np.array([1.0 - xi, xi], dtype=float)
    dNg_dxi = np.array([-1.0, 1.0], dtype=float)
    return Ng, dNg_dxi


class HellingerReissnerIsogeometricElement:
    """
    Timoshenko beam in mixed (Hellinger–Reissner / VMS-like) form with
    element-local discontinuous shear strain unknowns.

    Unknowns per element:
      - Nodal (p=2 IGA, 3 basis fns): w_i, th_i, i=0..2  -> 6 DOF
      - Element strain (p=1 discontinuous, 2 basis fns): gamma_L, gamma_R -> 2 DOF
      Total: 8 DOF per element.

    Intended usage:
      - Assemble element matrices including gamma DOFs (8x8).
      - Static condense gamma DOFs OUTSIDE the element to get 6x6 nodal matrix.
      - Global MG operators act only on nodal DOFs (same as your IGA restriction/prolongation).
    """

    def __init__(self, reduced_integrated: bool = False):
        self.reduced_integrated = reduced_integrated
        self.clamped = True

        # Nodal field is p=2 IGA
        self.ORDER = 2
        self.nodes_per_elem = 3
        self.dof_per_node = 2  # [w, th]


        self._P0_cache = {}
        self._P1_cache = {}
        self._kmat_cache = {}

        # Element-local discontinuous strain DOFs
        self.gamma_dof_per_elem = 2  # [gamma_L, gamma_R]

    def get_kelem(self, E: float, nu: float, thick: float,
                  elem_length: float, left_bndry: bool, right_bndry: bool):
        # Quadrature
        bend_pts, bend_wts = second_order_quadrature()
        if self.reduced_integrated:
            shear_pts, shear_wts = first_order_quadrature()
        else:
            shear_pts, shear_wts = second_order_quadrature()

        # Material
        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick

        # Mapping xi∈[0,1] -> x, J = dx/dxi = L
        L = elem_length
        J = L

        # sizes
        nd_u = self.nodes_per_elem * self.dof_per_node  # 6
        nd_g = self.gamma_dof_per_elem                  # 2

        # Build blocks for condensation:
        # [ Kuu Kug ] [u] = [fu]
        # [ Kgu Kgg ] [g]   [fg]
        Kuu = np.zeros((nd_u, nd_u), dtype=float)
        Kug = np.zeros((nd_u, nd_g), dtype=float)
        Kgu = np.zeros((nd_g, nd_u), dtype=float)
        Kgg = np.zeros((nd_g, nd_g), dtype=float)

        # index helpers in u-block
        def iw(a): return 2 * a + 0
        def ith(a): return 2 * a + 1

        # -------------------------
        # Bending: ∫ EI * (theta_x)^2 dx
        # -------------------------
        for _xi, _wt in zip(bend_pts, bend_wts):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            N, dN_dxi = get_iga2_basis(xi, left_bndry, right_bndry)
            dN_dx = dN_dxi / J

            for a in range(3):
                for b in range(3):
                    Kuu[ith(a), ith(b)] += EI * wt * J * dN_dx[a] * dN_dx[b]

        # -------------------------
        # Mixed shear (HRA-style with element-local gamma):
        # gamma(xi) = Ng(xi)·g, Ng = [1-xi, xi]
        #
        # Build:
        #   Kgg += ∫ ksGA * Ng^T Ng
        #   Kug (u->g coupling) from -∫ ksGA * [w_x, theta] * Ng
        # and symmetric Kgu
        #
        # This matches your "strain-int" intent: discontinuous strain field, condensed locally.
        # -------------------------
        for _xi, _wt in zip(shear_pts, shear_wts):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            N, dN_dxi = get_iga2_basis(xi, left_bndry, right_bndry)
            dN_dx = dN_dxi / J

            Ng, _ = iga1_disc_basis(xi)  # (2,)

            c = ksGA * wt * J

            # Kgg
            Kgg[0, 0] += c * Ng[0] * Ng[0]
            Kgg[0, 1] += c * Ng[0] * Ng[1]
            Kgg[1, 0] += c * Ng[1] * Ng[0]
            Kgg[1, 1] += c * Ng[1] * Ng[1]

            # w - g : -∫ ksGA * w_x * gamma
            # th - g: -∫ ksGA * th   * gamma  (your sign convention in strain-int)
            for a in range(3):
                # Kug rows correspond to u-dofs
                Kug[iw(a), 0] += -c * dN_dx[a] * Ng[0]
                Kug[iw(a), 1] += -c * dN_dx[a] * Ng[1]

                Kug[ith(a), 0] += -c * N[a] * Ng[0]
                Kug[ith(a), 1] += -c * N[a] * Ng[1]

                # symmetric Kgu
                Kgu[0, iw(a)] += -c * Ng[0] * dN_dx[a]
                Kgu[1, iw(a)] += -c * Ng[1] * dN_dx[a]

                Kgu[0, ith(a)] += -c * Ng[0] * N[a]
                Kgu[1, ith(a)] += -c * Ng[1] * N[a]

        # -------------------------
        # Static condensation: Kc = Kuu - Kug * inv(Kgg) * Kgu
        # -------------------------
        # Solve Kgg * X = Kgu
        X = np.linalg.solve(Kgg, Kgu)   # (2 x 6)
        Kc = Kuu - Kug @ X              # (6 x 6)

        return Kc

    def get_felem(self, mag, elem_length: float,
              left_bndry: bool, right_bndry: bool, xbnd: list):
        """
        Condensed nodal load vector (no auxiliary gamma DOFs).

        Returned ordering:
        f = [ w0, th0, w1, th1, w2, th2 ]

        mag(x) is your distributed load function (applied to w only).
        """
        pts, wts = third_order_quadrature()
        x0, x1 = xbnd[0], xbnd[1]

        L = elem_length
        J = L  # dx/dxi, xi ∈ [0,1]

        nd_u = self.nodes_per_elem * self.dof_per_node  # 3*2 = 6
        fe = np.zeros(nd_u, dtype=float)

        for _xi, _wt in zip(pts, wts):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            xval = x0 * (1.0 - xi) + x1 * xi
            load_mag = mag(xval)

            N, _ = get_iga2_basis(xi, left_bndry, right_bndry)

            for a in range(3):
                fe[2 * a + 0] += load_mag * wt * J * N[a]  # w only

        return fe

    # -------------------------
    # Static condensation helper (optional convenience)
    # -------------------------
    @staticmethod
    def condense_gamma(ke: np.ndarray, fe: np.ndarray):
        """
        Condense last 2 DOFs (gamma_L, gamma_R) from the element system.

        Returns (Kcond, Fcond) for the first 6 nodal DOFs only.
        """
        Kuu = ke[:6, :6]
        Kug = ke[:6, 6:]
        Kgu = ke[6:, :6]
        Kgg = ke[6:, 6:]

        fu = fe[:6]
        fg = fe[6:]

        # Schur complement: Kc = Kuu - Kug Kgg^{-1} Kgu
        X = np.linalg.solve(Kgg, Kgu)      # (2x6)
        Kc = Kuu - Kug @ X                 # (6x6)

        # Fc = fu - Kug Kgg^{-1} fg
        y = np.linalg.solve(Kgg, fg)       # (2,)
        Fc = fu - Kug @ y                  # (6,)

        return Kc, Fc

    # -------------------------
    # MG restriction/prolongation
    # (acts on nodal DOFs only; gamma is condensed so you never MG it)
    # Reuse your exact operators from the IGA displacement element.
    # -------------------------
    def _build_restriction_matrix(self, nxe_c):
        n_coarse = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_fine = nxe_f + 2
        R = np.zeros((n_coarse, n_fine))
        counts = 1e-20 * np.ones((n_coarse, n_fine))

        for ielem_c in range(nxe_c):
            left_felem = 2 * ielem_c
            l_mat = np.array([
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.75],
                [0.0,  0.0,  0.25],
            ])
            if ielem_c == 0:
                l_mat[0, :2] = np.array([1.0, 0.5])
                l_mat[1, :2] = np.array([0.0, 0.5])
            if ielem_c == nxe_c - 1:
                l_mat[1, 2] = 0.5
                l_mat[2, 2] = 0.5

            l_nz = l_mat / (l_mat + 1e-20)
            R[ielem_c:(ielem_c + 3), left_felem:(left_felem + 3)] += l_mat
            counts[ielem_c:(ielem_c + 3), left_felem:(left_felem + 3)] += l_nz

            right_felem = 2 * ielem_c + 1
            r_mat = np.array([
                [0.25, 0.0,  0.0],
                [0.75, 0.75, 0.25],
                [0.0,  0.25, 0.75],
            ])
            if ielem_c == 0:
                r_mat[0, 0] = 0.5
                r_mat[1, 0] = 0.5
            if ielem_c == nxe_c - 1:
                r_mat[1, 1:] = np.array([0.5, 0.0])
                r_mat[2, 1:] = np.array([0.5, 1.0])

            r_nz = r_mat / (r_mat + 1e-20)
            R[ielem_c:(ielem_c + 3), right_felem:(right_felem + 3)] += r_mat
            counts[ielem_c:(ielem_c + 3), right_felem:(right_felem + 3)] += r_nz

        R /= counts
        return R
    
    def _energy_smooth_prolong_local(self, P0: sp.csr_matrix, nxe_c: int,
                                    block_nodes: int = 1,
                                    n_sweeps: int = 1,
                                    omega: float = 1.0,
                                    tau: float = 1.0):
        """
        Energy smoothing of prolongator using the trace objective:

            E(P) = tr(P^T K_f P)

        Gradient: dE/dP = 2 K_f P.

        Local preconditioned gradient descent / Richardson step:
            P <- P - tau * D^{-1} (K_f P)

        where D is the block-diagonal of K_f with 1- or 2-node blocks.

        Parameters
        ----------
        P0 : csr_matrix
            Baseline prolong, shape (n_f, n_c).
        nxe_c : int
            Coarse elements, used to eliminate coarse BC columns (same convention as your locking routine).
        block_nodes : {1,2}
            1 -> 2x2 blocks (node-local), 2 -> 4x4 blocks (two-node local).
        n_sweeps : int
            Number of smoothing sweeps (each sweep applies one local-preconditioned gradient step).
        omega : float
            Relaxation on the block update (like your locking GS): X <- X + omega*(Xnew - X).
        tau : float
            Step size in the gradient step. Often tau ~ 0.5–1.0 works; if it oversmooths, reduce tau.

        Returns
        -------
        P : dense ndarray, shape (n_f, n_c)
        """

        # fine operator
        K = self._kmat_cache[nxe_c]
        n_f = K.shape[0]

        # coarse column handling (same as your locking routine)
        nx_c = nxe_c + self.ORDER
        n_c = 2 * nx_c
        fixed_cols = [0, 2 * (nx_c - 1)]  # SS: w left/right
        all_cols = np.arange(n_c, dtype=int)
        free_cols = np.array([j for j in all_cols if j not in set(fixed_cols)], dtype=int)

        # print(f"{nx_c=} {n_c=} {P0.shape=}")

        # dense working array
        P0 = P0 #.toarray()
        X = P0[:, free_cols].copy()   # (n_f, n_free)

        # block structure
        if block_nodes not in (1, 2):
            raise ValueError("block_nodes must be 1 or 2")

        dofs_per_node = 2
        block_size = dofs_per_node * block_nodes
        n_blocks = int(np.ceil(n_f / block_size))

        def blk_slice(b):
            i0 = b * block_size
            i1 = min(n_f, (b + 1) * block_size)
            return slice(i0, i1)

        # cache block-diagonal inverses of K (local)
        Dinv = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            Kb = K[sl, sl].toarray()
            Kb = 0.5 * (Kb + Kb.T)
            # small safeguard in case Kb is singular-ish locally (optional)
            # eps = 1e-14 * np.trace(Kb) / max(1, Kb.shape[0])
            # Kb = Kb + eps * np.eye(Kb.shape[0])
            Dinv.append(np.linalg.inv(Kb))

        # print(f"{K.shape=} {X.shape=}")

        # smoothing sweeps
        for _ in range(n_sweeps):
            # compute global gradient piece once: G = K X
            # (sparse-dense multiply; gives dense n_f x n_free)
            GX = K @ X

            # local block-preconditioned update
            for b in range(n_blocks):
                sl = blk_slice(b)

                # preconditioned gradient step on this block:
                # Xnew_b = X_b - tau * Dinv_b * (GX_b)
                Xnew = X[sl, :] - tau * (Dinv[b] @ GX[sl, :])

                # relax like your locking routine
                X[sl, :] = X[sl, :] + omega * (Xnew - X[sl, :])

        # print("ENERGY SMOOTH PROLONG LOCAL")

        # reinsert into full P
        P = np.zeros((n_f, n_c))
        P[:, free_cols] = X
        return P

    # def prolongate(self, coarse_disp: np.ndarray, length: float):
    #     dpn = self.dof_per_node
    #     ndof_coarse = coarse_disp.shape[0]
    #     nnodes_coarse = ndof_coarse // dpn
    #     nelems_coarse = nnodes_coarse - 2

    #     nelems_fine = 2 * nelems_coarse
    #     nnodes_fine = nelems_fine + 2
    #     ndof_fine = nnodes_fine * dpn

    #     R = self._build_restriction_matrix(nelems_coarse)
    #     P0 = R.T
    #     P = P0.copy()

    #     # P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)

    #     fine_disp = np.zeros(ndof_fine)
    #     fine_wts = np.zeros(ndof_fine)

    #     for idof in range(dpn):
    #         fine_disp[idof::dpn] += P @ coarse_disp[idof::dpn]
    #         fine_wts[idof::dpn] += P @ np.ones(nnodes_coarse)

    #     fine_disp /= fine_wts

    #     # Apply BCs (your pattern)
    #     fine_disp[0] = 0.0
    #     fine_disp[-2] = 0.0
    #     if self.clamped:
    #         fine_disp[1] = 0.0
    #         fine_disp[-1] = 0.0

    #     return fine_disp

    # def restrict_defect(self, fine_defect: np.ndarray, length: float):
    #     dpn = self.dof_per_node
    #     ndof_fine = fine_defect.shape[0]
    #     nnodes_fine = ndof_fine // dpn
    #     nelems_fine = nnodes_fine - 2

    #     nelems_coarse = nelems_fine // 2
    #     nnodes_coarse = nelems_coarse + 2
    #     ndof_coarse = nnodes_coarse * dpn

    #     R = self._build_restriction_matrix(nelems_coarse)
    #     P0 = R.T
    #     P = P0.copy()

    #     # P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)


    #     coarse_defect = np.zeros(ndof_coarse)
    #     for idof in range(dpn):
    #         coarse_defect[idof::dpn] += R @ fine_defect[idof::dpn]

    #     # Apply BCs (your pattern)
    #     coarse_defect[0] = 0.0
    #     coarse_defect[-2] = 0.0
    #     if self.clamped:
    #         coarse_defect[1] = 0.0
    #         coarse_defect[-1] = 0.0

    #     return coarse_defect

    def prolongate(self, coarse_disp:np.ndarray, length:float):
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // self.dof_per_node
        nelems_coarse = nnodes_coarse - 2
        nelems_fine = 2 * nelems_coarse
        nnodes_fine = nelems_fine + 2
        ndof_fine = nnodes_fine * self.dof_per_node

        # R = self._build_restriction_matrix(nelems_coarse)
        # P = R.T

        R = self._build_restriction_matrix(nelems_coarse)
        P0 = R.T
        P0_2 = np.zeros((2*P0.shape[0], 2 * P0.shape[1]))
        for i in range(P0.shape[0]):
            for j in range(P0.shape[1]):
                P0_2[2*i, 2*j] = P0[i,j]
                P0_2[2*i+1,2*j+1] = P0[i,j]

        # P = P0.copy()
        P = self._energy_smooth_prolong_local(P0_2, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)


        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # to do row-sum norm on P (since P^T is properly normalized but P is not?)
        dpn = self.dof_per_node
        # for idof in range(dpn):
        #     fine_disp[idof::dpn] += np.dot(P, coarse_disp[idof::dpn])
        #     fine_weights[idof::dpn] += np.dot(P, np.ones(nnodes_coarse))
        fine_disp = np.dot(P, coarse_disp)

        # fine_disp /= fine_weights

        # apply bcs again, no need same answer either way
        fine_disp[0] = 0.0
        fine_disp[-2] = 0.0

        if self.clamped:
            fine_disp[1] = 0.0
            fine_disp[-1] = 0.0
            
        return fine_disp
    
    def restrict_defect(self, fine_defect:np.ndarray, length:float):
        ndof_coarse = ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // self.dof_per_node
        nelems_fine = nnodes_fine - 2

        nelems_coarse = nelems_fine // 2
        nnodes_coarse = nelems_coarse + 2
        dpn = self.dof_per_node
        ndof_coarse = dpn * nnodes_coarse

        # R = self._build_restriction_matrix(nelems_coarse)

        R = self._build_restriction_matrix(nelems_coarse)
        P0 = R.T
        P0_2 = np.zeros((2*P0.shape[0], 2 * P0.shape[1]))
        for i in range(P0.shape[0]):
            for j in range(P0.shape[1]):
                P0_2[2*i, 2*j] = P0[i,j]
                P0_2[2*i+1,2*j+1] = P0[i,j]

        # P = P0.copy()
        P = self._energy_smooth_prolong_local(P0_2, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)
        R = P.T


        coarse_defect = np.zeros(ndof_coarse)
        dpn = self.dof_per_node
        # for idof in range(dpn):
        #     coarse_defect[idof::dpn] += np.dot(R, fine_defect[idof::dpn])
        coarse_defect = np.dot(R, fine_defect)

        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0
        if self.clamped:
            coarse_defect[1] = 0.0
            coarse_defect[-1] = 0.0

        return coarse_defect
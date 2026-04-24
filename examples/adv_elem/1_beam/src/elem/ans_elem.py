import numpy as np
from .basis import (
    second_order_quadrature,
    third_order_quadrature,
    get_iga2_basis,
)
import scipy.sparse as sp


class ANSTimoshenkoIGA2Element:
    """
    2nd-order IGA Timoshenko beam element with assumed natural strain (ANS)
    for the transverse shear term.

    DOFs per node: [w, theta]
    Nodes per element: 3

    Unknown order internally before permutation:
        [w1, w2, w3, th1, th2, th3]

    Returned element order:
        [w1, th1, w2, th2, w3, th3]

    Notes
    -----
    - Bending strain: kappa = d(theta)/dx
    - Shear strain:   gamma = dw/dx - theta
    - ANS shear: gamma is evaluated at tying points and then linearly
      interpolated through the element.
    - Parent coordinate for the IGA basis is xi in [0, 1].
    """

    def __init__(self):
        self.dof_per_node = 2
        self.nodes_per_elem = 3
        self.ORDER = 2
        self.clamped = True

        self._P0_cache = {}
        self._P1_cache = {}
        self._kmat_cache = {}

    def _get_shear_Bmat_at_xi(
        self,
        xi_eval: float,
        elem_length: float,
        left_bndry: bool,
        right_bndry: bool,
    ):
        """
        Standard shear B-matrix at a single xi in [0,1]:
            gamma = [Bw  Bth] * q
                  = sum_i dN_i/dx * w_i - sum_i N_i * th_i
        for q = [w1,w2,w3,th1,th2,th3]^T
        """
        J = elem_length
        N, dN = get_iga2_basis(xi_eval, left_bndry, right_bndry)
        dN = dN / J

        B = np.zeros((1, 6))
        B[0, 0:3] = dN
        B[0, 3:6] = -N
        return B

    def _get_ans_shear_Bmat(
        self,
        xi: float,
        elem_length: float,
        left_bndry: bool,
        right_bndry: bool,
    ):
        """
        Assumed natural shear strain interpolation.

        Use two tying points in the natural interval xi in [0,1]:
            xi_A = 0
            xi_B = 1

        Then interpolate:
            gamma_ans(xi) = (1-xi)*gamma_A + xi*gamma_B
        """
        xi_A = 0.0
        xi_B = 1.0

        B_A = self._get_shear_Bmat_at_xi(
            xi_A, elem_length, left_bndry, right_bndry
        )
        B_B = self._get_shear_Bmat_at_xi(
            xi_B, elem_length, left_bndry, right_bndry
        )

        N_ans_A = 1.0 - xi
        N_ans_B = xi

        B_ans = N_ans_A * B_A + N_ans_B * B_B
        return B_ans

    def get_kelem(
        self,
        E: float,
        nu: float,
        thick: float,
        elem_length: float,
        left_bndry: bool,
        right_bndry: bool,
    ):
        """
        Build 6x6 element stiffness for the ANS IGA2 Timoshenko beam.
        """
        pts, weights = second_order_quadrature()

        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick

        # IGA parent domain is xi in [0,1], so x = x0 + J*xi with J = elem_length
        J = elem_length

        # internal ordering: [w1,w2,w3,th1,th2,th3]
        kelem = np.zeros((6, 6))

        # -------------------------------------------------
        # bending: int EI * theta_x * delta(theta_x) dx
        # -------------------------------------------------
        for _xi, _wt in zip(pts, weights):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            _, dN = get_iga2_basis(xi, left_bndry, right_bndry)
            dN = dN / J

            # kappa = d(theta)/dx
            B_bend = np.zeros((1, 6))
            B_bend[0, 3:6] = dN

            kelem += EI * wt * J * (B_bend.T @ B_bend)

        # -------------------------------------------------
        # ANS shear: int ksGA * gamma_ans * delta(gamma_ans) dx
        # -------------------------------------------------
        for _xi, _wt in zip(pts, weights):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            B_shear = self._get_ans_shear_Bmat(
                xi, elem_length, left_bndry, right_bndry
            )

            kelem += ksGA * wt * J * (B_shear.T @ B_shear)

        # reorder [w1,w2,w3,th1,th2,th3] -> [w1,th1,w2,th2,w3,th3]
        new_order = np.array([0, 3, 1, 4, 2, 5])
        kelem = kelem[new_order, :][:, new_order]
        return kelem

    def get_felem(self, mag, elem_length, left_bndry: bool, right_bndry: bool, xbnd: list):
        """
        Distributed load element vector.

        Load acts on w only, not theta.
        `mag` can be a callable mag(x) or a scalar.
        """
        J = elem_length
        pts, wts = third_order_quadrature()
        x0, x1 = xbnd[0], xbnd[1]

        felem = np.zeros(6)

        for _xi, _wt in zip(pts, wts):
            xi = 0.5 * (_xi + 1.0)
            wt = 0.5 * _wt

            xval = x0 * (1.0 - xi) + x1 * xi
            load_mag = mag(xval) if callable(mag) else mag

            N, _ = get_iga2_basis(xi, left_bndry, right_bndry)

            # only w DOFs get transverse load
            for i in range(3):
                felem[2 * i] += load_mag * wt * J * N[i]

        return felem
    
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

    def _build_restriction_matrix(self, nxe_c):
        """
        Same quadratic IGA coarse/fine transfer operator structure you were
        already using for the hierarchic IGA element.
        """
        n_coarse = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_fine = nxe_f + 2

        R = np.zeros((n_coarse, n_fine))
        counts = 1e-20 * np.ones((n_coarse, n_fine))

        for ielem_c in range(nxe_c):
            # left child
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

            l_nz_mat = l_mat / (l_mat + 1e-20)
            R[ielem_c:ielem_c + 3, left_felem:left_felem + 3] += l_mat
            counts[ielem_c:ielem_c + 3, left_felem:left_felem + 3] += l_nz_mat

            # right child
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

            r_nz_mat = r_mat / (r_mat + 1e-20)
            R[ielem_c:ielem_c + 3, right_felem:right_felem + 3] += r_mat
            counts[ielem_c:ielem_c + 3, right_felem:right_felem + 3] += r_nz_mat

        R /= counts
        return R

    # def prolongate(self, coarse_disp: np.ndarray, length: float):
    #     """
    #     Prolongation for both w and theta using the same quadratic IGA transfer.
    #     """
    #     ndof_coarse = coarse_disp.shape[0]
    #     nnodes_coarse = ndof_coarse // self.dof_per_node
    #     nelems_coarse = nnodes_coarse - 2

    #     nelems_fine = 2 * nelems_coarse
    #     nnodes_fine = nelems_fine + 2
    #     ndof_fine = nnodes_fine * self.dof_per_node

    #     R = self._build_restriction_matrix(nelems_coarse)
    #     P = R.T

    #     fine_disp = np.zeros(ndof_fine)
    #     fine_weights = np.zeros(ndof_fine)

    #     dpn = self.dof_per_node
    #     for idof in range(dpn):
    #         fine_disp[idof::dpn] += P @ coarse_disp[idof::dpn]
    #         fine_weights[idof::dpn] += P @ np.ones(nnodes_coarse)

    #     fine_disp /= fine_weights

    #     # essential BC handling
    #     fine_disp[0] = 0.0
    #     fine_disp[-2] = 0.0
    #     if self.clamped:
    #         fine_disp[-2] = 0.0
    #         fine_disp[-1] = 0.0

    #     return fine_disp

    # def restrict_defect(self, fine_defect: np.ndarray, length: float):
    #     """
    #     Restriction for both w and theta using the same quadratic IGA transfer.
    #     """
    #     ndof_fine = fine_defect.shape[0]
    #     nnodes_fine = ndof_fine // self.dof_per_node
    #     nelems_fine = nnodes_fine - 2

    #     nelems_coarse = nelems_fine // 2
    #     nnodes_coarse = nelems_coarse + 2
    #     ndof_coarse = nnodes_coarse * self.dof_per_node

    #     R = self._build_restriction_matrix(nelems_coarse)

    #     coarse_defect = np.zeros(ndof_coarse)
    #     dpn = self.dof_per_node

    #     for idof in range(dpn):
    #         coarse_defect[idof::dpn] += R @ fine_defect[idof::dpn]

    #     # essential BC handling
    #     coarse_defect[0] = 0.0
    #     coarse_defect[-2] = 0.0
    #     if self.clamped:
    #         coarse_defect[-2] = 0.0
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

        P = P0_2.copy()
        P = self._energy_smooth_prolong_local(P0_2, nelems_coarse, block_nodes=1, 
                                              n_sweeps=2, omega=0.5, tau=0.5)


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

        P = P0_2.copy()
        P = self._energy_smooth_prolong_local(P0_2, nelems_coarse, block_nodes=1, 
                                              n_sweeps=2, omega=0.5, tau=0.5)
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
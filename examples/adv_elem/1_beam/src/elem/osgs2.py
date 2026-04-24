import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad
import scipy.sparse as sp


class OrthogonalSubGridScaleElement_V2:
    """
    Orthogonal sub-grid scales (OSGS) Timoshenko beam element with
    element-local constant subgrid variables:

        xi_w^e, xi_th^e

    Local full element ordering:
        [w1, w2, th1, th2, xi_w, xi_th]

    If schur_complement=True:
        returns condensed 4x4 physical element in ordering [w1, th1, w2, th2]

    If schur_complement=False:
        returns full 6x6 local element in ordering [w1, w2, th1, th2, xi_w, xi_th]
        (note: xi_w, xi_th are element-internal and should NOT be assembled globally
         as nodal dofs without a custom discontinuous-element assembly)
    """

    def __init__(self, schur_complement: bool = True):
        self.schur_complement = schur_complement

        # Exposed global dofs are always just [w, th]
        self.dof_per_node = 2
        self.full_dof_per_node = 2  # no nodal xi dofs anymore
        self.nodes_per_elem = 2
        self.clamped = True

        # stabilization constants
        self.c1 = 1e-4
        self.c3 = 12.0
        self.c2 = self.c4 = 1.0

        self._P0_cache = {}
        self._P1_cache = {}

        # stored blocks for debugging
        self.Kaa = None
        self.Kab = None
        self.Kbb = None

    def get_kelem(self, E: float, nu: float, thick: float, elem_length: float):
        pts, weights = second_order_quadrature()

        # Full local element: [w1, w2, th1, th2, xi_w, xi_th]
        kelem = np.zeros((6, 6))

        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick
        J = elem_length / 2.0

        # stabilization parameters
        tau_w = 1.0 / (self.c3 * ksGA / elem_length**2 + self.c4 * ksGA**2 / EI)
        tau_th = 1.0 / (self.c1 * EI / elem_length**2 + self.c2 * ksGA)

        # local indices
        W = [0, 1]
        TH = [2, 3]
        XIW = 4
        XITH = 5

        for xi, wt in zip(pts, weights):
            psi = np.array([lagrange(i, xi) for i in range(2)])           # N_i
            dpsi = np.array([lagrange_grad(i, xi, J) for i in range(2)])  # dN_i/dx
            meas = wt * J

            # -----------------------------------------------------------------
            # Standard stabilized physical part
            # -----------------------------------------------------------------
            c_shear = (ksGA - tau_th * ksGA**2) * meas
            c_bend = (EI - tau_w * ksGA**2) * meas

            # shear term: (w' + th)(dw' + dth)
            for i in range(2):
                for j in range(2):
                    kelem[W[i], W[j]]   += c_shear * dpsi[i] * dpsi[j]
                    kelem[W[i], TH[j]]  += c_shear * dpsi[i] * psi[j]
                    kelem[TH[i], W[j]]  += c_shear * psi[i] * dpsi[j]
                    kelem[TH[i], TH[j]] += c_shear * psi[i] * psi[j]

            # bending term: th' dth'
            for i in range(2):
                for j in range(2):
                    kelem[TH[i], TH[j]] += c_bend * dpsi[i] * dpsi[j]

            # -----------------------------------------------------------------
            # Coupling to element-local constant xi_w and xi_th
            #
            # Use P0 basis for xi's: phi_xi = 1 on the element
            # so there is one xi_w and one xi_th per element.
            # -----------------------------------------------------------------

            # (th', xi_w) and transpose-consistent partner
            # from your old code, but summed into one constant xi_w mode
            for j in range(2):
                kelem[XIW, TH[j]] += ksGA * meas * dpsi[j]
                kelem[TH[j], XIW] += tau_w * ksGA * meas * dpsi[j]

            # (w', xi_th) and -(th, xi_th), plus transpose partner
            for j in range(2):
                kelem[W[j], XITH]  += tau_th * ksGA * meas * dpsi[j]
                kelem[XITH, W[j]]  += ksGA * meas * dpsi[j]

                kelem[TH[j], XITH] -= tau_th * ksGA * meas * psi[j]
                kelem[XITH, TH[j]] -= ksGA * meas * psi[j]

            # xi_w-xi_w and xi_th-xi_th blocks with constant basis
            kelem[XIW, XIW]   -= meas
            kelem[XITH, XITH] -= meas

        if self.schur_complement:
            Kaa = kelem[:4, :4]
            Kab = kelem[:4, 4:]
            Kba = kelem[4:, :4]
            Kbb = kelem[4:, 4:]

            self.Kaa = Kaa.copy()
            self.Kab = Kab.copy()
            self.Kbb = Kbb.copy()

            # Safer than np.linalg.solve if nearly singular
            X = np.linalg.solve(Kbb, Kba)
            S = Kaa - Kab @ X

            # reorder [w1, w2, th1, th2] -> [w1, th1, w2, th2]
            new_order = np.array([0, 2, 1, 3], dtype=int)
            return S[new_order, :][:, new_order]

        return kelem

    def get_felem(self, mag, elem_length):
        """
        Element load vector for distributed transverse load on physical dofs only.

        Returns:
          - condensed/global ordering [w1, th1, w2, th2] if schur_complement=True
          - full physical ordering [w1, w2, th1, th2, xi_w, xi_th] otherwise
        """
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()

        if self.schur_complement:
            felem = np.zeros(4)
            for xi, wt in zip(pts, wts):
                psi = [lagrange(i, xi) for i in range(2)]
                # local physical ordering before reorder: [w1, w2, th1, th2]
                ftmp = np.zeros(4)
                for i in range(2):
                    ftmp[i] += mag * wt * J * psi[i]

                # reorder to [w1, th1, w2, th2]
                new_order = np.array([0, 2, 1, 3], dtype=int)
                felem += ftmp[new_order]
            return felem

        # full uncondensed local vector
        felem = np.zeros(6)
        for xi, wt in zip(pts, wts):
            psi = [lagrange(i, xi) for i in range(2)]
            for i in range(2):
                felem[i] += mag * wt * J * psi[i]
        return felem

    def _build_P1_scalar(self, nxe_coarse: int, apply_bcs: bool = False) -> sp.csr_matrix:
        """
        1D scalar nodal prolongation from nxe_coarse -> 2*nxe_coarse.
        Used only for the exposed nodal [w, th] variables.
        """
        key = (nxe_coarse, apply_bcs)
        if key in self._P1_cache:
            return self._P1_cache[key]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1

        rows, cols, vals = [], [], []

        # even fine nodes copy coarse
        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        # odd fine nodes average
        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()

        if apply_bcs:
            P1[[0, nf - 1], :] = 0.0
            P1[:, [0, nc - 1]] = 0.0
            P1.eliminate_zeros()

        self._P1_cache[key] = P1
        return P1

    def _build_P0_wth(self, nxe_coarse: int, ndof_per_node: int = 2) -> sp.csr_matrix:
        """
        Prolongation for exposed nodal dofs [w, th] only.
        """
        key = (nxe_coarse, ndof_per_node, self.clamped)
        if key in self._P0_cache:
            return self._P0_cache[key]

        P1 = self._build_P1_scalar(nxe_coarse, apply_bcs=False)
        P0 = sp.kron(P1, sp.eye(ndof_per_node, format="csr"), format="csr").tocsr()

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1

        def gdof(node: int, ldof: int) -> int:
            return ndof_per_node * node + ldof

        W, TH = 0, 1

        # always constrain w at ends
        wL_f, wR_f = gdof(0, W), gdof(nf - 1, W)
        wL_c, wR_c = gdof(0, W), gdof(nc - 1, W)
        P0[[wL_f, wR_f], :] = 0.0
        P0[:, [wL_c, wR_c]] = 0.0

        # clamped: also constrain theta at ends
        if self.clamped:
            thL_f, thR_f = gdof(0, TH), gdof(nf - 1, TH)
            thL_c, thR_c = gdof(0, TH), gdof(nc - 1, TH)
            P0[[thL_f, thR_f], :] = 0.0
            P0[:, [thL_c, thR_c]] = 0.0

        P0.eliminate_zeros()
        self._P0_cache[key] = P0
        return P0

    def prolongate(self, coarse_disp, length: float):
        """
        Prolongation on exposed nodal [w, th] dofs only.
        coarse_disp ordering: [w0, th0, w1, th1, ...]
        """
        ndof_coarse = coarse_disp.shape[0]
        dpn = self.dof_per_node
        nnodes_coarse = ndof_coarse // dpn
        nelems_coarse = nnodes_coarse - 1

        P = self._build_P0_wth(nelems_coarse, ndof_per_node=dpn)
        fine_disp = P @ coarse_disp
        return np.asarray(fine_disp).ravel()

    def restrict_defect(self, fine_defect, length: float):
        """
        Galerkin restriction on exposed nodal [w, th] dofs only.
        """
        ndof_fine = fine_defect.shape[0]
        dpn = self.dof_per_node
        nnodes_fine = ndof_fine // dpn
        nelems_fine = nnodes_fine - 1

        nelems_coarse = nelems_fine // 2
        P = self._build_P0_wth(nelems_coarse, ndof_per_node=dpn)
        coarse_defect = P.T @ fine_defect
        return np.asarray(coarse_defect).ravel()
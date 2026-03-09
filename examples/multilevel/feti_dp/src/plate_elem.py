import numpy as np
import scipy.sparse as sp

from .basis import second_order_quadrature
from .basis import get_lagrange_basis_2d_all
# from _utils import debug_print_bsr_matrix


import numpy as np
import scipy.sparse as sp

class MITCPlateElement:
    """
    Q4 Reissner–Mindlin with MITC4 shear (tying points):
      gamma_xz tied at (xi, eta) = (0, ±b)
      gamma_yz tied at (xi, eta) = (±a, 0)

    Then at each *regular* quadrature point (no reduced integration),
    gamma_xz(xi,eta) is interpolated from the two tying points in eta,
    gamma_yz(xi,eta) is interpolated from the two tying points in xi.

    Locking-aware prolong constraints are applied on the *tying strains*:
      gamma_xz(0,+b)=0, gamma_xz(0,-b)=0, gamma_yz(+a,0)=0, gamma_yz(-a,0)=0
    giving 4 constraints per element (fine and coarse).
    """

    def __init__(
        self,
        prolong_mode: str = "locking-global",  # 'standard'
        lam: float = 1e-2,
        a: float = 1.0,        # typical choice consistent w/ 2x2 Gauss
        b: float = 1.0,
        n_lock_sweeps:int=10,
        omega:float=0.5,
        debug:bool=False,
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard", "energy-jacobi"]

        self.dof_per_node = 3
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.debug = debug

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)
        self.n_lock_sweeps = n_lock_sweeps
        self.omega = float(omega)

        self.a = float(a)
        self.b = float(b)
        assert abs(self.a) > 0.0 and abs(self.b) > 0.0

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        self._lock_P_cache = {}
        self._kmat_cache = {}

    # ----------------------------
    # MITC helpers using (thy, -thx) directors + grad w ordering
    # ----------------------------
    @staticmethod
    def _interp_1d_pm(s: float, a: float) -> np.ndarray:
        """
        Linear 1D Lagrange basis to interpolate from points at [-a, +a] to s.
        Returns [N_- , N_+], where:
        N_- = (a - s)/(2a), N_+ = (a + s)/(2a)
        """
        a = float(a)
        return np.array([(a - s) / (2.0 * a), (a + s) / (2.0 * a)], dtype=float)

    @staticmethod
    def _geom_map_and_grads(Nxi, Neta, x, y, debug: bool = False):
        x_xi = float(np.dot(Nxi, x))
        x_eta = float(np.dot(Neta, x))
        y_xi = float(np.dot(Nxi, y))
        y_eta = float(np.dot(Neta, y))

        J = x_xi * y_eta - x_eta * y_xi
        invJ = 1.0 / J

        xi_x = y_eta * invJ
        xi_y = -x_eta * invJ
        eta_x = -y_xi * invJ
        eta_y = x_xi * invJ

        # if debug:
        #     print(f"{Nxi=}\n{Neta=}")

        Nx = Nxi * xi_x + Neta * eta_x
        Ny = Nxi * xi_y + Neta * eta_y
        return J, Nx, Ny

    def _Bs_rows_at_point(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Return the *pointwise* shear B rows (two 1x12 rows) for the director choice:
            director = (thy, -thx)

        i.e. transverse shear strains:
            gamma_xz = w_x + thy
            gamma_yz = w_y - thx

        evaluated at (xi,eta).

        DOF ordering per node: [w, thx, thy]
        """
        N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
        J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y, self.debug)

        bx = np.zeros((12,), dtype=float)
        by = np.zeros((12,), dtype=float)

        # if self.debug:
        #     print(f"{J=}\n{Nx=}\n{Ny=}\n{N=}")

        # gamma_xz = w_x + thy
        bx[0::3] = Nx
        bx[2::3] = +N

        # gamma_yz = w_y - thx
        by[0::3] = Ny
        by[1::3] = -N

        return J, bx, by

    def _Bs_mitc_at_quad(self, xi: float, eta: float, x: np.ndarray, y: np.ndarray):
        """
        Build the *effective* MITC shear B matrix (2x12) at the quadrature point (xi,eta):

        Uses the same director convention as _Bs_rows_at_point:
            gamma_xz = w_x + thy   tied from (0, ±b) and interpolated in eta
            gamma_yz = w_y - thx   tied from (±a, 0) and interpolated in xi
        """
        # Geometry/J evaluated at the quad point for integration measure
        Nq, Nxi_q, Neta_q = get_lagrange_basis_2d_all(xi, eta)
        Jq, _, _ = self._geom_map_and_grads(Nxi_q, Neta_q, x, y)

        # Tying evaluations
        _, bx_m, _ = self._Bs_rows_at_point(0.0, -self.b, x, y)
        _, bx_p, _ = self._Bs_rows_at_point(0.0, +self.b, x, y)

        _, _, by_m = self._Bs_rows_at_point(-self.a, 0.0, x, y)
        _, _, by_p = self._Bs_rows_at_point(+self.a, 0.0, x, y)

        # Interpolation weights
        w_eta = self._interp_1d_pm(eta, self.b)  # from [-b,+b] -> eta
        w_xi  = self._interp_1d_pm(xi,  self.a)  # from [-a,+a] -> xi

        row_gx = w_eta[0] * bx_m + w_eta[1] * bx_p
        row_gy = w_xi[0]  * by_m + w_xi[1]  * by_p

        Bs = np.vstack([row_gx, row_gy])  # 2x12
        return Jq, Bs

    # ----------------------------
    # Element stiffness (MITC4)
    # ----------------------------
    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        """
        elem_xpts length 12: [x0,y0,z0?, x1,y1,z1?, x2,y2,z2?, x3,y3,z3?]
        Uses x=elem_xpts[0::3], y=elem_xpts[1::3].

        Note: BENDING curvatures are formed consistently with director = (thy, -thx).
        """
        pts, wts = second_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))

        # material constants
        D0 = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = D0 * np.array([
            [1.0,  nu,          0.0],
            [nu,   1.0,         0.0],
            [0.0,  0.0, (1.0 - nu) / 2.0],
        ])
        ks = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        Ds = (ks * G * thick) * np.eye(2)

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        # ---- BENDING: consistent with director = (thy, -thx) ----
        # Let (rx, ry) = (thy, -thx). Curvatures:
        #   kappa_x  = d(rx)/dx  =  d(thy)/dx
        #   kappa_y  = d(ry)/dy  = -d(thx)/dy
        #   kappa_xy = d(rx)/dy + d(ry)/dx = d(thy)/dy - d(thx)/dx
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)
                J, Nx, Ny = self._geom_map_and_grads(Nxi, Neta, x, y)

                Bb = np.zeros((3, self.ndof))

                # kappa_x = d(thy)/dx
                Bb[0, 2::3] = Nx

                # kappa_y = -d(thx)/dy
                Bb[1, 1::3] = -Ny

                # kappa_xy = d(thy)/dy - d(thx)/dx
                Bb[2, 2::3] = Ny
                Bb[2, 1::3] = -Nx

                kelem += (Bb.T @ Db @ Bb) * (wt * J)

        # ---- SHEAR (MITC): full integration, tied/interpolated ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]
                Jq, Bs = self._Bs_mitc_at_quad(xi, eta, x, y)
                kelem += (Bs.T @ Ds @ Bs) * (wt * Jq)

        return kelem
    
    def get_felem(self, mag, elem_xpts:np.ndarray):
        """get element load vector"""

        pts, wts = second_order_quadrature()
        felem = np.zeros(self.ndof)
        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        for ipt in range(9):
            ii, jj = ipt % 3, ipt // 3
            xi = pts[ii]; eta = pts[jj]
            wt = wts[ii] * wts[jj]
            # basis (need N to map load; Nxi/Neta to get geometry jacobian)
            N, Nxi, Neta= get_lagrange_basis_2d_all(
                xi, eta, 
            )

            # geometry jacobian determinant
            x_xi  = np.dot(Nxi,  x);  x_eta = np.dot(Neta, x)
            y_xi  = np.dot(Nxi,  y);  y_eta = np.dot(Neta, y)
            J = x_xi * y_eta - x_eta * y_xi

            # physical point (x,y) at this quadrature point
            xq = float(np.dot(N, x))
            yq = float(np.dot(N, y))

            q = float(mag(xq, yq))   # distributed transverse load
            # q *= 60.0 # not sure where this correction is coming from tbh

            # consistent nodal load contribution: ∫ N^T q dA = Σ N_i q * wt * J
            fN = q * wt * J * N  # length 9

            # Apply load to the DOFs that contribute to transverse displacement.
            felem[0::3] += fN  # w

        return felem
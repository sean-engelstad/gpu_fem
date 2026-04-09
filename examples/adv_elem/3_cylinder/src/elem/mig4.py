import numpy as np
from math import comb
from numpy.polynomial import Polynomial as Poly

# ---------------------------------------------------------
# p=4 runtime restriction
# ---------------------------------------------------------
def get_iga4_BtoN(ielem, nxe):
    first_elem   = (ielem == 0)
    second_elem  = (ielem == 1)
    third_elem   = (ielem == 2)

    third_last   = (ielem == nxe - 3)
    second_last  = (ielem == nxe - 2)
    last_elem    = (ielem == nxe - 1)

    if first_elem:
        return np.array([
            [1.0,      0.0,      0.0,      0.0,      0.0],
            [0.0,      1.0,      0.5,      0.25,     0.125],
            [0.0,      0.0,      0.5,      7.0/12.0, 37.0/72.0],
            [0.0,      0.0,      0.0,      1.0/6.0,  23.0/72.0],
            [0.0,      0.0,      0.0,      0.0,      1.0/24.0],
        ])

    elif second_elem:
        return np.array([
            [1.0/8.0,   0.0,       0.0,       0.0,       0.0],
            [37.0/72.0, 4.0/9.0,   2.0/9.0,   1.0/9.0,   1.0/18.0],
            [23.0/72.0, 17.0/36.0, 11.0/18.0, 5.0/9.0,   4.0/9.0],
            [1.0/24.0,  1.0/12.0,  1.0/6.0,   1.0/3.0,   11.0/24.0],
            [0.0,       0.0,       0.0,       0.0,       1.0/24.0],
        ])

    elif third_elem:
        return np.array([
            [1.0/18.0,  0.0,       0.0,       0.0,       0.0],
            [4.0/9.0,   1.0/3.0,   1.0/6.0,   1.0/12.0,  1.0/24.0],
            [11.0/24.0, 7.0/12.0,  2.0/3.0,   7.0/12.0,  11.0/24.0],
            [1.0/24.0,  1.0/12.0,  1.0/6.0,   1.0/3.0,   11.0/24.0],
            [0.0,       0.0,       0.0,       0.0,       1.0/24.0],
        ])

    elif third_last:
        return np.array([
            [1.0/24.0,  0.0,       0.0,       0.0,       0.0],
            [11.0/24.0, 1.0/3.0,   1.0/6.0,   1.0/12.0,  1.0/24.0],
            [11.0/24.0, 7.0/12.0,  2.0/3.0,   7.0/12.0,  11.0/24.0],
            [1.0/24.0,  1.0/12.0,  1.0/6.0,   1.0/3.0,   4.0/9.0],
            [0.0,       0.0,       0.0,       0.0,       1.0/18.0],
        ])

    elif second_last:
        return np.array([
            [1.0/24.0,  0.0,       0.0,       0.0,       0.0],
            [11.0/24.0, 1.0/3.0,   1.0/6.0,   1.0/12.0,  1.0/24.0],
            [4.0/9.0,   5.0/9.0,   11.0/18.0, 17.0/36.0, 23.0/72.0],
            [1.0/18.0,  1.0/9.0,   2.0/9.0,   4.0/9.0,   37.0/72.0],
            [0.0,       0.0,       0.0,       0.0,       1.0/8.0],
        ])

    elif last_elem:
        return np.array([
            [1.0/24.0,  0.0,       0.0,       0.0,       0.0],
            [23.0/72.0, 1.0/6.0,   0.0,       0.0,       0.0],
            [37.0/72.0, 7.0/12.0,  0.5,       0.0,       0.0],
            [0.125,     0.25,      0.5,       1.0,       0.0],
            [0.0,       0.0,       0.0,       0.0,       1.0],
        ])

    else:
        return np.array([
            [1.0/24.0,  0.0,       0.0,       0.0,       0.0],
            [11.0/24.0, 1.0/3.0,   1.0/6.0,   1.0/12.0,  1.0/24.0],
            [11.0/24.0, 7.0/12.0,  2.0/3.0,   7.0/12.0,  11.0/24.0],
            [1.0/24.0,  1.0/12.0,  1.0/6.0,   1.0/3.0,   11.0/24.0],
            [0.0,       0.0,       0.0,       0.0,       1.0/24.0],
        ])


def bernstein_to_power_matrix_subst(p, a, b):
    M = np.zeros((p + 1, p + 1))
    x = Poly([0.0, 1.0])
    t = Poly([b, a])

    for i in range(p + 1):
        poly = comb(p, i) * (t**i) * ((1 - t)**(p - i))
        coef = poly.coef
        M[i, :len(coef)] = coef

    return M


def build_local_restriction_matrix_p4_runtime(ielem_c, nxe_c):
    p = 4

    BtoN_c  = get_iga4_BtoN(ielem_c, nxe_c)
    BtoN_Lf = get_iga4_BtoN(2 * ielem_c,     2 * nxe_c)
    BtoN_Rf = get_iga4_BtoN(2 * ielem_c + 1, 2 * nxe_c)

    Bc   = bernstein_to_power_matrix_subst(p, a=1.0, b=0.0)
    B_Lf = bernstein_to_power_matrix_subst(p, a=2.0, b=0.0)
    B_Rf = bernstein_to_power_matrix_subst(p, a=2.0, b=-1.0)

    N_H  = BtoN_c  @ Bc
    N_Lh = BtoN_Lf @ B_Lf
    N_Rh = BtoN_Rf @ B_Rf

    R_left_basis  = np.linalg.solve(N_Lh.T, N_H.T).T
    R_right_basis = np.linalg.solve(N_Rh.T, N_H.T).T

    R = np.zeros((5, 6))
    R[:, :5] += R_left_basis
    R[:, 1:] += R_right_basis

    R /= np.sum(R, axis=0, keepdims=True)
    return R


def build_R_p4_runtime(nxe_c):
    n_c = nxe_c + 4
    n_f = 2 * nxe_c + 4
    R = np.zeros((n_c, n_f))

    for ielem_c in range(nxe_c):
        R_loc = build_local_restriction_matrix_p4_runtime(ielem_c, nxe_c)
        R[ielem_c:ielem_c + 5, 2 * ielem_c:2 * ielem_c + 6] += R_loc

    R /= np.sum(R, axis=0, keepdims=True)
    return R

import numpy as np
import scipy.sparse as sp

from .basis import (
    # you need a quartic 1D basis in basis.py now
    get_iga2_basis,
    get_iga3_basis,
    get_iga4_basis,
    get_lagrange_basis_01,
    # choose the highest quadrature rule you have available
    fourth_order_quadrature,
    first_order_quadrature,
)

class MIG4CylinderElement:
    """
    MIG4 cylinder element with spaces

      u   : (p2,p4)
      v   : (p3,p3)
      w   : (p3,p2)
      thx : (p2,p2)
      thy : (p3,p1)

    Local ordering:
      Ue = [ w(12), u(15), v(16), thx(9), thy(8) ]
    """

    def __init__(
        self,
        r: float,
        clamped: bool = False,
        curvature_on: bool = True,
        reduced_integrate_exy: bool = False,
        rax: float = None,
        prolong_mode: str = "standard",
        omega: float = 0.7,
        n_Psweeps: int = 2,
    ):
        self.r = float(r)
        self.rax = float(rax) if rax is not None else None
        self.clamped = bool(clamped)
        self.curvature_on = bool(curvature_on)
        self.reduced_integrate_exy = bool(reduced_integrate_exy)

        self.prolong_mode = prolong_mode
        self.omega = omega
        self.n_Psweeps = n_Psweeps

        self._P_cache = {}
        self._kmat_cache = {}
        self.dof_per_node = 5

    # -------------------------------------------------
    # tensor helper
    # -------------------------------------------------
    @staticmethod
    def _tensor_product_basis(bx, by):
        Nx, dNx = bx
        Ny, dNy = by
        N    = np.kron(Ny, Nx)
        Nxi  = np.kron(Ny, dNx)
        Neta = np.kron(dNy, Nx)
        return N, Nxi, Neta

    # -------------------------------------------------
    # 1D basis helpers
    # -------------------------------------------------
    @staticmethod
    def _iga2_1d(x, ie, ne):
        left  = (ie == 0)
        right = (ie == ne - 1)
        return get_iga2_basis(x, left, right)

    @staticmethod
    def _iga3_1d(x, ie, ne):
        return get_iga3_basis(x, ie, ne)

    @staticmethod
    def _iga4_1d(x, ie, ne):
        return get_iga4_basis(x, ie, ne)

    @staticmethod
    def _p1_1d(x):
        return get_lagrange_basis_01(x)

    # -------------------------------------------------
    # 1D restriction / prolongation
    # -------------------------------------------------
    def _build_R_p4(self, nxe_c: int) -> np.ndarray:
        return build_R_p4_runtime(nxe_c)

    def _build_R_p3(self, nxe_c: int) -> np.ndarray:
        # reuse your existing p3 builder from the old class
        n_c = nxe_c + 3
        n_f = 2 * nxe_c + 3
        R = np.zeros((n_c, n_f))
        for ielem in range(nxe_c):
            if ielem == 0:
                R_loc = np.array([
                    [1.0   , 0.5   , 0.0   , 0.0   , 0.0],
                    [0.0   , 0.5   , 0.75  , 0.1875, 0.0],
                    [0.0   , 0.0   , 0.25  , 0.6875, 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])
            elif ielem == 1:
                R_loc = np.array([
                    [0.75  , 0.1875, 0.0   , 0.0   , 0.0],
                    [0.25  , 0.6875, 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.75  , 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])
            elif ielem == nxe_c - 2:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.6875, 0.25],
                    [0.0   , 0.0   , 0.0   , 0.1875, 0.75]
                ])
            elif ielem == nxe_c - 1:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.6875, 0.25  , 0.0   , 0.0],
                    [0.0   , 0.1875, 0.75  , 0.5   , 0.0],
                    [0.0   , 0.0   , 0.0   , 0.5   , 1.0]
                ])
            else:
                R_loc = np.array([
                    [0.5   , 0.125 , 0.0   , 0.0   , 0.0],
                    [0.5   , 0.75  , 0.5   , 0.125 , 0.0],
                    [0.0   , 0.125 , 0.5   , 0.75  , 0.5],
                    [0.0   , 0.0   , 0.0   , 0.125 , 0.5]
                ])
            R[ielem:ielem+4, 2*ielem:2*ielem+5] += R_loc
        R /= np.sum(R, axis=0)
        return R

    def _build_R_p2(self, nxe_c: int) -> np.ndarray:
        # reuse your existing p2 builder from the old class
        n_c = nxe_c + 2
        n_f = 2 * nxe_c + 2
        R = np.zeros((n_c, n_f))
        counts = 1e-14 * np.ones((n_c, n_f))

        for ielem_c in range(nxe_c):
            lf = 2 * ielem_c
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

            l_nz = l_mat / (l_mat + 1e-14)
            R[ielem_c:ielem_c+3, lf:lf+3] += l_mat
            counts[ielem_c:ielem_c+3, lf:lf+3] += l_nz

            rf = 2 * ielem_c + 1
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

            r_nz = r_mat / (r_mat + 1e-14)
            R[ielem_c:ielem_c+3, rf:rf+3] += r_mat
            counts[ielem_c:ielem_c+3, rf:rf+3] += r_nz

        R /= counts
        return R

    def _build_P_p1(self, nxe_c: int) -> np.ndarray:
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        P = np.zeros((n_f, n_c))
        for i in range(nxe_c):
            P[2*i,   i]   = 1.0
            P[2*i+1, i]   = 0.5
            P[2*i+1, i+1] = 0.5
        P[2*nxe_c, nxe_c] = 1.0
        return P

    # -------------------------------------------------
    # stiffness
    # -------------------------------------------------
    def get_kelem(self, E, nu, thick, dx, dy, ixe, nxe, iye, nye):
        pts, wts   = fourth_order_quadrature()
        pts_r, wts_r = first_order_quadrature()

        Jdet  = dx * dy
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        r = self.r
        invRx = (1.0 / self.rax) if (self.curvature_on and self.rax is not None) else 0.0
        invRy = (1.0 / r)        if (self.curvature_on and r != 0.0) else 0.0

        n_w   = 12   # (p3,p2)
        n_u   = 15   # (p2,p4)
        n_v   = 16   # (p3,p3)
        n_thx = 9    # (p2,p2)
        n_thy = 8    # (p3,p1)

        def Z(a, b): return np.zeros((a, b))

        Kww   = Z(n_w, n_w);   Kwu   = Z(n_w, n_u);   Kwv   = Z(n_w, n_v);   Kwtx  = Z(n_w, n_thx); Kwty  = Z(n_w, n_thy)
        Kuw   = Z(n_u, n_w);   Kuu   = Z(n_u, n_u);   Kuv   = Z(n_u, n_v);   Kutx  = Z(n_u, n_thx); Kuty  = Z(n_u, n_thy)
        Kvw   = Z(n_v, n_w);   Kvu   = Z(n_v, n_u);   Kvv   = Z(n_v, n_v);   Kvtx  = Z(n_v, n_thx); Kvty  = Z(n_v, n_thy)
        Ktxw  = Z(n_thx, n_w); Ktxu  = Z(n_thx, n_u); Ktxv  = Z(n_thx, n_v); Ktxtx = Z(n_thx, n_thx); Ktxty = Z(n_thx, n_thy)
        Ktyw  = Z(n_thy, n_w); Ktyu  = Z(n_thy, n_u); Ktyv  = Z(n_thy, n_v); Ktytx = Z(n_thy, n_thx); Ktyty = Z(n_thy, n_thy)

        EI = E * thick**3 / (12.0 * (1.0 - nu**2))
        Db = EI * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        A0 = E * thick / (1.0 - nu**2)
        Dm = A0 * np.array([
            [1.0, nu, 0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        Ds = ks * G * thick * np.eye(2)

        for _eta, w_eta_raw in zip(pts, wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw

            # y-bases
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)
            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)
            Ny4, dNy4 = self._iga4_1d(eta, iye, nye)
            Ny1, dNy1 = self._p1_1d(eta)

            for _xi, w_xi_raw in zip(pts, wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = (w_xi * w_eta) * Jdet

                # x-bases
                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)
                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)
                Nx4, dNx4 = self._iga4_1d(xi, ixe, nxe)
                Nx1, dNx1 = self._p1_1d(xi)

                # spaces
                Nw,  Nw_xi,  Nw_eta  = self._tensor_product_basis((Nx3, dNx3), (Ny2, dNy2))  # (p3,p2)
                Nu,  Nu_xi,  Nu_eta  = self._tensor_product_basis((Nx2, dNx2), (Ny4, dNy4))  # (p2,p4)
                Nv,  Nv_xi,  Nv_eta  = self._tensor_product_basis((Nx3, dNx3), (Ny3, dNy3))  # (p3,p3)
                Ntx, Ntx_xi, Ntx_eta = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))  # (p2,p2)
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis((Nx3, dNx3), (Ny1, dNy1))  # (p3,p1)

                Nw_x  = Nw_xi  * xi_x
                Nw_y  = Nw_eta * eta_y
                Nu_x  = Nu_xi  * xi_x
                Nu_y  = Nu_eta * eta_y
                Nv_x  = Nv_xi  * xi_x
                Nv_y  = Nv_eta * eta_y
                Ntx_x = Ntx_xi * xi_x
                Ntx_y = Ntx_eta * eta_y
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                # bending
                D11, D12, D22, D33 = Db[0,0], Db[0,1], Db[1,1], Db[2,2]
                Ktxtx += wt * (D11 * np.outer(Ntx_x, Ntx_x))
                Ktyty += wt * (D22 * np.outer(Nty_y, Nty_y))
                Ktxty += wt * (D12 * np.outer(Ntx_x, Nty_y))
                Ktytx += wt * (D12 * np.outer(Nty_y, Ntx_x))

                Ktxtx += wt * (D33 * np.outer(Ntx_y, Ntx_y))
                Ktyty += wt * (D33 * np.outer(Nty_x, Nty_x))
                Ktxty += wt * (D33 * np.outer(Ntx_y, Nty_x))
                Ktytx += wt * (D33 * np.outer(Nty_x, Ntx_y))

                # shear
                Ds11, Ds22 = Ds[0,0], Ds[1,1]
                Kww   += wt * (Ds11 * np.outer(Nw_x, Nw_x))
                Kwtx  += wt * (Ds11 * np.outer(Nw_x, Ntx))
                Ktxw  += wt * (Ds11 * np.outer(Ntx, Nw_x))
                Ktxtx += wt * (Ds11 * np.outer(Ntx, Ntx))

                Kww   += wt * (Ds22 * np.outer(Nw_y, Nw_y))
                Kwty  += wt * (Ds22 * np.outer(Nw_y, Nty))
                Ktyw  += wt * (Ds22 * np.outer(Nty, Nw_y))
                Ktyty += wt * (Ds22 * np.outer(Nty, Nty))

                # membrane exx, eyy, coupling
                exx_u = Nu_x
                exx_w = invRx * Nw
                eyy_v = Nv_y
                eyy_w = invRy * Nw

                D11m, D12m, D22m = Dm[0,0], Dm[0,1], Dm[1,1]

                Kuu += wt * (D11m * np.outer(exx_u, exx_u))
                Kuw += wt * (D11m * np.outer(exx_u, exx_w))
                Kwu += wt * (D11m * np.outer(exx_w, exx_u))
                Kww += wt * (D11m * np.outer(exx_w, exx_w))

                Kvv += wt * (D22m * np.outer(eyy_v, eyy_v))
                Kvw += wt * (D22m * np.outer(eyy_v, eyy_w))
                Kwv += wt * (D22m * np.outer(eyy_w, eyy_v))
                Kww += wt * (D22m * np.outer(eyy_w, eyy_w))

                Kuv += wt * (D12m * np.outer(exx_u, eyy_v))
                Kvu += wt * (D12m * np.outer(eyy_v, exx_u))
                Kuw += wt * (D12m * np.outer(exx_u, eyy_w))
                Kwu += wt * (D12m * np.outer(eyy_w, exx_u))
                Kwv += wt * (D12m * np.outer(exx_w, eyy_v))
                Kvw += wt * (D12m * np.outer(eyy_v, exx_w))

        # exy = u_y + v_x
        quad_pts, quad_wts = (pts_r, wts_r) if self.reduced_integrate_exy else (pts, wts)
        for _eta, w_eta_raw in zip(quad_pts, quad_wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw

            Ny3, dNy3 = self._iga3_1d(eta, iye, nye)
            Ny4, dNy4 = self._iga4_1d(eta, iye, nye)

            for _xi, w_xi_raw in zip(quad_pts, quad_wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = (w_xi * w_eta) * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)
                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)

                Nu, Nu_xi, Nu_eta = self._tensor_product_basis((Nx2, dNx2), (Ny4, dNy4))
                Nv, Nv_xi, Nv_eta = self._tensor_product_basis((Nx3, dNx3), (Ny3, dNy3))

                exy_u = Nu_eta * eta_y
                exy_v = Nv_xi  * xi_x
                D33m = Dm[2,2]

                Kuu += wt * (D33m * np.outer(exy_u, exy_u))
                Kvv += wt * (D33m * np.outer(exy_v, exy_v))
                Kuv += wt * (D33m * np.outer(exy_u, exy_v))
                Kvu += wt * (D33m * np.outer(exy_v, exy_u))

        return (
            Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
            Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
            Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
            Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
            Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
        )

    # -------------------------------------------------
    # load vector
    # -------------------------------------------------
    def get_felem(self, load_fcn, x0, y0, dx, dy, ixe, nxe, iye, nye):
        pts, wts = fourth_order_quadrature()

        fw  = np.zeros(12)  # w:(p3,p2)
        fu  = np.zeros(15)
        fv  = np.zeros(16)
        ftx = np.zeros(9)
        fty = np.zeros(8)

        Jdet = dx * dy

        for _eta, w_eta_raw in zip(pts, wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw
            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)

            for _xi, w_xi_raw in zip(pts, wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = (w_xi * w_eta) * Jdet

                Nx3, dNx3 = self._iga3_1d(xi, ixe, nxe)
                Nw, _, _ = self._tensor_product_basis((Nx3, dNx3), (Ny2, dNy2))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += q * Nw * wt

        return fw, fu, fv, ftx, fty

    # -------------------------------------------------
    # prolongation
    # -------------------------------------------------
    def build_prolongation_operator(self, nxe_c: int, nye_c: int) -> sp.csr_matrix:
        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 3, nye_c + 2   # w:(p3,p2)
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 4   # u:(p2,p4)
        nxv_c,  nyv_c  = nxe_c + 3, nye_c + 3   # v:(p3,p3)
        nxtx_c, nytx_c = nxe_c + 2, nye_c + 2   # thx:(p2,p2)
        nxty_c, nyty_c = nxe_c + 3, nye_c + 1   # thy:(p3,p1)

        nw_c  = nxw_c * nyw_c
        nu_c  = nxu_c * nyu_c
        nv_c  = nxv_c * nyv_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c

        # 1D prolongations
        Px4 = sp.csr_matrix(self._build_R_p4(nxe_c).T)
        Py4 = sp.csr_matrix(self._build_R_p4(nye_c).T)
        Px3 = sp.csr_matrix(self._build_R_p3(nxe_c).T)
        Py3 = sp.csr_matrix(self._build_R_p3(nye_c).T)
        Px2 = sp.csr_matrix(self._build_R_p2(nxe_c).T)
        Py2 = sp.csr_matrix(self._build_R_p2(nye_c).T)
        Px1 = sp.csr_matrix(self._build_P_p1(nxe_c))
        Py1 = sp.csr_matrix(self._build_P_p1(nye_c))

        # 2D prolongations
        Pw  = sp.kron(Py2, Px3, format="csr")   # w:(p3,p2)
        Pu  = sp.kron(Py4, Px2, format="csr")   # u:(p2,p4)
        Pv  = sp.kron(Py3, Px3, format="csr")   # v:(p3,p3)
        Ptx = sp.kron(Py2, Px2, format="csr")   # thx:(p2,p2)
        Pty = sp.kron(Py1, Px3, format="csr")   # thy:(p3,p1)

        P = sp.block_diag((Pw, Pu, Pv, Ptx, Pty), format="csr")
        return P

    # reuse your old apply_bcs_2d, get_bc_dofs_for_layout,
    # apply_bcs_to_prolongation, build_bc_prolongation_operator,
    # _energy_smooth_jacobi_v1, prolongate, restrict_defect

    def apply_bcs_2d(
        self, u: np.ndarray,
        nxw: int, nyw: int,
        nxu: int, nyu: int,
        nxv: int, nyv: int,
        nxtx: int, nytx: int,
        nxty: int, nyty: int,
    ):
        """
        Apply BCs to concatenated global vector layout [w, u, v, thx, thy].

        MIG4 sizes:
        w   : (p3,p2)
        u   : (p2,p4)
        v   : (p3,p3)
        thx : (p2,p2)
        thy : (p3,p1)
        """
        u = np.asarray(u)

        nw   = nxw  * nyw
        nu   = nxu  * nyu
        nv   = nxv  * nyv
        ntx  = nxtx * nytx
        nty  = nxty * nyty

        assert u.size == (nw + nu + nv + ntx + nty)

        off = 0
        w  = u[off:off+nw];   off += nw
        U  = u[off:off+nu];   off += nu
        V  = u[off:off+nv];   off += nv
        tx = u[off:off+ntx];  off += ntx
        ty = u[off:off+nty]

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)

        # w boundary always
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    w[i + nxw * j] = 0.0

        if self.clamped:
            for j in range(nyu):
                for i in range(nxu):
                    if on_bndry(i, j, nxu, nyu):
                        U[i + nxu * j] = 0.0

            for j in range(nyv):
                for i in range(nxv):
                    if on_bndry(i, j, nxv, nyv):
                        V[i + nxv * j] = 0.0

            for j in range(nytx):
                for i in range(nxtx):
                    if on_bndry(i, j, nxtx, nytx):
                        tx[i + nxtx * j] = 0.0

            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        ty[i + nxty * j] = 0.0
        else:
            # SS-ish choice, same logic as old class:
            # u = 0 on x = 0 edge of u-grid
            for j in range(nyu):
                U[0 + nxu * j] = 0.0

            # v = 0 on y = 0 edge of v-grid
            for i in range(nxv):
                V[i + nxv * 0] = 0.0

        return np.concatenate([w, U, V, tx, ty])


    def get_bc_dofs_for_layout(
        self,
        nxw: int, nyw: int,
        nxu: int, nyu: int,
        nxv: int, nyv: int,
        nxtx: int, nytx: int,
        nxty: int, nyty: int,
    ) -> np.ndarray:
        """
        Return constrained global dof indices in concatenated layout [w, u, v, thx, thy].

        Matches apply_bcs_2d exactly.
        """
        nw   = nxw  * nyw
        nu   = nxu  * nyu
        nv   = nxv  * nyv
        ntx  = nxtx * nytx
        nty  = nxty * nyty

        off_w  = 0
        off_u  = off_w  + nw
        off_v  = off_u  + nu
        off_tx = off_v  + nv
        off_ty = off_tx + ntx

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)

        bc = []

        # w boundary always
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    bc.append(off_w + (i + nxw * j))

        if self.clamped:
            for j in range(nyu):
                for i in range(nxu):
                    if on_bndry(i, j, nxu, nyu):
                        bc.append(off_u + (i + nxu * j))

            for j in range(nyv):
                for i in range(nxv):
                    if on_bndry(i, j, nxv, nyv):
                        bc.append(off_v + (i + nxv * j))

            for j in range(nytx):
                for i in range(nxtx):
                    if on_bndry(i, j, nxtx, nytx):
                        bc.append(off_tx + (i + nxtx * j))

            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        bc.append(off_ty + (i + nxty * j))
        else:
            # u = 0 on x = 0 edge
            for j in range(nyu):
                bc.append(off_u + (0 + nxu * j))

            # v = 0 on y = 0 edge
            for i in range(nxv):
                bc.append(off_v + (i + nxv * 0))

        bc = np.array(bc, dtype=int)
        if bc.size:
            bc = np.unique(bc)
        return bc


    def apply_bcs_to_prolongation(
        self,
        P: sp.csr_matrix,
        fine_bc_dofs: np.ndarray,
        coarse_bc_dofs: np.ndarray = None,
        inject_identity_on_fine: bool = False,
    ) -> sp.csr_matrix:
        """
        Zero constrained fine rows and optionally constrained coarse cols in prolongation.
        """
        P = P.tolil()

        fine_bc_dofs = np.asarray(fine_bc_dofs, dtype=int)
        fine_bc_dofs = fine_bc_dofs[(fine_bc_dofs >= 0) & (fine_bc_dofs < P.shape[0])]

        for i in fine_bc_dofs:
            P.rows[i] = []
            P.data[i] = []

        if coarse_bc_dofs is not None:
            coarse_bc_dofs = np.asarray(coarse_bc_dofs, dtype=int)
            coarse_bc_dofs = coarse_bc_dofs[(coarse_bc_dofs >= 0) & (coarse_bc_dofs < P.shape[1])]
            Pc = P.tocsc(copy=True)
            Pc[:, coarse_bc_dofs] = 0.0
            P = Pc.tolil()

        if inject_identity_on_fine and P.shape[0] == P.shape[1]:
            for i in fine_bc_dofs:
                P[i, i] = 1.0

        return P.tocsr()


    def build_prolongation_operator(self, nxe_c: int, nye_c: int) -> sp.csr_matrix:
        """
        Build full prolongation operator P : u_f = P * u_c for MIG4 layout [w, u, v, thx, thy].

        MIG4 spaces:
        w   : (p3,p2)
        u   : (p2,p4)
        v   : (p3,p3)
        thx : (p2,p2)
        thy : (p3,p1)
        """
        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 3, nye_c + 2   # w:(p3,p2)
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 4   # u:(p2,p4)
        nxv_c,  nyv_c  = nxe_c + 3, nye_c + 3   # v:(p3,p3)
        nxtx_c, nytx_c = nxe_c + 2, nye_c + 2   # thx:(p2,p2)
        nxty_c, nyty_c = nxe_c + 3, nye_c + 1   # thy:(p3,p1)

        nw_c  = nxw_c  * nyw_c
        nu_c  = nxu_c  * nyu_c
        nv_c  = nxv_c  * nyv_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c
        N_c = nw_c + nu_c + nv_c + ntx_c + nty_c

        # fine sizes
        nxe_f, nye_f = 2 * nxe_c, 2 * nye_c
        nxw_f,  nyw_f  = nxe_f + 3, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 4
        nxv_f,  nyv_f  = nxe_f + 3, nye_f + 3
        nxtx_f, nytx_f = nxe_f + 2, nye_f + 2
        nxty_f, nyty_f = nxe_f + 3, nye_f + 1

        nw_f  = nxw_f  * nyw_f
        nu_f  = nxu_f  * nyu_f
        nv_f  = nxv_f  * nyv_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f
        N_f = nw_f + nu_f + nv_f + ntx_f + nty_f

        # 1D prolongations
        Px4 = sp.csr_matrix(self._build_R_p4(nxe_c).T)
        Py4 = sp.csr_matrix(self._build_R_p4(nye_c).T)

        Px3 = sp.csr_matrix(self._build_R_p3(nxe_c).T)
        Py3 = sp.csr_matrix(self._build_R_p3(nye_c).T)

        Px2 = sp.csr_matrix(self._build_R_p2(nxe_c).T)
        Py2 = sp.csr_matrix(self._build_R_p2(nye_c).T)

        Px1 = sp.csr_matrix(self._build_P_p1(nxe_c))
        Py1 = sp.csr_matrix(self._build_P_p1(nye_c))

        # 2D prolongations
        Pw  = sp.kron(Py2, Px3, format="csr")   # w:(p3,p2)
        Pu  = sp.kron(Py4, Px2, format="csr")   # u:(p2,p4)
        Pv  = sp.kron(Py3, Px3, format="csr")   # v:(p3,p3)
        Ptx = sp.kron(Py2, Px2, format="csr")   # thx:(p2,p2)
        Pty = sp.kron(Py1, Px3, format="csr")   # thy:(p3,p1)

        P = sp.block_diag((Pw, Pu, Pv, Ptx, Pty), format="csr")
        assert P.shape == (N_f, N_c)
        return P


    def build_bc_prolongation_operator(self, nxe_c: int, nye_c: int):
        """
        Build prolongation and enforce BC compatibility on fine/coarse spaces.
        """
        P = self.build_prolongation_operator(nxe_c, nye_c)

        # fine sizes
        nxe_f, nye_f = 2 * nxe_c, 2 * nye_c
        nxw_f,  nyw_f  = nxe_f + 3, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 4
        nxv_f,  nyv_f  = nxe_f + 3, nye_f + 3
        nxtx_f, nytx_f = nxe_f + 2, nye_f + 2
        nxty_f, nyty_f = nxe_f + 3, nye_f + 1

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 3, nye_c + 2
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 4
        nxv_c,  nyv_c  = nxe_c + 3, nye_c + 3
        nxtx_c, nytx_c = nxe_c + 2, nye_c + 2
        nxty_c, nyty_c = nxe_c + 3, nye_c + 1

        fine_bc_dofs = self.get_bc_dofs_for_layout(
            nxw_f, nyw_f, nxu_f, nyu_f, nxv_f, nyv_f, nxtx_f, nytx_f, nxty_f, nyty_f
        )
        coarse_bc_dofs = self.get_bc_dofs_for_layout(
            nxw_c, nyw_c, nxu_c, nyu_c, nxv_c, nyv_c, nxtx_c, nytx_c, nxty_c, nyty_c
        )

        P_bc = self.apply_bcs_to_prolongation(
            P,
            fine_bc_dofs=fine_bc_dofs,
            coarse_bc_dofs=coarse_bc_dofs,
            inject_identity_on_fine=False,
        )
        return P_bc


    def _assemble_prolongation(self, nxe_f):
        nxe_coarse = nxe_f // 2
        if nxe_f in self._P_cache:
            return self._P_cache[nxe_f]

        if self.prolong_mode == "standard":
            P = self.build_bc_prolongation_operator(nxe_coarse, nxe_coarse)
        elif self.prolong_mode == "energy-jacobi":
            raise NotImplementedError("energy-jacobi not included here")
        else:
            raise ValueError(f"Unknown prolong_mode: {self.prolong_mode}")

        self._P_cache[nxe_f] = sp.csr_matrix(P)
        return self._P_cache[nxe_f]


    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Dyadic prolongation for MIG4 layout [w, u, v, thx, thy].
        """
        u_c = np.asarray(u_c)
        nxe_f = 2 * nxe_c
        P = self._assemble_prolongation(nxe_f)
        u_f = P.dot(u_c)
        return u_f


    def restrict_defect(self, r_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Restrict fine defect -> coarse defect for MIG4 layout [w, u, v, thx, thy].
        Uses transpose of BC-consistent prolongation.
        """
        r_f = np.asarray(r_f)
        nxe_f = 2 * nxe_c
        P = self._assemble_prolongation(nxe_f)
        r_c = P.T.dot(r_f)
        return r_c
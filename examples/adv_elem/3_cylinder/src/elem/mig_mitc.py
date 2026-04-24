import numpy as np
import scipy.sparse as sp

from .basis import (
    third_order_quadrature,
    get_iga2_basis,
    get_lagrange_basis_01,
)


class MIG_MITC_CylinderElement:
    """
    Mixed-IGA + MITC membrane cylinder shell element

    Local unknown ordering:
        Ue = [ w(9), u(9), v(9), thx(6), thy(6) ]

    Spaces:
        w   : (p2,p2)
        u   : (p2,p2)
        v   : (p2,p2)
        thx : (p1,p2)
        thy : (p2,p1)

    Kinematics:
        bending:
            kxx = thx_x
            kyy = thy_y
            kxy = thy_x + thx_y

        transverse shear (mixed IGA):
            gamx = w_x + thx
            gamy = w_y + thy

        membrane raw strains:
            exx = u_x + w/Rx
            eyy = v_y + w/Ry
            exy = u_y + v_x

    Membrane locking treatment:
        MITC-style assumed membrane strains:
          - exx tied at two xi-line points and interpolated in xi
          - eyy tied at two eta-line points and interpolated in eta
          - exy tied at four corner-like points and bilinearly interpolated

    Notes:
      1) This is a cylinder-specialized MITC-style membrane projection.
      2) It keeps the same algebraic unknowns; no extra strain DOFs.
      3) This is the most reasonable hybridization of your uploaded DRIG element
         with the MITC tying/interpolation pattern from your plate code.
    """

    def __init__(
        self,
        r: float,
        clamped: bool = False,
        curvature_on: bool = True,
        rax: float = None,
        prolong_mode: str = "standard",
        omega: float = 0.7,
        n_Psweeps: int = 2,
        tie_loc: float = None,
    ):
        self.r = float(r)
        self.rax = float(rax) if rax is not None else None
        self.clamped = bool(clamped)
        self.curvature_on = bool(curvature_on)

        self._P_cache = {}
        self._kmat_cache = {}
        self.dof_per_node = 5

        self.prolong_mode = prolong_mode
        self.omega = omega
        self.n_Psweeps = n_Psweeps

        # Parent cell is [0,1]^2 in this IGA code.
        # Use mapped 2-point Gauss locations by default:
        #   s_- = 0.5*(1 - 1/sqrt(3)), s_+ = 0.5*(1 + 1/sqrt(3))
        if tie_loc is None:
            g = 1.0 / np.sqrt(3.0)
            self.s_minus = 0.5 * (1.0 - g)
            self.s_plus  = 0.5 * (1.0 + g)
        else:
            a = float(tie_loc)
            self.s_minus = a
            self.s_plus = 1.0 - a

    # ------------------------------------------------------------------
    # basis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tensor_product_basis(bx, by):
        Nx, dNx = bx
        Ny, dNy = by
        N    = np.kron(Ny, Nx)
        Nxi  = np.kron(Ny, dNx)
        Neta = np.kron(dNy, Nx)
        return N, Nxi, Neta

    @staticmethod
    def _iga2_1d(x, ixe, nxe):
        left = (ixe == 0)
        right = (ixe == nxe - 1)
        return get_iga2_basis(x, left, right)

    @staticmethod
    def _p1_1d(x):
        return get_lagrange_basis_01(x)

    @staticmethod
    def _interp_1d_ab(s, sm, sp):
        """
        Linear interpolation from two points s=sm and s=sp.
        Returns [N_m, N_p].
        """
        return np.array([
            (sp - s) / (sp - sm),
            (s - sm) / (sp - sm),
        ], dtype=float)

    @classmethod
    def _interp_bilinear_4(cls, xi, eta, sm, sp):
        """
        Bilinear interpolation from 4 tying points:
            (sm,sm), (sp,sm), (sm,sp), (sp,sp)

        Returns weights [w00, w10, w01, w11].
        """
        Lx = cls._interp_1d_ab(xi, sm, sp)
        Ly = cls._interp_1d_ab(eta, sm, sp)
        return np.array([
            Lx[0] * Ly[0],  # (sm,sm)
            Lx[1] * Ly[0],  # (sp,sm)
            Lx[0] * Ly[1],  # (sm,sp)
            Lx[1] * Ly[1],  # (sp,sp)
        ], dtype=float)

    # ------------------------------------------------------------------
    # membrane MITC helpers
    # ------------------------------------------------------------------
    def _raw_membrane_rows_at_point(self, xi, eta, dx, dy, ixe, nxe, iye, nye):
        """
        Returns raw membrane B rows (each is length ndof_local = 39):
            exx = u_x + w/Rx
            eyy = v_y + w/Ry
            exy = u_y + v_x
        in local ordering [w(9), u(9), v(9), thx(6), thy(6)].
        """
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        invRx = (1.0 / self.rax) if (self.curvature_on and self.rax is not None) else 0.0
        invRy = (1.0 / self.r)   if (self.curvature_on and self.r != 0.0) else 0.0

        Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)
        Ny2, dNy2 = self._iga2_1d(eta, iye, nye)

        Nw, Nw_xi, Nw_eta = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))
        Nu, Nu_xi, Nu_eta = Nw, Nw_xi, Nw_eta
        Nv, Nv_xi, Nv_eta = Nw, Nw_xi, Nw_eta

        Nw_x = Nw_xi * xi_x
        Nw_y = Nw_eta * eta_y
        Nu_x = Nu_xi * xi_x
        Nu_y = Nu_eta * eta_y
        Nv_x = Nv_xi * xi_x
        Nv_y = Nv_eta * eta_y

        ndof = 9 + 9 + 9 + 6 + 6

        bxx = np.zeros((ndof,), dtype=float)
        byy = np.zeros((ndof,), dtype=float)
        bxy = np.zeros((ndof,), dtype=float)

        off_w   = 0
        off_u   = 9
        off_v   = 18
        off_thx = 27
        off_thy = 33

        # exx = u_x + w/Rx
        bxx[off_w:off_w+9] = invRx * Nw
        bxx[off_u:off_u+9] = Nu_x

        # eyy = v_y + w/Ry
        byy[off_w:off_w+9] = invRy * Nw
        byy[off_v:off_v+9] = Nv_y

        # exy = u_y + v_x
        bxy[off_u:off_u+9] = Nu_y
        bxy[off_v:off_v+9] = Nv_x

        return bxx, byy, bxy

    def _Bm_mitc_at_quad(self, xi, eta, dx, dy, ixe, nxe, iye, nye):
        """
        Build MITC-style assumed membrane B matrix at quadrature point:
            [exx_tilde, eyy_tilde, exy_tilde]
        """
        sm = self.s_minus
        sp = self.s_plus

        # exx: tie on xi-line at eta = 0.5, interpolate in xi
        bxx_m, _, _ = self._raw_membrane_rows_at_point(sm, 0.5, dx, dy, ixe, nxe, iye, nye)
        bxx_p, _, _ = self._raw_membrane_rows_at_point(sp, 0.5, dx, dy, ixe, nxe, iye, nye)
        wx = self._interp_1d_ab(xi, sm, sp)
        row_exx = wx[0] * bxx_m + wx[1] * bxx_p

        # eyy: tie on eta-line at xi = 0.5, interpolate in eta
        _, byy_m, _ = self._raw_membrane_rows_at_point(0.5, sm, dx, dy, ixe, nxe, iye, nye)
        _, byy_p, _ = self._raw_membrane_rows_at_point(0.5, sp, dx, dy, ixe, nxe, iye, nye)
        wy = self._interp_1d_ab(eta, sm, sp)
        row_eyy = wy[0] * byy_m + wy[1] * byy_p

        # exy: tie at four interior/corner-like points, bilinear interpolation
        _, _, bxy_00 = self._raw_membrane_rows_at_point(sm, sm, dx, dy, ixe, nxe, iye, nye)
        _, _, bxy_10 = self._raw_membrane_rows_at_point(sp, sm, dx, dy, ixe, nxe, iye, nye)
        _, _, bxy_01 = self._raw_membrane_rows_at_point(sm, sp, dx, dy, ixe, nxe, iye, nye)
        _, _, bxy_11 = self._raw_membrane_rows_at_point(sp, sp, dx, dy, ixe, nxe, iye, nye)

        w4 = self._interp_bilinear_4(xi, eta, sm, sp)
        row_exy = (
            w4[0] * bxy_00 +
            w4[1] * bxy_10 +
            w4[2] * bxy_01 +
            w4[3] * bxy_11
        )

        Bm = np.vstack([row_exx, row_eyy, row_exy])
        return Bm

    # ------------------------------------------------------------------
    # element stiffness
    # ------------------------------------------------------------------
    def get_kelem(self, E, nu, thick, dx, dy, ixe, nxe, iye, nye):
        """
        Returns dense block stiffness for local dofs ordered:
            [ w(9), u(9), v(9), thx(6), thy(6) ]

        Returns 25 blocks in (w,u,v,thx,thy)x(w,u,v,thx,thy) order.
        """
        pts, wts = third_order_quadrature()

        Jdet = dx * dy
        xi_x  = 1.0 / dx
        eta_y = 1.0 / dy

        n_w   = 9
        n_u   = 9
        n_v   = 9
        n_thx = 6
        n_thy = 6

        def Z(a, b):
            return np.zeros((a, b))

        # blocks: (w, u, v, thx, thy)
        Kww   = Z(n_w,   n_w)
        Kwu   = Z(n_w,   n_u)
        Kwv   = Z(n_w,   n_v)
        Kwtx  = Z(n_w,   n_thx)
        Kwty  = Z(n_w,   n_thy)

        Kuw   = Z(n_u,   n_w)
        Kuu   = Z(n_u,   n_u)
        Kuv   = Z(n_u,   n_v)
        Kutx  = Z(n_u,   n_thx)
        Kuty  = Z(n_u,   n_thy)

        Kvw   = Z(n_v,   n_w)
        Kvu   = Z(n_v,   n_u)
        Kvv   = Z(n_v,   n_v)
        Kvtx  = Z(n_v,   n_thx)
        Kvty  = Z(n_v,   n_thy)

        Ktxw  = Z(n_thx, n_w)
        Ktxu  = Z(n_thx, n_u)
        Ktxv  = Z(n_thx, n_v)
        Ktxtx = Z(n_thx, n_thx)
        Ktxty = Z(n_thx, n_thy)

        Ktyw  = Z(n_thy, n_w)
        Ktyu  = Z(n_thy, n_u)
        Ktyv  = Z(n_thy, n_v)
        Ktytx = Z(n_thy, n_thx)
        Ktyty = Z(n_thy, n_thy)

        # material matrices
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
        Ds = (ks * G * thick) * np.eye(2)

        # ----------------------------------------------------------
        # BENDING
        # ----------------------------------------------------------
        for _eta, w_eta_raw in zip(pts, wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw

            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)
            Ny1, dNy1 = self._p1_1d(eta)

            for _xi, w_xi_raw in zip(pts, wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = w_xi * w_eta * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)
                Nx1, dNx1 = self._p1_1d(xi)

                # thx : (p1,p2)
                Ntx, Ntx_xi, Ntx_eta = self._tensor_product_basis((Nx1, dNx1), (Ny2, dNy2))
                Ntx_x = Ntx_xi * xi_x
                Ntx_y = Ntx_eta * eta_y

                # thy : (p2,p1)
                Nty, Nty_xi, Nty_eta = self._tensor_product_basis((Nx2, dNx2), (Ny1, dNy1))
                Nty_x = Nty_xi * xi_x
                Nty_y = Nty_eta * eta_y

                D11, D12, D22, D33 = Db[0, 0], Db[0, 1], Db[1, 1], Db[2, 2]

                # kxx
                Ktxtx += wt * (D11 * np.outer(Ntx_x, Ntx_x))

                # kyy
                Ktyty += wt * (D22 * np.outer(Nty_y, Nty_y))

                # coupling
                Ktxty += wt * (D12 * np.outer(Ntx_x, Nty_y))
                Ktytx += wt * (D12 * np.outer(Nty_y, Ntx_x))

                # kxy = thy_x + thx_y
                Ktyty += wt * (D33 * np.outer(Nty_x, Nty_x))
                Ktxtx += wt * (D33 * np.outer(Ntx_y, Ntx_y))
                Ktxty += wt * (D33 * np.outer(Ntx_y, Nty_x))
                Ktytx += wt * (D33 * np.outer(Nty_x, Ntx_y))

        # ----------------------------------------------------------
        # SHEAR (mixed IGA, unchanged from your DRIG idea)
        # ----------------------------------------------------------
        for _eta, w_eta_raw in zip(pts, wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw

            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)
            Ny1, dNy1 = self._p1_1d(eta)

            for _xi, w_xi_raw in zip(pts, wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = w_xi * w_eta * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)
                Nx1, dNx1 = self._p1_1d(xi)

                # w : (p2,p2)
                Nw, Nw_xi, Nw_eta = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))
                Nw_x = Nw_xi * xi_x
                Nw_y = Nw_eta * eta_y

                # thx : (p1,p2)
                Ntx, _, _ = self._tensor_product_basis((Nx1, dNx1), (Ny2, dNy2))

                # thy : (p2,p1)
                Nty, _, _ = self._tensor_product_basis((Nx2, dNx2), (Ny1, dNy1))

                # gamx = w_x + thx
                Kww   += wt * (Ds[0, 0] * np.outer(Nw_x, Nw_x))
                Kwtx  += wt * (Ds[0, 0] * np.outer(Nw_x, Ntx))
                Ktxw  += wt * (Ds[0, 0] * np.outer(Ntx, Nw_x))
                Ktxtx += wt * (Ds[0, 0] * np.outer(Ntx, Ntx))

                # gamy = w_y + thy
                Kww   += wt * (Ds[1, 1] * np.outer(Nw_y, Nw_y))
                Kwty  += wt * (Ds[1, 1] * np.outer(Nw_y, Nty))
                Ktyw  += wt * (Ds[1, 1] * np.outer(Nty, Nw_y))
                Ktyty += wt * (Ds[1, 1] * np.outer(Nty, Nty))

        # ----------------------------------------------------------
        # MEMBRANE (MITC-style assumed membrane strains)
        # ----------------------------------------------------------
        for _eta, w_eta_raw in zip(pts, wts):
            eta = 0.5 * (_eta + 1.0)
            w_eta = 0.5 * w_eta_raw

            for _xi, w_xi_raw in zip(pts, wts):
                xi = 0.5 * (_xi + 1.0)
                w_xi = 0.5 * w_xi_raw
                wt = w_xi * w_eta * Jdet

                Bm = self._Bm_mitc_at_quad(xi, eta, dx, dy, ixe, nxe, iye, nye)
                Kmem = wt * (Bm.T @ Dm @ Bm)

                # unpack into block form
                i0, i1, i2, i3, i4, i5 = 0, 9, 18, 27, 33, 39

                Kww += Kmem[i0:i1, i0:i1]
                Kwu += Kmem[i0:i1, i1:i2]
                Kwv += Kmem[i0:i1, i2:i3]
                Kwtx += Kmem[i0:i1, i3:i4]
                Kwty += Kmem[i0:i1, i4:i5]

                Kuw += Kmem[i1:i2, i0:i1]
                Kuu += Kmem[i1:i2, i1:i2]
                Kuv += Kmem[i1:i2, i2:i3]
                Kutx += Kmem[i1:i2, i3:i4]
                Kuty += Kmem[i1:i2, i4:i5]

                Kvw += Kmem[i2:i3, i0:i1]
                Kvu += Kmem[i2:i3, i1:i2]
                Kvv += Kmem[i2:i3, i2:i3]
                Kvtx += Kmem[i2:i3, i3:i4]
                Kvty += Kmem[i2:i3, i4:i5]

                Ktxw += Kmem[i3:i4, i0:i1]
                Ktxu += Kmem[i3:i4, i1:i2]
                Ktxv += Kmem[i3:i4, i2:i3]
                Ktxtx += Kmem[i3:i4, i3:i4]
                Ktxty += Kmem[i3:i4, i4:i5]

                Ktyw += Kmem[i4:i5, i0:i1]
                Ktyu += Kmem[i4:i5, i1:i2]
                Ktyv += Kmem[i4:i5, i2:i3]
                Ktytx += Kmem[i4:i5, i3:i4]
                Ktyty += Kmem[i4:i5, i4:i5]

        return (
            Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
            Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
            Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
            Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
            Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
        )
    
        # -------------------------------------------------------------------------
    # Element load vector for layout [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    def get_felem(
        self,
        load_fcn,     # callable q(x,y) acting on w only
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        ixe: int, nxe: int,
        iye: int, nye: int,
    ):
        """
        Consistent load vector for q(x,y) acting on w only.

        Local unknown ordering:
            [ w(9), u(9), v(9), thx(6), thy(6) ]

        Returns:
            fw(9), fu(9), fv(9), ftx(6), fty(6)
        """
        pts, wts = third_order_quadrature()

        fw  = np.zeros(9)   # w   : (p2,p2)
        fu  = np.zeros(9)   # u   : (p2,p2)
        fv  = np.zeros(9)   # v   : (p2,p2)
        ftx = np.zeros(6)   # thx : (p1,p2)
        fty = np.zeros(6)   # thy : (p2,p1)

        Jdet = dx * dy

        for _eta, w_eta_raw in zip(pts, wts):
            w_eta = 0.5 * w_eta_raw
            eta = 0.5 * (_eta + 1.0)

            Ny2, dNy2 = self._iga2_1d(eta, iye, nye)

            for _xi, w_xi_raw in zip(pts, wts):
                w_xi = 0.5 * w_xi_raw
                xi = 0.5 * (_xi + 1.0)

                wt = (w_xi * w_eta) * Jdet

                Nx2, dNx2 = self._iga2_1d(xi, ixe, nxe)

                Nw, _, _ = self._tensor_product_basis((Nx2, dNx2), (Ny2, dNy2))

                xq = x0 + xi * dx
                yq = y0 + eta * dy
                q = float(load_fcn(xq, yq))

                fw += (q * Nw) * wt

        return fw, fu, fv, ftx, fty

    # -------------------------------------------------------------------------
    # BCs for global vector layout: [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    def apply_bcs_2d(
        self, u: np.ndarray,
        nxw: int, nyw: int,
        nxu: int, nyu: int,
        nxv: int, nyv: int,
        nxtx: int, nytx: int,
        nxty: int, nyty: int,
    ):
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
        ty = u[off:off+nty];  off += nty

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

        # w boundary always
        for j in range(nyw):
            for i in range(nxw):
                if on_bndry(i, j, nxw, nyw):
                    w[i + nxw*j] = 0.0

        if self.clamped:
            for j in range(nyu):
                for i in range(nxu):
                    if on_bndry(i, j, nxu, nyu):
                        U[i + nxu*j] = 0.0

            for j in range(nyv):
                for i in range(nxv):
                    if on_bndry(i, j, nxv, nyv):
                        V[i + nxv*j] = 0.0

            for j in range(nytx):
                for i in range(nxtx):
                    if on_bndry(i, j, nxtx, nytx):
                        tx[i + nxtx*j] = 0.0

            for j in range(nyty):
                for i in range(nxty):
                    if on_bndry(i, j, nxty, nyty):
                        ty[i + nxty*j] = 0.0
        else:
            # SS-ish: u=0 on x=0 edge, v=0 on y=0 edge
            for j in range(nyu):
                U[0 + nxu*j] = 0.0

            for i in range(nxv):
                V[i + nxv*0] = 0.0

        return np.concatenate([w, U, V, tx, ty])

    # -------------------------------------------------------------------------
    # Multigrid transfers (dyadic refinement) for [w, u, v, thx, thy]
    # -------------------------------------------------------------------------
    @staticmethod
    def _kron2(Ry: np.ndarray, Rx: np.ndarray) -> np.ndarray:
        return np.kron(Ry, Rx)

    def _build_R_p2(self, nxe_c: int) -> np.ndarray:
        n_c = nxe_c + 2
        n_f = 2 * nxe_c + 2

        R = np.zeros((n_c, n_f))
        counts = 1e-14 * np.ones((n_c, n_f))

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

            l_nz = l_mat / (l_mat + 1e-14)
            R[ielem_c:ielem_c+3, left_felem:left_felem+3] += l_mat
            counts[ielem_c:ielem_c+3, left_felem:left_felem+3] += l_nz

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

            r_nz = r_mat / (r_mat + 1e-14)
            R[ielem_c:ielem_c+3, right_felem:right_felem+3] += r_mat
            counts[ielem_c:ielem_c+3, right_felem:right_felem+3] += r_nz

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

    def _build_R_p1(self, nxe_c: int) -> np.ndarray:
        n_c = nxe_c + 1
        n_f = 2 * nxe_c + 1
        R = np.zeros((n_c, n_f))
        R[0, 0] = 1.0
        R[-1, -1] = 1.0
        for i in range(1, n_c - 1):
            R[i, 2*i - 1] = 0.25
            R[i, 2*i]     = 0.50
            R[i, 2*i + 1] = 0.25
        return R

    def build_prolongation_operator(self, nxe_c: int, nye_c: int) -> sp.csr_matrix:
        """
        Build full prolongation P : u_f = P u_c for layout [w, u, v, thx, thy].
        """

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 2, nye_c + 2
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 2
        nxv_c,  nyv_c  = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        nw_c  = nxw_c  * nyw_c
        nu_c  = nxu_c  * nyu_c
        nv_c  = nxv_c  * nyv_c
        ntx_c = nxtx_c * nytx_c
        nty_c = nxty_c * nyty_c
        N_c = nw_c + nu_c + nv_c + ntx_c + nty_c

        # fine sizes
        nxe_f, nye_f = 2 * nxe_c, 2 * nye_c
        nxw_f,  nyw_f  = nxe_f + 2, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 2
        nxv_f,  nyv_f  = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        nw_f  = nxw_f  * nyw_f
        nu_f  = nxu_f  * nyu_f
        nv_f  = nxv_f  * nyv_f
        ntx_f = nxtx_f * nytx_f
        nty_f = nxty_f * nyty_f
        N_f = nw_f + nu_f + nv_f + ntx_f + nty_f

        # 1D prolongations
        Rx2 = self._build_R_p2(nxe_c); Px2 = Rx2.T
        Ry2 = self._build_R_p2(nye_c); Py2 = Ry2.T

        Px1 = self._build_P_p1(nxe_c)
        Py1 = self._build_P_p1(nye_c)

        Px2 = sp.csr_matrix(Px2); Py2 = sp.csr_matrix(Py2)
        Px1 = sp.csr_matrix(Px1); Py1 = sp.csr_matrix(Py1)

        # 2D prolongations
        Pw  = sp.kron(Py2, Px2, format="csr")  # w   : (p2,p2)
        Pu  = sp.kron(Py2, Px2, format="csr")  # u   : (p2,p2)
        Pv  = sp.kron(Py2, Px2, format="csr")  # v   : (p2,p2)
        Ptx = sp.kron(Py2, Px1, format="csr")  # thx : (p1,p2)
        Pty = sp.kron(Py1, Px2, format="csr")  # thy : (p2,p1)

        P = sp.block_diag((Pw, Pu, Pv, Ptx, Pty), format="csr")
        assert P.shape == (N_f, N_c)
        return P

    def apply_bcs_to_prolongation(
        self,
        P: sp.csr_matrix,
        fine_bc_dofs: np.ndarray,
        coarse_bc_dofs: np.ndarray = None,
        inject_identity_on_fine: bool = False,
    ) -> sp.csr_matrix:
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

    def get_bc_dofs_for_layout(
        self,
        nxw: int, nyw: int,
        nxu: int, nyu: int,
        nxv: int, nyv: int,
        nxtx: int, nytx: int,
        nxty: int, nyty: int,
    ) -> np.ndarray:
        """
        Return global constrained DOF indices for layout [w, u, v, thx, thy].
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
            for j in range(nyu):
                bc.append(off_u + (0 + nxu * j))

            for i in range(nxv):
                bc.append(off_v + (i + nxv * 0))

        bc = np.array(bc, dtype=int)
        if bc.size:
            bc = np.unique(bc)
        return bc

    def build_bc_prolongation_operator(self, nxe_c: int, nye_c: int):
        P = self.build_prolongation_operator(nxe_c, nye_c)

        # fine sizes
        nxe_f, nye_f = 2 * nxe_c, 2 * nye_c
        nxw_f,  nyw_f  = nxe_f + 2, nye_f + 2
        nxu_f,  nyu_f  = nxe_f + 2, nye_f + 2
        nxv_f,  nyv_f  = nxe_f + 2, nye_f + 2
        nxtx_f, nytx_f = nxe_f + 1, nye_f + 2
        nxty_f, nyty_f = nxe_f + 2, nye_f + 1

        # coarse sizes
        nxw_c,  nyw_c  = nxe_c + 2, nye_c + 2
        nxu_c,  nyu_c  = nxe_c + 2, nye_c + 2
        nxv_c,  nyv_c  = nxe_c + 2, nye_c + 2
        nxtx_c, nytx_c = nxe_c + 1, nye_c + 2
        nxty_c, nyty_c = nxe_c + 2, nye_c + 1

        fine_bc_dofs = self.get_bc_dofs_for_layout(
            nxw_f, nyw_f, nxu_f, nyu_f, nxv_f, nyv_f, nxtx_f, nytx_f, nxty_f, nyty_f
        )
        coarse_bc_dofs = self.get_bc_dofs_for_layout(
            nxw_c, nyw_c, nxu_c, nyu_c, nxv_c, nyv_c, nxtx_c, nytx_c, nxty_c, nyty_c
        )

        return self.apply_bcs_to_prolongation(
            P,
            fine_bc_dofs=fine_bc_dofs,
            coarse_bc_dofs=coarse_bc_dofs,
            inject_identity_on_fine=False,
        )

    def _energy_smooth_jacobi_v1(
        self,
        nxe_c: int,
        n_sweeps: int = 10,
        omega: float = 0.7,
        with_fillin: bool = False,
        use_mask: bool = True,
    ):
        import scipy.sparse as sp

        P0 = self.build_bc_prolongation_operator(nxe_c, nxe_c)
        P = P0.tocsr(copy=True)

        if (int(nxe_c) not in self._kmat_cache):
            raise RuntimeError("Expected self._kmat_cache[nxe_c] to exist for energy smoothing.")
        K = self._kmat_cache[int(nxe_c)]
        K = K.tocsr() if sp.isspmatrix(K) else sp.csr_matrix(K)

        N, M = K.shape
        if N != M:
            raise ValueError(f"kmat must be square, got {K.shape}")
        if P.shape[0] != N:
            raise ValueError(f"Shape mismatch: K is {K.shape} but P is {P.shape}")

        D = K.diagonal().astype(float)
        eps = 1e-30
        Dinv = np.zeros_like(D)
        good = np.abs(D) > eps
        Dinv[good] = 1.0 / D[good]

        mask = None
        if (not with_fillin) and use_mask:
            mask = (K @ P)
            mask.data[:] = 1.0
            mask.eliminate_zeros()
            mask = mask.tocsr()

        def control(Z: sp.csr_matrix) -> sp.csr_matrix:
            if mask is None:
                return Z
            return Z.multiply(mask)

        def left_scale_rows_csr(A: sp.csr_matrix, scale: np.ndarray) -> sp.csr_matrix:
            A = A.tocsr(copy=True)
            row_nnz = np.diff(A.indptr)
            A.data *= np.repeat(scale, row_nnz)
            return A

        P = control(P)

        w = float(omega)
        for _ in range(int(n_sweeps)):
            KP = K @ P
            if (not with_fillin) and (mask is not None):
                KP = control(KP)

            DinvKP = left_scale_rows_csr(KP, Dinv)
            P = P - w * DinvKP

            if (not with_fillin) and (mask is not None):
                P = control(P)

        return P.toarray()

    def _assemble_prolongation(self, nxe_f):
        nxe_c = nxe_f // 2
        if nxe_f in self._P_cache:
            return self._P_cache[nxe_f]

        if self.prolong_mode == "standard":
            P = self.build_bc_prolongation_operator(nxe_c, nxe_c)
        elif self.prolong_mode == "energy-jacobi":
            P = self._energy_smooth_jacobi_v1(
                nxe_c,
                self.n_Psweeps,
                self.omega,
                with_fillin=True,
            )
        else:
            raise ValueError(f"Unsupported prolong_mode: {self.prolong_mode}")

        self._P_cache[nxe_f] = sp.csr_matrix(P)
        return self._P_cache[nxe_f]

    def prolongate(self, u_c: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Dyadic prolongation for layout [w, u, v, thx, thy].
        """
        if nxe_c != nye_c:
            raise NotImplementedError("Current cached dyadic prolongation assumes square grids.")

        P = self._assemble_prolongation(2 * nxe_c)
        return P @ np.asarray(u_c)

    def restrict(self, u_f: np.ndarray, nxe_c: int, nye_c: int) -> np.ndarray:
        """
        Galerkin-style restriction via transpose of prolongation.
        """
        if nxe_c != nye_c:
            raise NotImplementedError("Current cached dyadic restriction assumes square grids.")

        P = self._assemble_prolongation(2 * nxe_c)
        return P.T @ np.asarray(u_f)
import numpy as np
from .basis import second_order_quadrature, zero_order_quadrature, first_order_quadrature
from .basis import get_lagrange_basis_2d_all

import scipy.sparse as sp

class PoissonPlateElement:
    # Poisson C0-continuous element

    def __init__(self):              
        self.dof_per_node = 1
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)

    def get_kelem(self, elem_xpts: np.ndarray):
        """
        elem_xpts expected length 12: [x0,y0,z0?, x1,y1,z1?, x2,y2,z2?, x3,y3,z3?]
        You currently use x=elem_xpts[0::3], y=elem_xpts[1::3].
        Assumes an axis-aligned rectangular mapping (x_eta ~ 0, y_xi ~ 0) as in your code.
        """
        # pts, wts = first_order_quadrature()
        pts, wts = second_order_quadrature()
        kelem = np.zeros((self.ndof, self.ndof))

        x = elem_xpts[0::3]
        y = elem_xpts[1::3]

        # ---- weak form term: nabla u dot nabla v ----
        for ii, xi in enumerate(pts):
            for jj, eta in enumerate(pts):
                wt = wts[ii] * wts[jj]

                N, Nxi, Neta = get_lagrange_basis_2d_all(xi, eta)

                # geometry derivatives
                x_xi  = np.dot(Nxi,  x); x_eta = np.dot(Neta, x)
                y_xi  = np.dot(Nxi,  y); y_eta = np.dot(Neta, y)

                # if you're truly axis-aligned rectangles, use diagonal map
                # but keep the general 2x2 inverse in case x_eta,y_xi aren't exactly zero
                J = x_xi * y_eta - x_eta * y_xi
                invJ = 1.0 / J
                xi_x  =  y_eta * invJ
                xi_y  = -x_eta * invJ
                eta_x = -y_xi  * invJ
                eta_y =  x_xi  * invJ

                # physical grads of shape functions
                Nx = Nxi * xi_x + Neta * eta_x
                Ny = Nxi * xi_y + Neta * eta_y

                # dot product of the gradient with itself
                kelem += (np.outer(Nx, Nx) + np.outer(Ny, Ny)) * (wt * J)

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
            felem += fN  # w

        return felem
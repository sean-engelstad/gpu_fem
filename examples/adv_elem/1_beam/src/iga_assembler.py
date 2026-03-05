import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt


def quad_bernstein(xi):
    N = np.array([(1-xi)**2, 2*xi*(1-xi), xi**2])
    dN = np.array([-2*(1-xi), 2*(1-2*xi), 2*xi])
    return N, dN


def get_iga2_basis(xi, left_bndry, right_bndry):
    B, dB = quad_bernstein(xi)

    # bndry adjustment and regular basis
    # on GPU can code it up with ? ternary operators probably
    N = 0.5 * np.array([B[0], np.sum(B) + B[1], B[2]])
    N += 0.5 * left_bndry * np.array([B[0], -B[0], 0.0])
    N += 0.5 * right_bndry * np.array([0.0, -B[2], B[2]])

    # bndry adjustment and regular derivs
    dN = 0.5 * np.array([dB[0], np.sum(dB) + dB[1], dB[2]])
    dN += 0.5 * left_bndry * np.array([dB[0], -dB[0], 0.0])
    dN += 0.5 * right_bndry * np.array([0.0, -dB[2], dB[2]])
    return N, dN

class IGABeamAssembler:
    # for IGA bases or elements

    def __init__(
        self,
        ELEMENT,
        nxe:int,
        E:float=70e9,
        nu:float=0.3,
        thick:float=1.0e-2,
        L:float=1.0,
        load_fcn=lambda x : 1.0,
        clamped:bool=False,
        split_disp_bc:bool=False,
    ):
        
        self.element = ELEMENT
        self.nxe = nxe
        self.E = E
        self.nu = nu
        self.thick = thick
        self.L = L
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc
        
        # internal data
        self.kmat = None
        self.force = None
        self.u = None
        
        assert self.element.ORDER == 2 # 2nd order IGA only currently
        self.nnodes = nxe + 2 # for 2nd order IGA elements p = 2, nnodes = nxe + p generally
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.elem_length = self.L / (self.nnodes - 1)
        # self.elem_length = 2.0 * self.L / (self.nnodes-1)
        dpn = self.dof_per_node
        
        if clamped:
            self.element.clamped = True
            if dpn == 3:
                self.bcs = [0,dpn*(self.nnodes-1)]
            else:
                # standard fully clamped
                self.bcs = list(range(dpn)) + list(range(dpn*(self.nnodes-1), dpn*self.nnodes))
        else:
            self.element.clamped = False
            self.bcs = [0, dpn*(self.nnodes-1)]
        
        self.conn = [[ielem,ielem+1,ielem+2] for ielem in range(self.nxe)]

        # matrix sparsity
        self.rowp = [0]; self.cols = []; self.nnzb = 0
        for inode in range(self.nnodes):
            if inode == 0:
                current_cols = [0, 1, 2]
            elif inode == 1:
                current_cols = [0,1,2,3]
            elif inode == self.nnodes-2:
                current_cols = [self.nnodes-4, self.nnodes-3, self.nnodes-2, self.nnodes-1]
            elif inode == self.nnodes-1:
                current_cols = [self.nnodes-3, self.nnodes-2, self.nnodes-1]
            else:
                current_cols = [inode-2, inode-1, inode, inode+1, inode+2]
            self.nnzb += len(current_cols)
            self.rowp += [self.nnzb]
            self.cols += current_cols
        # print(f"{self.rowp=}\n{self.nnzb=}\n{self.cols=}")
        self.rowp = np.array(self.rowp); self.cols = np.array(self.cols)
    
    @property
    def dof_conn(self):
        dpn = self.dof_per_node
        return [[dpn*ix + j for j in range(3*dpn)] for ix in range(self.nxe)]


    def _assemble_system(self):
        # assemble BSR matrix
        self.data = np.zeros((self.nnzb, self.dof_per_node, self.dof_per_node), dtype=np.double)
        xbnd_vals = [[ielem * self.elem_length, (ielem+2) * self.elem_length] for ielem in range(self.nxe)]

        interior_kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=False, right_bndry=False)
        dpn = self.dof_per_node
        self.force = np.zeros(self.N)
        
        # compute LHS and RHS no BCs
        for ielem in range(self.nxe):
            # print(f"{ielem=} {xbnd_vals[ielem]=}")
            if ielem == 0:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=True, right_bndry=False)
                felem = self.element.get_felem(mag=self.load_fcn, elem_length=self.elem_length, left_bndry=True, right_bndry=False, xbnd=xbnd_vals[ielem])
            elif ielem == self.nxe - 1:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=False, right_bndry=True)
                felem = self.element.get_felem(mag=self.load_fcn, elem_length=self.elem_length, left_bndry=False, right_bndry=True, xbnd=xbnd_vals[ielem])
            else: # interior
                kelem = interior_kelem
                felem = self.element.get_felem(mag=self.load_fcn, elem_length=self.elem_length, left_bndry=False, right_bndry=False, xbnd=xbnd_vals[ielem])

            local_conn = np.array(self.dof_conn[ielem])
            # add kelem into LHS sparse structure
            for lblock_row,block_row in enumerate([ielem, ielem+1, ielem+2]):
                for colp in range(self.rowp[block_row], self.rowp[block_row+1]):
                    block_col = self.cols[colp]
                    if block_col in [ielem, ielem+1, ielem+2]:
                        lblock_col = block_col - ielem

                        # my_mat = kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                        #                              dpn*lblock_col:dpn*(lblock_col+1)]
                        # print(f"{my_mat.shape=} {lblock_col=} {kelem.shape=}")                    
                        
                        self.data[colp,:,:] += kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                                                     dpn*lblock_col:dpn*(lblock_col+1)]

            # add felem into RHS
            np.add.at(self.force, local_conn, felem)

        if self.split_disp_bc:
            # dpn = 3 with local dofs: [w_b, (dw/dxi)_b, w_s]
            # SS: enforce w_b + w_s = 0 at BOTH ends by overwriting the w_b row (idof=0).
            # Extra gauge-fix: pin w_s(0) = 0 by overwriting the w_s row (idof=2) at the left end.

            # ---- LEFT END (node 0): w_b + w_s = 0 (overwrite w_b row) ----
            # tried changing it to just w_b = 0 on left side since also w_s = 0
            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0  # w_b row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0  # w_b
                    self.data[colp, 0, 1] = 1.0  # w_s

            # ---- LEFT END (node 0): gauge fix w_s(0) = 0 (overwrite w_s row) ----
            # the extra gauge constraint here removes constant mode from integrated shear strains th_s => w_s (cause non-unique)
            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 1  # w_s row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 1, 1] = 1.0  # w_s = 0

            # ---- RIGHT END (node nnodes-1): w_b + w_s = 0 (overwrite w_b row) ----
            inode = self.nnodes - 1
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0  # w_b row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0  # w_b
                    self.data[colp, 0, 1] = 1.0  # w_s

            # RHS for those constraint rows:
            self.force[dpn * 0 + 0] = 0.0                 # (w_b + w_s)(0) = 0
            self.force[dpn * 0 + 1] = 0.0                 # w_s(0) = 0  (gauge fix)
            self.force[dpn * (self.nnodes - 1) + 0] = 0.0 # (w_b + w_s)(L) = 0


        else: # not split disp BC (regular SS or clamped)

            # apply bcs to LHS and RHS
            # node 1 - SS BC
            for colp in range(self.rowp[0], self.rowp[1]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = idof
                    if not(row in self.bcs): continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0
            # last node - SS BC
            for colp in range(self.rowp[self.nnodes-1], self.rowp[self.nnodes]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = dpn * (self.nnodes-1) + idof
                    if not(row in self.bcs): continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0

            for bc in self.bcs:
                self.force[bc] = 0.0
        
        self.kmat = sp.bsr_matrix(
            (self.data, self.cols, self.rowp),
            shape=(self.N, self.N)
        )

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        # print(f"{self.u=}")
        return self.u
    
    def get_node_pts(self) -> list:
        """
        Quadratic (p=2) 1D IGA: return physical 'node points' using Greville abscissae.
        Output length = nxe + 2.

        u_i = (U[i+1] + U[i+2]) / 2   for i = 0..n_ctrl-1, with n_ctrl = nxe + 2
        x(u) = sum_j N_j(u) * x_ctrl[j]
        """
        nxe = int(self.nxe)
        p = 2
        n_ctrl = nxe + p  # = nxe + 2

        # Open-uniform knot vector on [0,1], length = n_ctrl + p + 1 = nxe + 5
        U = np.array([0.0] * (p + 1) +
                    [i / nxe for i in range(1, nxe)] +
                    [1.0] * (p + 1), dtype=float)

        x_ctrl = np.asarray(self.control_pts, dtype=float)
        if x_ctrl.size != n_ctrl:
            raise ValueError(
                f"Expected {n_ctrl} control points for p=2, nxe={nxe}, "
                f"but got {x_ctrl.size}."
            )

        def find_span(n_ctrl, degree, u, U):
            # Cox–de Boor span search; returns span in [degree, n_ctrl-1]
            if u >= U[-1] - 1e-14:
                return n_ctrl - 1
            low = degree
            high = len(U) - degree - 2
            mid = (low + high) // 2
            while True:
                if u < U[mid]:
                    high = mid - 1
                elif u >= U[mid + 1]:
                    low = mid + 1
                else:
                    return mid
                mid = (low + high) // 2

        
        def basis_functions_and_derivatives(span, u, degree, U, n_deriv=1):
            # Compute nonzero basis functions and first derivatives using Cox-de Boor + derivative formula
            # Returns arrays N[0:degree] and dN[0:degree]
            left = np.zeros(degree+1)
            right = np.zeros(degree+1)
            ndu = np.zeros((degree+1, degree+1))
            ndu[0,0] = 1.0
            for j in range(1, degree+1):
                left[j] = u - U[span+1-j]
                right[j] = U[span+j] - u
                saved = 0.0
                for r in range(j):
                    ndu[j,r] = right[r+1] + left[j-r]
                    temp = ndu[r,j-1]/ndu[j,r]
                    ndu[r,j] = saved + right[r+1]*temp
                    saved = left[j-r]*temp
                ndu[j,j] = saved
            N = ndu[:,degree].copy()
            # derivatives
            ders = np.zeros((n_deriv+1, degree+1))
            a = np.zeros((2, degree+1))
            # compute a triangular table of derivatives
            for r in range(degree+1):
                s1 = 0; s2 = 1
                a[0,0] = 1.0
                for k in range(1, n_deriv+1):
                    d = 0.0
                    rk = r - k
                    pk = degree - k
                    if r >= k:
                        a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                        d = a[s2,0]*ndu[rk,pk]
                    j1 = 1 if rk >= -1 else -rk
                    j2 = k-1 if r-1 <= pk else degree - r
                    for j in range(j1, j2+1):
                        a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1, rk+j]
                        d += a[s2,j]*ndu[rk+j, pk]
                    if r <= pk:
                        a[s2,k] = -a[s1,k-1]/ndu[pk+1, r]
                        d += a[s2,k]*ndu[r, pk]
                    ders[k,r] = d
                    s1, s2 = s2, s1
            # Multiply by correct factors
            for k in range(1, n_deriv+1):
                for j in range(degree+1):
                    ders[k,j] *= degree
            return N, ders[1]
        # Greville abscissae (parametric "nodes"), length n_ctrl = nxe+2 for p=2
        u_nodes = 0.5 * (U[1:1 + n_ctrl] + U[2:2 + n_ctrl])

        node_pts = []
        for u in u_nodes:
            span = find_span(n_ctrl, p, float(u), U)
            N, _ = basis_functions_and_derivatives(span, float(u), p, U, n_deriv=1)

            i0 = span - p  # active basis indices: i0..i0+p
            x = 0.0
            for a in range(p + 1):
                x += N[a] * x_ctrl[i0 + a]
            node_pts.append(float(x))
        # print(f"{node_pts=}\n{self.control_pts=}")

        return node_pts
    
    @property
    def control_pts(self) -> list:
        # control points for IGA
        return [i*self.elem_length for i in range(self.nnodes)]
    
    @property
    def xvec(self) -> list:
        return self.control_pts # just for simple plots debugging

    def plot_disp(self, idof:int=0):
        xvec = self.get_node_pts() # not same as control points
        # xvec = self.control_pts
        # print(f"{self.u=}")
        dpn = self.dof_per_node
        w = self.u[idof::dpn]
        if self.split_disp_bc:
            if dpn == 3: # hhd hermite hierarchic disp
                w = self.u[0::3] + self.u[2::3] # wb + ws
            elif dpn == 2: # higd iga hierarchic disp
                w = self.u[0::2] + self.u[1::2] # wb + ws
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)" if idof == 0 else "th(x)")
        plt.show()     

    def get_max_deflection_greville(self):
        """
        Return max(|w|) evaluated at Greville abscissae (physical points), not at control DOFs.

        Assumes quadratic (p=2) open-uniform IGA with nnodes = nxe + 2 and element connectivity
        [e, e+1, e+2] for e = 0..nxe-1.

        Uses your get_node_pts() for Greville x-locations, then for each Greville point xg:
        - find element e containing xg
        - evaluate quadratic IGA basis N(xi) on that element
        - compute w(xg) = sum_a N_a(xi) * w_ctrl[e+a]

        For split_disp_bc:
        dpn==3 : w_ctrl = u_wb + u_ws  (0::3 + 2::3)  [matches your plot_disp]
        dpn==2 : w_ctrl = u_wb + u_ws  (0::2 + 1::2)
        """
        if self.u is None:
            raise RuntimeError("Call direct_solve() (or otherwise set self.u) before evaluating deflection.")

        dpn = self.dof_per_node

        # control DOFs for w (on control points)
        if self.split_disp_bc:
            if dpn == 3:
                w_ctrl = self.u[0::3] + self.u[2::3]
            elif dpn == 2:
                w_ctrl = self.u[0::2] + self.u[1::2]
            else:
                w_ctrl = self.u[0::dpn]
        else:
            w_ctrl = self.u[0::dpn]

        # Greville physical points (length = nnodes)
        xg = np.asarray(self.get_node_pts(), dtype=float)

        # evaluate w(x) at each Greville point using local element basis
        w_g = np.zeros_like(xg)

        # clamp helper
        def _clamp(a, lo, hi):
            return lo if a < lo else (hi if a > hi else a)

        L = float(self.L)
        h = float(self.elem_length)

        for ig, x in enumerate(xg):
            # choose containing element index e in [0, nxe-1]
            if x <= 0.0:
                e = 0
            elif x >= L:
                e = self.nxe - 1
            else:
                e = int(np.floor(x / h))
                e = _clamp(e, 0, self.nxe - 1)

            # local coord xi in [0,1] for this element
            x0 = e * h
            xi = (x - x0) / h
            xi = _clamp(xi, 0.0, 1.0)

            # basis on this element (use boundary flags like your assembly)
            left_bndry = (e == 0)
            right_bndry = (e == self.nxe - 1)

            N, _ = get_iga2_basis(float(xi), left_bndry, right_bndry)  # expects xi in [0,1]

            # local control indices are [e, e+1, e+2]
            w_g[ig] = N[0] * w_ctrl[e] + N[1] * w_ctrl[e + 1] + N[2] * w_ctrl[e + 2]

        return float(np.max(np.abs(w_g)))


    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.L)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.L)
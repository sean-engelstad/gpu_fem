import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class ILU0Preconditioner:
    def __init__(self, A: sp.csr_matrix, dof_per_node=3):
        """
        Build ILU(0) preconditioner with node-wise random reordering.
        """
        A = A.copy()
        assert sp.isspmatrix_csr(A)
        n_dof = A.shape[0]
        n_nodes = n_dof // dof_per_node

        # Random permutation of nodes
        node_perm = np.random.permutation(n_nodes)
        dof_perm = np.concatenate([
            np.arange(i*dof_per_node, (i+1)*dof_per_node)
            for i in node_perm
        ])

        # temp debug
        # dof_perm = np.arange(n_dof)

        A_perm = A[dof_perm, :][:, dof_perm]
        # ilu = spla.spilu(A_perm, fill_factor=1.0, drop_tol=0.0)
        ilu = spla.spilu(A_perm, fill_factor=10.0, drop_tol=1e-5)

        self.solve = ilu.solve
        self.perm = dof_perm
        self.perm_inv = np.argsort(dof_perm)

    def apply(self, r):
        """
        Apply M⁻¹ to r: z = M⁻¹ r using ILU(0) solve.
        """
        r_perm = r[self.perm]
        z_perm = self.solve(r_perm)
        return z_perm[self.perm_inv]
    
def pcg(A, b, M: ILU0Preconditioner, x0=None, tol=1e-8, maxiter=200):
    """
    PCG with right preconditioning: solve A M⁻¹ y = b, x = M⁻¹ y
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Right-preconditioned system: A M⁻¹ y = b, where x = M⁻¹ y
    r = b - A @ x
    z = M.apply(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    for i in range(maxiter):
        Ap = A @ p
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        if np.linalg.norm(r) < tol:
            print(f"PCG converged in {i+1} iterations.")
            return x

        z = M.apply(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    print("PCG did not converge within maxiter.")
    return x

class SchwarzSmoother:
    def __init__(self, K:np.ndarray, nx_sd:int=4, overlap_frac:float=0.2, additive:bool=True):
        self.K = K.copy()
        self.nx_sd = nx_sd # num of subdomains in x-dir
        self.overlap_frac = overlap_frac
        self.additive = additive

        self.ndof = K.shape[0]
        self.nnodes = self.ndof // 3
        self.nx = int(self.nnodes**0.5)
        self.nxe = self.nx - 1
        self._nxe_sd_no = self.nxe / nx_sd # no overlap (keep as float here)
        self.nx_sd_no = self._nxe_sd_no + 1
        self._nxe_sd_wo = self._nxe_sd_no * overlap_frac # with overlap part each side (overlap only though)
        self.nx_sd_wo = self._nxe_sd_wo + 1
        self.N_sd = nx_sd**2

    def _get_1d_domain(self, ix_sd):
        """get the start and stop nodes for 1d subdomain"""
        ix_no = self._nxe_sd_no * np.array([ix_sd, ix_sd + 1])
        ix_no[-1] + 1
        # print(F"{ix_no=}")
        ix_no[0] -= self._nxe_sd_wo
        ix_no[-1] += self._nxe_sd_wo
        ix_no[0] = np.max([0.0, ix_no[0]])
        ix_no[-1] = np.min([self.nx - 1, ix_no[-1]])
        ix_no[0] = np.floor(ix_no[0])
        ix_no[-1] = np.ceil(ix_no[-1])
        return ix_no.astype(np.int32)

    @property
    def sd_nodes_list(self):
        nodes_list = []
        for i_sd in range(self.N_sd):
            ix_sd, iy_sd = i_sd % self.nx_sd, i_sd // self.nx_sd
            ix_bounds = self._get_1d_domain(ix_sd)
            iy_bounds = self._get_1d_domain(iy_sd)

            _sd_nodes = []
            for iy in range(iy_bounds[0], iy_bounds[1] + 1):
                for ix in range(ix_bounds[0], ix_bounds[1] + 1):
                    _sd_nodes += [self.nx * iy + ix]
            nodes_list += [_sd_nodes]
        return nodes_list
    
    def _get_boundary_nodes_on_sd(self, sd_nodes:np.ndarray):
        ix_list, iy_list = sd_nodes % self.nx, sd_nodes // self.nx
        ix_min, ix_max = np.min(ix_list), np.max(ix_list)
        iy_min, iy_max = np.min(iy_list), np.max(iy_list)
        x_bndry_mask = np.logical_or(ix_list == ix_min, ix_list == ix_max)
        y_bndry_mask = np.logical_or(iy_list == iy_min, iy_list == iy_max)
        bndry_mask = np.logical_or(x_bndry_mask, y_bndry_mask)
        return sd_nodes[bndry_mask]
    
    def _node_to_dof_list(self, node_list):
        # return np.array([3*inode+idof for inode in node_list for idof in range(3)])
        return np.array([3*inode+idof for inode in node_list for idof in range(1)]) # here try just w dof constrained on schwarz subdomains
    
    def _get_bndry_sd_dof(self, sd_dof, bndry_dof):
        bndry_sd_dof = []
        for i, inode in enumerate(sd_dof):
            if inode in bndry_dof:
                bndry_sd_dof += [i]
        return np.array(bndry_sd_dof)

    def smooth(self, b0, x0, omega:float=1.0):
        """do additive or multiplicative schwarz smoothing"""
        b = b0.copy()
        x0 = x0.copy()
        x = x0.copy()

        if not self.additive and not(omega == 1.0):
            print(f"warning changed {omega=:.2e} back to 1.0 for multiplicative schwarz") 
            omega = 1.0 # change back to full update if multiplicative schwarz

        nodes_list = self.sd_nodes_list
        for i_sd in range(self.N_sd):
            sd_nodes = np.array(nodes_list[i_sd])
            bndry_nodes = self._get_boundary_nodes_on_sd(sd_nodes)
            sd_dof, bndry_dof = self._node_to_dof_list(sd_nodes), self._node_to_dof_list(bndry_nodes)

            # now compute smaller problem
            bndry_sd_dof = self._get_bndry_sd_dof(sd_dof, bndry_dof)
            K_sd = self.K[sd_dof, :][:, sd_dof]

            int_dof = np.array([idof for idof in sd_dof if not(idof in bndry_dof)])
            int_sd_dof = self._get_bndry_sd_dof(sd_dof, int_dof)
            K_sd_red = K_sd[int_sd_dof,:][:,int_sd_dof]
            K_FC = K_sd[int_sd_dof, :][:, bndry_sd_dof]
            x_c = x[bndry_dof]
            
            # compute rhs accounting for non-zero disp boundaries (cause disp boundaries in middle)
            if self.additive:
                b_sd = b0[int_dof]
                x_c = x0[bndry_dof]
                b_sd -= np.dot(K_FC, x_c) # nz dirichlet bc adjustment

            else:
                b_sd = b[int_dof]
                x_c = x[bndry_dof]
                b_sd -= np.dot(K_FC, x_c)

            dx_sd = np.linalg.solve(K_sd_red, b_sd) * omega # damp the update here..
            x[int_dof] += dx_sd

            # now adjust the rhs
            dx_sd_full = np.zeros(sd_dof.shape[0])
            dx_sd_full[int_sd_dof] = dx_sd[:]
            # need to include nodal force updates on subdomain bndry also as interior nodes near it do affect these points
            b[sd_dof] -= np.dot(K_sd, dx_sd_full) 
    
        return x, b
    
    def multi_smooth(self, b0, x0, num_smooth:int=2, omega:float=1.0):
        """do multiple schwarz smoothing steps"""

        # b = b0.copy()
        # x = x0.copy()
        # for ismooth in range(num_smooth):
        #     x, b = self.smooth(b, x0=x, omega=omega)
        # return x, b
    
    # TODO : need zero disps each time like below?
        b = b0.copy()
        x, x0 = x0.copy(), x0.copy()
        for ismooth in range(num_smooth):
            _du, b = self.smooth(b, x0=np.zeros(self.ndof), omega=omega)
            # x += _du
            x -= _du # TODO : plus or minus here?
        return x, b

            

def gauss_seidel_csr(A, b, x0, num_iter=1):
    """
    Perform Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]
    for _ in range(num_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]
            x[i] = (b[i] - sum_) / diag
    return x

def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, ndof:int=3):
    """
    Perform Block Gauss-Seidel smoothing for 3 DOF per node.
    A: csr_matrix of size (3*nnodes, 3*nnodes)
    b: RHS vector (3*nnodes,)
    x0: initial guess (3*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x
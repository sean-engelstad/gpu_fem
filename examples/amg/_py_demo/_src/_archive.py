

class AMG_BSRSolver:
    """general multilevel AMG solver..."""
    def __init__(self, A_free:sp.bsr_matrix, A:sp.bsr_matrix, B:np.ndarray, threshold:float=0.25,
                 omega:float=0.7, pre_smooth=1, post_smooth=1, level:int=0, ncyc:int=1, near_kernel:bool=True):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        B : fine rigid body modes
        threshold: float for coarsening threshold
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)

        self.A_free = A_free
        self.A = A
        self.near_kernel = near_kernel
        bs = self.A_free.data.shape[-1]
        nnodes = self.A_free.shape[0] // bs
        self.B = B
        self.ncyc = ncyc # how many inner AMG cycles before Krylov step
        self.iters = 0 # count how many AMG cycles used

        # compute node aggregate sets
        # make sure to use unconstrained matrix for aggregation indicators originally
        aggregate_ind = greedy_serial_aggregation_bsr(A_free, threshold=threshold)
        num_agg = np.max(aggregate_ind) + 1

        # print(f"{aggregate_ind=}")
        # TODO: do this
        # bc_flags = None
        bc_flags = get_bc_flags_bsr(A)
        # print(f"{bc_flags=}")

        # create tentative prolongator then smooth it
        block_dim = A.data.shape[-1]
        self.T, self.Bc = tentative_prolongator_bsr(self.B, aggregate_ind, bc_flags, vpn=block_dim)
        # self.P = smooth_prolongator_bsr(self.T, A, self.Bc, bc_flags, omega=omega, near_kernel=near_kernel) # single damped jacobi step, so only one step of fillin
        self.P = smooth_prolongator_bsr_iterative(self.T, A, self.Bc, bc_flags, omega=omega, near_kernel=near_kernel, niter=1) # single damped jacobi step, so only one step of fillin
        # self.P = self.T # DEBUG
        self.R = self.P.T # sym matrix so restriction is transpose prolong

        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.level = level

        # Galerkin coarse operator
        self.Ac = self.R @ (A @ self.P)
        # on GPU would not do extra allocation for this
        self.Ac_free = self.R @ (A_free @ self.P)

        self.Ac = self.Ac.tobsr()
        self.Ac_free = self.Ac_free.tobsr()
        # print(f"{type(self.Ac)=}")

        # plt.spy(self.Ac_free)
        # plt.show()
        # plt.imshow(self.Ac_free.toarray())
        # plt.show()

        # compute two-grid operator complexity
        self.fine_nnz = self.A.nnz
        self.coarse_nnz = self.Ac.nnz
        self.coarse_solver = None 

        if level == 0:
            print("level 0 is AMG solver..")

        # check fillin of coarse grid solver..
        coarse_nnodes = self.Ac.shape[0]
        max_coarse_nnz = coarse_nnodes**2

        # print(f"{level=} {self.fine_nnz=} {self.coarse_nnz=} {max_coarse_nnz=}")
        if self.fine_nnz == self.coarse_nnz:
            raise RuntimeError(f"ERROR: {self.fine_nnz=} == {self.coarse_nnz=} : lower aggregation threshold from {threshold=} to lower value..\n")

        if self.coarse_nnz >= 0.4 * max_coarse_nnz or coarse_nnodes <= 100:
            # then do direct solver
            print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            print(f"level {level+1} building AMG solver")
            # print(f"{type(self.Ac_free)=} {type(self.Ac)=}")
            self.coarse_solver = AMG_BSRSolver(self.Ac_free, self.Ac, self.Bc, threshold, omega, pre_smooth, post_smooth, level+1, near_kernel=near_kernel)

        if level == 0:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=}")
            print(f"\tnum nodes per level = [{self.num_nodes_list}]")

    @property
    def total_nnz(self) -> int:
        # get total nnz across all levels
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else: # direct solver
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz     

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return self.coarse_solver.num_levels + 1
        else: # direct solver
            return 2
        
    @property
    def num_nodes_list(self) -> str:
        bs = self.A.data.shape[-1]
        nnodes_f = self.A.shape[0] // bs
        nnodes_c = self.Ac.shape[0] // bs
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return str(nnodes_f) + "," + self.coarse_solver.num_nodes_list
        else: # direct solver
            return f"{nnodes_f},{nnodes_c}"

    # def solve(self, rhs):
    #     """
    #     Fully symmetric AMG V-cycle (SPD) for PCG
    #     """
    #     x = np.zeros_like(rhs)

    #     # Pre-smoothing (forward GS)
    #     if self.pre_smooth > 0:
    #         x = block_gauss_seidel_6dof(self.A, rhs, x0=x,
    #                                     num_iter=self.pre_smooth)

    #     # Coarse-grid correction
    #     r = rhs - self.A @ x
    #     rc = self.R @ r
    #     ec = self.coarse_solver.solve(rc)
    #     x += self.P @ ec

    #     # Post-smoothing (backward GS, applied to x!)
    #     if self.post_smooth > 0:
    #         x = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x,
    #                                             num_iter=self.post_smooth)

    #     return x

    def solve(self, rhs):
        """
        Fully symmetric AMG V-cycle (SPD) for PCG
        """
        x = np.zeros_like(rhs)

        self.iters += self.ncyc

        for _ in range(self.ncyc):

            # Pre-smoothing (forward GS)
            if self.pre_smooth > 0:
                x = block_gauss_seidel_6dof(self.A, rhs, x0=x,
                                            num_iter=self.pre_smooth)

            # # Coarse-grid correction
            r = rhs - self.A.dot(x)
            rc = self.R.dot(r)
            ec = self.coarse_solver.solve(rc)
            xc = self.P.dot(ec)

            # now do line-search update of this
            # fc = self.A.dot(xc)
            # omega_LS = np.dot(rhs, xc) / np.dot(xc, fc)
            # print(f"{omega_LS=}")
            # x += omega_LS * xc
            # just results in omega_LS = 1 from AMG
            x += xc

            # Post-smoothing (backward GS, applied to x!)
            if self.post_smooth > 0:
                x = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x,
                                                num_iter=self.post_smooth)

        return x
    
    @property
    def total_vcycles(self) -> int:
        return self.iters
    


def build_asw_block_inverses_bsr(A: sp.bsr_matrix, asw_sd_size: int):
    """
    Build dense block-Jacobi inverses from contiguous diagonal blocks of A.

    Parameters
    ----------
    A : bsr_matrix
        Square system matrix on this level.
    asw_sd_size : int
        Block size in scalar DOFs.

    Returns
    -------
    block_inv : list[np.ndarray]
        Dense inverses of each diagonal block.
    block_ranges : list[(int,int)]
        Half-open DOF index ranges (start, end) for each block.
    """
    assert sp.isspmatrix_bsr(A)
    assert A.shape[0] == A.shape[1]
    assert asw_sd_size >= 1

    A_csr = A.tocsr(copy=True)

    n = A.shape[0]
    block_inv = []
    block_ranges = []

    start = 0
    while start < n:
        end = min(start + asw_sd_size, n)

        Ablock = A_csr[start:end, start:end].toarray()
        block_inv.append(np.linalg.inv(Ablock))
        block_ranges.append((start, end))

        start = end

    return block_inv, block_ranges


def apply_asw_blocks_bsr(rhs: np.ndarray,
                         block_inv: list[np.ndarray],
                         block_ranges: list[tuple[int, int]]) -> np.ndarray:
    """
    Apply stored block-Jacobi inverse:
        z = D^{-1} rhs
    where D is block diagonal with dense diagonal blocks.
    """
    z = np.zeros_like(rhs)

    for Binv, (s, e) in zip(block_inv, block_ranges):
        z[s:e] = Binv @ rhs[s:e]

    return z
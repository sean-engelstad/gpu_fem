import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# you already have these in your BSR library
from _smoothers import block_gauss_seidel_6dof, block_gauss_seidel_6dof_transpose


# ----------------------------------------------------------------------
# basic BSR graph / BC helpers
# ----------------------------------------------------------------------

def strength_matrix_bsr(A: sp.bsr_matrix, threshold: float = 0.25):
    """
    Strength-of-connection graph on BSR block rows/nodes.

    Returns
    -------
    STRENGTH : list[list[int]]
        STRENGTH[i] contains strong block-neighbors of node i.
    """
    assert sp.isspmatrix_bsr(A)

    b = A.blocksize[0]
    nnodes = A.shape[0] // b

    diag_nrms = np.zeros(nnodes, dtype=np.double)
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[jp]
            if i == j:
                diag_nrms[i] = np.linalg.norm(A.data[jp])
                break

    STRENGTH = [[] for _ in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[jp]
            nrm = np.linalg.norm(A.data[jp])
            lb = threshold * np.sqrt(diag_nrms[i] * diag_nrms[j])
            if nrm >= lb:
                STRENGTH[i].append(j)

    return STRENGTH


def get_bc_flags_bsr(A: sp.bsr_matrix, tol: float = 1e-14) -> np.ndarray:
    """
    Detect constrained scalar DOFs from a BSR matrix with identity Dirichlet rows.
    """
    assert sp.isspmatrix_bsr(A)

    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    flags = np.zeros(A.shape[0], dtype=bool)

    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i + 1]
        idx = A.indices[start:end]
        blk = A.data[start:end]

        diag = blk[idx == i]
        off = blk[idx != i]

        flags[i * b:(i + 1) * b] = (
            np.all(np.abs(diag - np.eye(b)) < tol) and
            (off.size == 0 or np.all(np.abs(off) < tol))
        )

    return flags


def spectral_radius_block_DinvA_bsr(
    A: sp.bsr_matrix,
    maxiter: int = 200,
    tol: float = 1e-8,
):
    """
    Estimate spectral radius of block Jacobi operator D_b^{-1} A.
    """
    if not sp.isspmatrix_bsr(A):
        raise TypeError("A must be a BSR matrix")

    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    n = A.shape[0]

    Dinv_blocks = np.zeros((nblocks, b, b), dtype=A.data.dtype)
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i + 1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        Dinv_blocks[i] = np.linalg.inv(A.data[start + diag_idx[0]])

    def matvec(x):
        y = A @ x
        y = y.reshape(nblocks, b)
        for i in range(nblocks):
            y[i] = Dinv_blocks[i] @ y[i]
        return y.reshape(n)

    M = spla.LinearOperator(shape=(n, n), matvec=matvec, dtype=A.dtype)
    eigval = spla.eigs(
        M, k=1, which="LM", maxiter=maxiter, tol=tol, return_eigenvectors=False
    )
    return np.abs(eigval[0])


# ----------------------------------------------------------------------
# root-node BSR aggregation
# ----------------------------------------------------------------------

def greedy_rn_serial_aggregation_bsr(
    A: sp.bsr_matrix,
    C_nodes: np.ndarray,
    threshold: float = 0.25,
):
    """
    Greedy root-node serial aggregation for BSR.

    Unassigned nodes are first added only to aggregates whose ROOT NODE
    is a strong neighbor of that node. Remaining nodes are swept into
    neighboring already-formed aggregates.
    """
    assert sp.isspmatrix_bsr(A)

    b = A.blocksize[0]
    nnodes = A.shape[0] // b
    assert C_nodes.shape == (nnodes,)
    assert C_nodes.dtype == bool

    STRENGTH = strength_matrix_bsr(A, threshold)

    aggregate_ind = np.full(nnodes, -1, dtype=np.int32)
    aggregate_groups = []

    C_node_list = [i for i in range(nnodes) if C_nodes[i]]
    root_of_agg = []

    for iagg, root_node in enumerate(C_node_list):
        aggregate_groups.append([root_node])
        aggregate_ind[root_node] = iagg
        root_of_agg.append(root_node)

    # pass 1: only assign to aggregates whose root is a strong neighbor
    for i in range(nnodes):
        if aggregate_ind[i] != -1:
            continue

        strong_neighbors = np.array(STRENGTH[i], dtype=np.int32)

        candidate_aggs = []
        for iagg, root_node in enumerate(root_of_agg):
            if root_node in strong_neighbors:
                candidate_aggs.append(iagg)

        if len(candidate_aggs) == 0:
            continue

        candidate_sizes = np.array(
            [len(aggregate_groups[iagg]) for iagg in candidate_aggs],
            dtype=np.int32,
        )
        agg_ind = candidate_aggs[np.argmin(candidate_sizes)]

        aggregate_groups[agg_ind].append(i)
        aggregate_ind[i] = agg_ind

    # pass 2: sweep remaining nodes into neighboring assigned aggregates
    for i in range(nnodes):
        if aggregate_ind[i] != -1:
            continue

        strong_neighbors = np.array(STRENGTH[i], dtype=np.int32)
        if strong_neighbors.size == 0:
            continue

        nb_agg_ind = aggregate_ind[strong_neighbors]
        valid_nb_agg_ind = nb_agg_ind[nb_agg_ind != -1]

        if valid_nb_agg_ind.size == 0:
            continue

        valid_nb_agg_ind = np.unique(valid_nb_agg_ind)
        nb_agg_sizes = np.array(
            [len(aggregate_groups[ind]) for ind in valid_nb_agg_ind],
            dtype=np.int32,
        )
        agg_ind = valid_nb_agg_ind[np.argmin(nb_agg_sizes)]

        aggregate_groups[agg_ind].append(i)
        aggregate_ind[i] = agg_ind

    return aggregate_ind


# ----------------------------------------------------------------------
# root-node BSR tentative prolongation + near-kernel smoothing
# ----------------------------------------------------------------------

def block_diag_inv_bsr(A: sp.bsr_matrix) -> np.ndarray:
    """
    Return dense inverses of the BSR diagonal blocks.
    """
    assert sp.isspmatrix_bsr(A)
    b = A.blocksize[0]
    nblocks = A.shape[0] // b

    Dinv_blocks = np.zeros((nblocks, b, b), dtype=A.data.dtype)
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i + 1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        Dinv_blocks[i] = np.linalg.inv(A.data[start + diag_idx[0]])
    return Dinv_blocks


def apply_block_Dinv_to_dense_candidates(
    Dinv_blocks: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Apply block diagonal inverse to dense candidate matrix.

    Parameters
    ----------
    Dinv_blocks : (nnodes, b, b)
    X : (nnodes, b, nmodes)

    Returns
    -------
    Y : (nnodes, b, nmodes)
    """
    nnodes, b, nmodes = X.shape
    Y = np.zeros_like(X)
    for i in range(nnodes):
        Y[i] = Dinv_blocks[i] @ X[i]
    return Y


def energy_smooth_candidates_bsr(
    B: np.ndarray,
    A: sp.bsr_matrix,
    omega: float = 0.7,
    nsmooth: int = 1,
):
    """
    Pure block-Jacobi energy smoothing of dense fine-grid candidate blocks.

    Parameters
    ----------
    B : (nnodes, b, nmodes) ndarray
    A : bsr_matrix of shape (nnodes*b, nnodes*b)

    Returns
    -------
    Bnew : (nnodes, b, nmodes) ndarray
    """
    assert sp.isspmatrix_bsr(A)
    assert B.ndim == 3

    b = A.blocksize[0]
    nnodes = A.shape[0] // b
    assert B.shape[0] == nnodes
    assert B.shape[1] == b

    Dinv_blocks = block_diag_inv_bsr(A)

    nmodes = B.shape[2]
    Bnew = B.copy()

    for _ in range(nsmooth):
        Bflat = Bnew.reshape(nnodes * b, nmodes)
        ABflat = A @ Bflat
        AB = ABflat.reshape(nnodes, b, nmodes)
        DinvAB = apply_block_Dinv_to_dense_candidates(Dinv_blocks, AB)
        Bnew -= omega * DinvAB

    return Bnew


def tentative_rootnode_prolongator_bsr(
    aggregate_ind: np.ndarray,
    C_nodes: np.ndarray,
    B: np.ndarray,
    Bc: np.ndarray = None,
    rcond: float = 1e-12,
):
    """
    Construct a BSR root-node tentative prolongator.

    Parameters
    ----------
    aggregate_ind : (nnodes,) int
    C_nodes : (nnodes,) bool
    B : (nnodes, b, nmodes)
        Fine candidate blocks
    Bc : (num_agg, nmodes, nmodes) or None
        Coarse candidate blocks. If None, use B at the aggregate root node.

    Returns
    -------
    T : bsr_matrix, shape (nnodes*b, num_agg*nmodes)
    Bc : (num_agg, nmodes, nmodes)
    """
    nnodes = aggregate_ind.shape[0]
    assert C_nodes.shape == (nnodes,)
    assert C_nodes.dtype == bool
    assert B.ndim == 3

    b = B.shape[1]
    nmodes = B.shape[2]
    num_agg = np.max(aggregate_ind) + 1

    # for this version we take nmodes == blocksize, like your existing BSR SA code
    if nmodes != b:
        raise ValueError(
            f"Current root-node BSR version assumes nmodes == blocksize, got {nmodes=} and {b=}"
        )

    agg_root = np.full(num_agg, -1, dtype=np.int32)
    for inode in range(nnodes):
        if C_nodes[inode]:
            iagg = aggregate_ind[inode]
            if agg_root[iagg] != -1:
                raise ValueError(f"Aggregate {iagg} has more than one root/C-node")
            agg_root[iagg] = inode

    if np.any(agg_root < 0):
        missing = np.where(agg_root < 0)[0]
        raise ValueError(f"Missing root/C-node in aggregates {missing}")

    if Bc is None:
        Bc = B[agg_root].copy()
    else:
        Bc = np.asarray(Bc, dtype=B.dtype)
        if Bc.shape != (num_agg, nmodes, nmodes):
            raise ValueError(
                f"Bc must have shape ({num_agg}, {nmodes}, {nmodes}), got {Bc.shape}"
            )

    rowp = np.arange(nnodes + 1, dtype=np.int32)
    cols = aggregate_ind.astype(np.int32).copy()
    data = np.zeros((nnodes, b, nmodes), dtype=np.double)

    for inode in range(nnodes):
        iagg = aggregate_ind[inode]
        if C_nodes[inode]:
            # exact root injection
            data[inode] = np.eye(b, nmodes)
        else:
            # minimum-Frobenius-norm solve: Ti * Bc[a] ~= B[i]
            data[inode] = B[inode] @ np.linalg.pinv(Bc[iagg], rcond=rcond)

    T = sp.bsr_matrix((data, cols, rowp), shape=(nnodes * b, num_agg * nmodes))
    return T, Bc


def orthog_nullspace_projector_bsr_rootnode(
    P: sp.bsr_matrix,
    Bc: np.ndarray,
):
    """
    Row-wise orthogonal projector for a BSR prolongation/update.

    Ensures each block row satisfies
        sum_j P_ij Bc_j = 0
    in the local least-squares sense.
    """
    assert sp.isspmatrix_bsr(P)
    assert Bc.ndim == 3

    Pnew = sp.bsr_matrix(
        (P.data.copy(), P.indices.copy(), P.indptr.copy()),
        shape=P.shape,
        blocksize=P.blocksize,
    )

    nblock_rows = P.shape[0] // P.blocksize[0]

    for brow in range(nblock_rows):
        start = P.indptr[brow]
        end = P.indptr[brow + 1]
        if start == end:
            continue

        bcols = P.indices[start:end]
        ncols = len(bcols)

        U_list = [Bc[bcol] for bcol in bcols]

        PU = sum(P.data[start + j] @ U_list[j] for j in range(ncols))
        UTU = sum(U_list[j].T @ U_list[j] for j in range(ncols))
        UTU_inv = np.linalg.pinv(UTU)

        for j in range(ncols):
            Pnew.data[start + j] -= PU @ UTU_inv @ U_list[j].T

    return Pnew


def enforce_bsr_sparsity_mask(X: sp.bsr_matrix, mask: sp.bsr_matrix):
    """
    Enforce fixed sparsity pattern from `mask` on BSR matrix X.
    """
    if not sp.isspmatrix_bsr(X):
        X = X.tobsr(blocksize=mask.blocksize)
    if not sp.isspmatrix_bsr(mask):
        mask = mask.tobsr(blocksize=X.blocksize)

    Xcsr = X.tocsr()
    Mcsr = mask.tocsr()
    Ycsr = Xcsr.multiply(Mcsr != 0)
    Ycsr.sum_duplicates()
    Ycsr.eliminate_zeros()
    return Ycsr.tobsr(blocksize=mask.blocksize)

def enforce_rootnode_injection_bsr(
    P: sp.bsr_matrix,
    T: sp.bsr_matrix,
    C_nodes: np.ndarray,
):
    """
    Replace root-node block rows in P with the original injection block rows from T.

    Assumes:
      - P and T have the same shape and blocksize
      - each root/C-node row in T has exactly one block nonzero
    """
    assert sp.isspmatrix_bsr(P)
    assert sp.isspmatrix_bsr(T)
    assert P.shape == T.shape
    assert P.blocksize == T.blocksize

    nblock_rows = P.shape[0] // P.blocksize[0]
    assert C_nodes.shape == (nblock_rows,)

    new_indptr = np.zeros(nblock_rows + 1, dtype=np.int32)
    new_indices = []
    new_data = []

    for i in range(nblock_rows):
        if C_nodes[i]:
            start = T.indptr[i]
            end = T.indptr[i + 1]

            if end - start != 1:
                raise ValueError(
                    f"Expected exactly one block nonzero in root row {i}"
                )

            new_indices.append(int(T.indices[start]))
            new_data.append(T.data[start].copy())
            new_indptr[i + 1] = new_indptr[i] + 1
        else:
            start = P.indptr[i]
            end = P.indptr[i + 1]

            row_nnz = end - start
            new_indices.extend(P.indices[start:end].tolist())
            new_data.extend([blk.copy() for blk in P.data[start:end]])
            new_indptr[i + 1] = new_indptr[i] + row_nnz

    if len(new_data) == 0:
        data = np.zeros((0, P.blocksize[0], P.blocksize[1]), dtype=P.data.dtype)
        indices = np.zeros((0,), dtype=np.int32)
    else:
        data = np.asarray(new_data, dtype=P.data.dtype)
        indices = np.asarray(new_indices, dtype=np.int32)

    return sp.bsr_matrix((data, indices, new_indptr), shape=P.shape)

def smooth_prolongator_bsr_rootnode(
    T: sp.bsr_matrix,
    A: sp.bsr_matrix,
    Bc: np.ndarray,
    C_nodes: np.ndarray,
    omega: float = 0.7,
    near_kernel: bool = True,
    nsmooth: int = 1,
    mask: sp.bsr_matrix = None,
):
    """
    Multi-step block-Jacobi smoothing of tentative prolongator for root-node AMG.

    - fixed sparsity pattern defaults to A @ T
    - root-node rows are restored to exact injection after each step
    """
    assert sp.isspmatrix_bsr(T)
    assert sp.isspmatrix_bsr(A)
    assert A.blocksize[0] == T.blocksize[0]

    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    Dinv_blocks = block_diag_inv_bsr(A)

    if mask is None:
        mask = (A @ T).tobsr(blocksize=T.blocksize)
        mask.eliminate_zeros()

    P = T.copy()

    for _ in range(nsmooth):
        AP = (A @ P).tobsr(blocksize=T.blocksize)
        AP = enforce_bsr_sparsity_mask(AP, mask)

        if near_kernel:
            AP = orthog_nullspace_projector_bsr_rootnode(AP, Bc)

        dP = AP.copy()
        for i in range(nblocks):
            start, end = dP.indptr[i], dP.indptr[i + 1]
            dP.data[start:end] = Dinv_blocks[i] @ dP.data[start:end]

        P = (P - omega * dP).tobsr(blocksize=T.blocksize)
        P = enforce_rootnode_injection_bsr(P, T, C_nodes)

    P.sum_duplicates()
    P.eliminate_zeros()
    return P


# ----------------------------------------------------------------------
# coarse-grid direct solver
# ----------------------------------------------------------------------

class DirectBSRSolver:
    def __init__(self, A_bsr: sp.bsr_matrix):
        self.A = A_bsr.tocsc()

    def solve(self, rhs: np.ndarray):
        return spla.spsolve(self.A, rhs)


# ----------------------------------------------------------------------
# root-node BSR AMG solver
# ----------------------------------------------------------------------

class RootNodeAMG_BSRSolver:
    """
    Multilevel BSR root-node AMG solver.

    This follows your CSR root-node structure, but uses BSR candidates and
    BSR prolongation blocks.
    """
    def __init__(
        self,
        A_free: sp.bsr_matrix,
        A: sp.bsr_matrix,
        B: np.ndarray,
        threshold: float = 0.25,
        omega: float = 0.7,
        nsmooth: int = 1,
        level: int = 0,
        near_kernel: bool = True,
        coarsening_fcn=None,
        aggregation_fcn=greedy_rn_serial_aggregation_bsr,
        rbm_omega: float = 0.5,
        rbm_nsmooth: int = 2,
        prol_nsmooth: int = 1,
        smoother: str = "gs",
        asw_overlap: int = 0,
        omegaSmooth: float = 0.7,
        asw_sd_size: int = None,
        verbose: bool = True,
    ):
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)
        assert B.ndim == 3

        smoother = smoother.lower()
        if smoother not in ("gs", "asw"):
            raise ValueError(f"Unknown smoother '{smoother}'")
        if smoother == "asw" and asw_sd_size is None:
            raise ValueError("Must provide asw_sd_size when smoother='asw'")

        if coarsening_fcn is None:
            raise ValueError(
                "Please pass a BSR coarsening_fcn returning (C_nodes, F_nodes)"
            )

        self.A_free = A_free
        self.A = A
        self.B_input = B
        self.threshold = threshold
        self.omega = omega
        self.nsmooth = nsmooth
        self.level = level
        self.near_kernel = near_kernel
        self.verbose = verbose
        self.smoother = smoother
        self.omegaSmooth = omegaSmooth
        self.asw_sd_size = asw_sd_size
        self.iters = 0

        b = A.blocksize[0]
        self.block_dim = b
        self.nfine_nodes = A.shape[0] // b
        self.fine_nnz = A.nnz

        # 1) coarse/fine split on block graph
        self.C_nodes, self.F_nodes = coarsening_fcn(A_free, threshold=threshold)

        # 2) root-node aggregation
        self.aggregate_ind = aggregation_fcn(A_free, self.C_nodes, threshold=threshold)
        self.num_agg = np.max(self.aggregate_ind) + 1

        # 3) smooth fine-grid candidates with block Jacobi
        self.B = energy_smooth_candidates_bsr(
            B, A, omega=rbm_omega, nsmooth=rbm_nsmooth
        )

        # 4) tentative prolongator + coarse candidates
        self.T, self.Bc = tentative_rootnode_prolongator_bsr(
            self.aggregate_ind, self.C_nodes, self.B
        )

        if verbose:
            B_flat = self.B.reshape(self.nfine_nodes * b, b)
            Bc_flat = self.Bc.reshape(self.num_agg * b, b)
            resid = self.T @ Bc_flat - B_flat
            print(
                f"level {level}: num_agg={self.num_agg}, "
                f"||T Bc - B||_2 = {np.linalg.norm(resid):.4e}, "
                f"||T Bc - B||_inf = {np.linalg.norm(resid, ord=np.inf):.4e}"
            )

        if verbose and near_kernel:
            AT = (A @ self.T).tobsr(blocksize=self.T.blocksize)
            U = orthog_nullspace_projector_bsr_rootnode(AT, self.Bc)
            Bc_flat = self.Bc.reshape(self.num_agg * b, b)
            proj_resid = U @ Bc_flat
            print(
                f"level {level}: ||Proj(A T) Bc||_2 = {np.linalg.norm(proj_resid):.4e}, "
                f"||Proj(A T) Bc||_inf = {np.linalg.norm(proj_resid, ord=np.inf):.4e}"
            )

        # 5) smooth prolongator
        self.P = smooth_prolongator_bsr_rootnode(
            self.T,
            A,
            self.Bc,
            self.C_nodes,
            omega=omega,
            near_kernel=near_kernel,
            nsmooth=prol_nsmooth,
        )
        self.R = self.P.T.tobsr(blocksize=(b, b))

        if verbose:
            B_flat = self.B.reshape(self.nfine_nodes * b, b)
            Bc_flat = self.Bc.reshape(self.num_agg * b, b)
            resid_post = self.P @ Bc_flat - B_flat
            print(
                f"level {level}: ||P Bc - B||_2 = {np.linalg.norm(resid_post):.4e}, "
                f"||P Bc - B||_inf = {np.linalg.norm(resid_post, ord=np.inf):.4e}"
            )

        # 6) Galerkin operators
        self.Ac = (self.R @ (A @ self.P)).tobsr()
        self.Ac_free = (self.R @ (A_free @ self.P)).tobsr()

        self.coarse_nnz = self.Ac.nnz
        self.ncoarse = self.Ac.shape[0]

        # optional ASW data
        self.asw_block_inv = None
        self.asw_block_ranges = None
        if self.smoother == "asw":
            self.asw_block_inv, self.asw_block_ranges = build_asw_block_inverses_bsr(
                self.A, self.asw_sd_size, overlap=asw_overlap
            )

        # 7) recurse or direct solve
        max_coarse_nnz = self.ncoarse ** 2

        if level == 0 and verbose:
            print("level 0 is root-node BSR AMG solver..")

        if self.coarse_nnz >= 0.4 * max_coarse_nnz or self.ncoarse <= 150:
            if verbose:
                print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectBSRSolver(self.Ac)
        else:
            if verbose:
                print(f"level {level+1} building AMG solver")
            self.coarse_solver = RootNodeAMG_BSRSolver(
                self.Ac_free,
                self.Ac,
                self.Bc,
                threshold=threshold,
                omega=omega,
                nsmooth=nsmooth,
                level=level + 1,
                near_kernel=near_kernel,
                coarsening_fcn=coarsening_fcn,
                aggregation_fcn=aggregation_fcn,
                rbm_omega=rbm_omega,
                rbm_nsmooth=rbm_nsmooth,
                prol_nsmooth=prol_nsmooth,
                smoother=smoother,
                asw_overlap=asw_overlap,
                omegaSmooth=omegaSmooth,
                asw_sd_size=asw_sd_size,
                verbose=verbose,
            )

        if level == 0 and verbose:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=:.4f}")
            print(f"\tnum nodes per level = {self.num_nodes_list}")

    def _smooth_gs(self, rhs: np.ndarray, x: np.ndarray, transpose: bool = False) -> np.ndarray:
        if transpose:
            x_gs = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x, num_iter=1)
        else:
            x_gs = block_gauss_seidel_6dof(self.A, rhs, x0=x, num_iter=1)
        return x + self.omegaSmooth * (x_gs - x)

    def _smooth_asw(self, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
        r = rhs - self.A.dot(x)
        z = apply_asw_blocks_bsr(r, self.asw_block_inv, self.asw_block_ranges)
        return x + self.omegaSmooth * z

    def _apply_smoother(
        self,
        rhs: np.ndarray,
        x: np.ndarray,
        nsweeps: int,
        transpose: bool = False,
    ) -> np.ndarray:
        if nsweeps <= 0:
            return x

        if self.smoother == "gs":
            for _ in range(nsweeps):
                x = self._smooth_gs(rhs, x, transpose=transpose)
            return x

        if self.smoother == "asw":
            for _ in range(nsweeps):
                x = self._smooth_asw(rhs, x)
            return x

        raise RuntimeError(f"Unsupported smoother '{self.smoother}'")

    def solve(self, rhs: np.ndarray):
        x = np.zeros_like(rhs)
        self.iters += 1

        x = self._apply_smoother(rhs, x, self.nsmooth, transpose=False)

        r = rhs - self.A.dot(x)
        rc = self.R.dot(r)
        ec = self.coarse_solver.solve(rc)
        x += self.P.dot(ec)

        x = self._apply_smoother(rhs, x, self.nsmooth, transpose=True)
        return x

    @property
    def total_nnz(self) -> int:
        if isinstance(self.coarse_solver, RootNodeAMG_BSRSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, RootNodeAMG_BSRSolver):
            return 1 + self.coarse_solver.num_levels
        return 2

    @property
    def num_nodes_list(self):
        if isinstance(self.coarse_solver, RootNodeAMG_BSRSolver):
            return [self.A.shape[0]] + self.coarse_solver.num_nodes_list
        return [self.A.shape[0], self.Ac.shape[0]]
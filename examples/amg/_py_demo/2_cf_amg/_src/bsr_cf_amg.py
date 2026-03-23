

import numpy as np
import scipy.sparse as sp

from _smoothers import block_gauss_seidel_6dof, block_gauss_seidel_6dof_transpose

import numpy as np
import scipy.sparse as sp


def _check_bsr(A_bsr: sp.bsr_matrix):
    if not sp.isspmatrix_bsr(A_bsr):
        raise TypeError("A_bsr must be a scipy.sparse.bsr_matrix")
    br, bc = A_bsr.blocksize
    if br != bc:
        raise ValueError("Only square BSR blocks are supported")
    if A_bsr.shape[0] % br != 0 or A_bsr.shape[1] % bc != 0:
        raise ValueError("Matrix shape must be divisible by block size")
    nbrows = A_bsr.shape[0] // br
    nbcols = A_bsr.shape[1] // bc
    # if nbrows != nbcols:
    #     raise ValueError("A_bsr must be square in block sense")
    return br, nbrows


def _block_nodes_to_dofs(nodes, bs):
    """
    Expand block-node indices into scalar DOF indices.
    Example: nodes=[2,5], bs=3 -> [6,7,8,15,16,17]
    """
    nodes = np.asarray(nodes, dtype=int)
    return (nodes[:, None] * bs + np.arange(bs)[None, :]).ravel()


def _submatrix_bsr(A_bsr: sp.bsr_matrix, row_blocks, col_blocks):
    """
    Extract a block submatrix using block-node indices.

    Safe version:
      BSR -> CSR -> scalar slicing -> BSR
    """
    bs, _ = _check_bsr(A_bsr)

    row_blocks = np.asarray(row_blocks, dtype=int)
    col_blocks = np.asarray(col_blocks, dtype=int)

    row_dofs = _block_nodes_to_dofs(row_blocks, bs)
    col_dofs = _block_nodes_to_dofs(col_blocks, bs)

    A_csr = A_bsr.tocsr()
    A_sub_csr = A_csr[row_dofs, :][:, col_dofs]

    return A_sub_csr.tobsr(blocksize=(bs, bs))


def build_initial_FC_bsr_matrix(A_bsr: sp.bsr_matrix, C_nodes, F_nodes):
    """
    Block analogue of your build_initial_FC_matrix().
    Returns A_FC on the block graph.
    """
    return _submatrix_bsr(A_bsr, F_nodes, C_nodes)


def _block_row_sum(A_bsr: sp.bsr_matrix):
    """
    For each block row i, compute the dense bs x bs block sum:
        D_i = sum_j A_ij
    Returns array of shape (n_block_rows, bs, bs).
    """
    bs, nb = _check_bsr(A_bsr)
    D = np.zeros((nb, bs, bs), dtype=A_bsr.data.dtype)

    for i in range(nb):
        start, end = A_bsr.indptr[i], A_bsr.indptr[i + 1]
        if end > start:
            D[i] = np.sum(A_bsr.data[start:end], axis=0)

    return D


def _invert_blocks(D_blocks, rcond=1e-12):
    """
    Invert each dense block. Falls back to pinv if singular.
    """
    n, bs, _ = D_blocks.shape
    Dinv = np.zeros_like(D_blocks)

    for i in range(n):
        try:
            Dinv[i] = np.linalg.inv(D_blocks[i])
        except np.linalg.LinAlgError:
            Dinv[i] = np.linalg.pinv(D_blocks[i], rcond=rcond)

    return Dinv


def _left_scale_block_rows(A_bsr: sp.bsr_matrix, L_blocks):
    """
    Compute B = L * A where L is block diagonal with dense blocks L_blocks[i].
    That is, each block row i gets left-multiplied by L_blocks[i].
    """
    bs, nb = _check_bsr(A_bsr)
    if L_blocks.shape != (nb, bs, bs):
        raise ValueError("L_blocks has wrong shape")

    data = A_bsr.data.copy()
    for i in range(nb):
        start, end = A_bsr.indptr[i], A_bsr.indptr[i + 1]
        for k in range(start, end):
            data[k] = L_blocks[i] @ data[k]

    return sp.bsr_matrix((data, A_bsr.indices.copy(), A_bsr.indptr.copy()),
                         shape=A_bsr.shape, blocksize=A_bsr.blocksize)

def repair_bsr_CF_splitting(A_bsr: sp.bsr_matrix, C_nodes, F_nodes):
    """
    Promote any fine block node with no coarse-neighbor connection in A_FC
    to a coarse node.

    Returns
    -------
    C_nodes_new, F_nodes_new
    """
    bs, n_block_rows = _check_bsr(A_bsr)

    C_nodes = np.asarray(C_nodes, dtype=int)
    F_nodes = np.asarray(F_nodes, dtype=int)

    A_FC = _submatrix_bsr(A_bsr, F_nodes, C_nodes)

    keep_F = []
    add_C = []

    for lf, node in enumerate(F_nodes):
        start, end = A_FC.indptr[lf], A_FC.indptr[lf + 1]
        if end == start:
            add_C.append(node)
        else:
            keep_F.append(node)

    if len(add_C) > 0:
        C_nodes = np.unique(np.concatenate([C_nodes, np.array(add_C, dtype=int)]))
    F_nodes = np.array(keep_F, dtype=int)

    return C_nodes, F_nodes

def _assemble_bsr_prolongation(W_bsr: sp.bsr_matrix, C_nodes, F_nodes, n_block_rows, bs):
    """
    Assemble full block prolongation P from:
      - identity on coarse rows
      - W on fine rows
    Columns correspond to coarse block nodes in the given C_nodes ordering.
    """
    nc = len(C_nodes)
    coarse_col_of_node = -np.ones(n_block_rows, dtype=int)
    coarse_col_of_node[np.asarray(C_nodes, dtype=int)] = np.arange(nc)

    F_nodes = np.asarray(F_nodes, dtype=int)
    C_nodes = np.asarray(C_nodes, dtype=int)

    data_blocks = []
    indices = []
    indptr = [0]

    # map global fine block row -> local row in W
    fine_local = -np.ones(n_block_rows, dtype=int)
    fine_local[F_nodes] = np.arange(len(F_nodes))

    Ibs = np.eye(bs, dtype=W_bsr.data.dtype if W_bsr.nnz > 0 else float)

    for i in range(n_block_rows):
        if coarse_col_of_node[i] >= 0:
            # coarse row: injection
            indices.append(coarse_col_of_node[i])
            data_blocks.append(Ibs.copy())
        else:
            # fine row: use W row
            lf = fine_local[i]
            if lf < 0:
                raise RuntimeError(f"Block row {i} is neither coarse nor fine")
            start, end = W_bsr.indptr[lf], W_bsr.indptr[lf + 1]
            indices.extend(W_bsr.indices[start:end].tolist())
            data_blocks.extend(W_bsr.data[start:end])

        indptr.append(len(indices))

    data_blocks = np.asarray(data_blocks)
    indices = np.asarray(indices, dtype=int)
    indptr = np.asarray(indptr, dtype=int)

    return sp.bsr_matrix((data_blocks, indices, indptr),
                         shape=(n_block_rows * bs, nc * bs),
                         blocksize=(bs, bs))


def direct_bsr_interpolation(A_bsr: sp.bsr_matrix, C_nodes, F_nodes):
    """
    BSR/block version of your direct CSR interpolation.

    Parameters
    ----------
    A_bsr : sp.bsr_matrix
        System matrix with block AMG nodes.
    C_nodes, F_nodes : array-like
        Coarse/fine block-node indices.

    Returns
    -------
    P_bsr : sp.bsr_matrix
        Block prolongation of shape (n_block*bs, nC*bs).
    """
    bs, n_block_rows = _check_bsr(A_bsr)

    A_FC = build_initial_FC_bsr_matrix(A_bsr, C_nodes, F_nodes)

    # D_i = sum_j A_ij over coarse neighbors j
    D_FF = _block_row_sum(A_FC)
    Dinv_FF = _invert_blocks(D_FF)

    # W = D^{-1} A_FC
    W = _left_scale_block_rows(A_FC, Dinv_FF)

    # Full prolongation
    P_bsr = _assemble_bsr_prolongation(W, C_nodes, F_nodes, n_block_rows, bs)
    return P_bsr

def _extract_block_diag(A_bsr: sp.bsr_matrix):
    """
    Extract block diagonal from a square BSR matrix.

    Returns
    -------
    D : ndarray of shape (nb, bs, bs)
        D[i] is the diagonal block A[i,i].
    """
    if not sp.isspmatrix_bsr(A_bsr):
        raise TypeError("A_bsr must be BSR")

    bs = A_bsr.blocksize[0]
    nb = A_bsr.shape[0] // bs

    D = np.zeros((nb, bs, bs), dtype=A_bsr.data.dtype)

    for i in range(nb):
        start = A_bsr.indptr[i]
        end = A_bsr.indptr[i + 1]
        cols = A_bsr.indices[start:end]

        hit = np.where(cols == i)[0]
        if hit.size == 0:
            raise RuntimeError(f"Missing diagonal block in row {i}")
        D[i, :, :] = A_bsr.data[start + hit[0]]

    return D

# def standard_bsr_interpolation(A_bsr: sp.bsr_matrix, C_nodes, F_nodes):
#     """
#     BSR/block version of your standard CSR interpolation.

#     This mirrors:
#         W1 = D^{-1} A_FC
#         W  = W1 - D^{-1} A_FF W1
#         W2 = D2^{-1} W
#     but in block form.

#     Returns
#     -------
#     P_bsr : sp.bsr_matrix
#         Block prolongation of shape (n_block*bs, nC*bs).
#     """
#     bs, n_block_rows = _check_bsr(A_bsr)

#     A_FC = _submatrix_bsr(A_bsr, F_nodes, C_nodes)
#     A_FF = _submatrix_bsr(A_bsr, F_nodes, F_nodes)

#     # First get direct interpolation
#     D_FF = _block_row_sum(A_FC)
#     Dinv_FF = _invert_blocks(D_FF)
#     W1 = _left_scale_block_rows(A_FC, Dinv_FF)

#     # Then standard interpolation
#     W = (W1 - _left_scale_block_rows(A_FF @ W1, Dinv_FF)).tobsr(blocksize=(bs, bs))

#     # Rescale again to preserve blockwise constants
#     D2_FF = _block_row_sum(W)
#     Dinv2_FF = _invert_blocks(D2_FF)
#     W2 = _left_scale_block_rows(W, Dinv2_FF)

#     # Full prolongation
#     P_bsr = _assemble_bsr_prolongation(W2, C_nodes, F_nodes, n_block_rows, bs)
#     return P_bsr


def standard_bsr_interpolation(A_bsr: sp.bsr_matrix, C_nodes, F_nodes):
    """
    Block version of standard / approximate ideal interpolation using
    two block-Jacobi sweeps on

        A_FF W = -A_FC

    followed by optional block row normalization.
    """
    bs, n_block_rows = _check_bsr(A_bsr)

    A_FC = _submatrix_bsr(A_bsr, F_nodes, C_nodes)
    A_FF = _submatrix_bsr(A_bsr, F_nodes, F_nodes)

    # block diagonal of A_FF, shape (nF, bs, bs)
    D_FF = _extract_block_diag(A_FF)
    Dinv_FF = _invert_blocks(D_FF)

    # sweep 1: W1 = -D^{-1} A_FC
    W1 = -_left_scale_block_rows(A_FC, Dinv_FF)

    # sweep 2: W2 = W1 + D^{-1}(-A_FC - A_FF W1)
    residual = (-A_FC - A_FF @ W1).tobsr(blocksize=(bs, bs))
    W2 = (W1 + _left_scale_block_rows(residual, Dinv_FF)).tobsr(blocksize=(bs, bs))

    # optional block row normalization to preserve blockwise constants / RBMs
    D2_FF = _block_row_sum(W2)
    Dinv2_FF = _invert_blocks(D2_FF)
    W = _left_scale_block_rows(W2, Dinv2_FF).tobsr(blocksize=(bs, bs))

    # assemble full prolongation
    P_bsr = _assemble_bsr_prolongation(W, C_nodes, F_nodes, n_block_rows, bs)
    return P_bsr


def build_asw_block_inverses_bsr(A: sp.bsr_matrix,
                                 asw_sd_size: int,
                                 overlap: int = 0):
    """
    Build dense ASW / block-Jacobi local inverses from contiguous diagonal blocks of A,
    optionally with overlap.

    Parameters
    ----------
    A : bsr_matrix
        Square system matrix on this level.
    asw_sd_size : int
        Subdomain size in scalar DOFs.
    overlap : int
        Overlap size in scalar DOFs.
        overlap = 0 gives non-overlapping block Jacobi.
        overlap > 0 gives overlapping ASW-style subdomains.

    Returns
    -------
    block_inv : list[np.ndarray]
        Dense inverses of each local subdomain matrix.
    block_ranges : list[tuple[int, int]]
        Half-open DOF index ranges (start, end) for each subdomain.
    """
    assert sp.isspmatrix_bsr(A)
    assert A.shape[0] == A.shape[1]
    assert asw_sd_size >= 1
    assert overlap >= 0
    if overlap >= asw_sd_size:
        raise ValueError(f"Require overlap < asw_sd_size, got {overlap=} and {asw_sd_size=}")

    A_csr = A.tocsr(copy=True)

    n = A.shape[0]
    stride = asw_sd_size - overlap

    block_inv = []
    block_ranges = []

    start = 0
    while start < n:
        end = min(start + asw_sd_size, n)

        Ablock = A_csr[start:end, start:end].toarray()
        block_inv.append(np.linalg.inv(Ablock))
        block_ranges.append((start, end))

        if end == n:
            break
        start += stride

    return block_inv, block_ranges


def apply_asw_blocks_bsr(rhs: np.ndarray,
                         block_inv: list[np.ndarray],
                         block_ranges: list[tuple[int, int]],
                         weights: np.ndarray = None) -> np.ndarray:
    """
    Apply stored ASW / block-Jacobi inverse:
        z = sum_i R_i^T A_i^{-1} R_i rhs

    For non-overlapping blocks, this reduces to ordinary block Jacobi.
    For overlapping blocks, local corrections are accumulated.

    Parameters
    ----------
    rhs : ndarray
        Global residual / RHS.
    block_inv : list[np.ndarray]
        Dense local inverses.
    block_ranges : list[(int,int)]
        Half-open DOF index ranges for each subdomain.
    weights : ndarray or None
        Optional partition-of-unity weights of shape (n,). If provided,
        each accumulated DOF is scaled by weights[dof] at the end.

    Returns
    -------
    z : ndarray
        Global accumulated correction.
    """
    z = np.zeros_like(rhs)

    for Binv, (s, e) in zip(block_inv, block_ranges):
        z[s:e] += Binv @ rhs[s:e]

    if weights is not None:
        z *= weights

    return z



class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x


class ClassicalAMG_BSRSolver:
    """general multilevel AMG solver..."""

    def __init__(self,
                 A_free: sp.bsr_matrix,
                 A: sp.bsr_matrix,
                 threshold: float = 0.25,
                 level: int = 0,
                 nsmooth: int = 1,
                 smoother: str = "gs",
                 asw_overlap:int=0,
                 omegaSmooth: float = 0.7,
                 asw_sd_size: int = None,
                 coarsening_fcn=None, interpolation_fcn=None):
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)

        smoother = smoother.lower()
        if smoother not in ("gs", "asw"):
            raise ValueError(f"Unknown smoother '{smoother}', expected 'gs' or 'asw'")

        if smoother == "asw" and asw_sd_size is None:
            raise ValueError("Must provide asw_sd_size when smoother='asw'")

        self.A_free = A_free
        self.A = A
        self.nsmooth = nsmooth
        self.iters = 0
        self.level = level
        self.smoother = smoother
        self.omegaSmooth = omegaSmooth
        self.asw_sd_size = asw_sd_size

        self.coarsening_fcn = coarsening_fcn
        self.interpolation_fcn = interpolation_fcn

        # tentative + smoothed prolongator
        block_dim = A.data.shape[-1]

        # do coarsening and get interpolation
        C_mask, F_mask = coarsening_fcn(A_free, threshold)
        # C_mask, F_mask = coarsening_fcn(A, threshold)
        # print(f"{np.sum(C_mask)=} {np.sum(F_mask)=}")
        C_mask, F_mask = repair_bsr_CF_splitting(A_free, C_mask, F_mask)
        # print(f"{C_mask=} {F_mask=}")
        C_mask = np.where(C_mask)[0]
        F_mask = np.where(F_mask)[0]
        self.P = interpolation_fcn(A_free, C_mask, F_mask)
        self.R = self.P.T

        # Galerkin coarse operators
        self.Ac = (self.R @ (A @ self.P)).tobsr()
        self.Ac_free = (self.R @ (A_free @ self.P)).tobsr()

        # build ASW smoother data for this level if requested
        self.asw_block_inv = None
        self.asw_block_ranges = None
        if self.smoother == "asw":
            self.asw_block_inv, self.asw_block_ranges = build_asw_block_inverses_bsr(
                self.A, self.asw_sd_size, overlap=asw_overlap
            )

        # complexity bookkeeping
        self.fine_nnz = self.A.nnz
        self.coarse_nnz = self.Ac.nnz
        self.coarse_solver = None

        if level == 0:
            print("level 0 is AMG solver..")

        coarse_nnodes = self.Ac.shape[0]
        max_coarse_nnz = coarse_nnodes**2

        if self.fine_nnz == self.coarse_nnz:
            raise RuntimeError(
                f"ERROR: {self.fine_nnz=} == {self.coarse_nnz=} : "
                f"lower aggregation threshold from {threshold=} to lower value..\n"
            )

        if self.coarse_nnz >= 0.4 * max_coarse_nnz or coarse_nnodes <= 100:
            print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            print(f"level {level+1} building AMG solver")
            self.coarse_solver = ClassicalAMG_BSRSolver(
                self.Ac_free,
                self.Ac,
                threshold=threshold,
                level=level + 1,
                nsmooth=nsmooth,
                smoother=smoother,
                omegaSmooth=omegaSmooth,
                asw_sd_size=asw_sd_size,
                coarsening_fcn=coarsening_fcn, interpolation_fcn=interpolation_fcn
            )

        if level == 0:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=}")
            print(f"\tnum nodes per level = [{self.num_nodes_list}]")

    def _smooth_gs(self, rhs: np.ndarray, x: np.ndarray, transpose: bool = False) -> np.ndarray:
        """
        One damped GS sweep:
            x <- x + omegaSmooth * (x_gs - x)
        """
        if transpose:
            x_gs = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x, num_iter=1)
        else:
            x_gs = block_gauss_seidel_6dof(self.A, rhs, x0=x, num_iter=1)
        return x + self.omegaSmooth * (x_gs - x)

    def _smooth_asw(self, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        One damped ASW / block-Jacobi defect-correction sweep:
            x <- x + omegaSmooth * M^{-1}(rhs - A x)
        """
        r = rhs - self.A.dot(x)
        z = apply_asw_blocks_bsr(r, self.asw_block_inv, self.asw_block_ranges)
        return x + self.omegaSmooth * z

    def _apply_smoother(self,
                        rhs: np.ndarray,
                        x: np.ndarray,
                        nsweeps: int,
                        transpose: bool = False) -> np.ndarray:
        if nsweeps <= 0:
            return x

        if self.smoother == "gs":
            for _ in range(nsweeps):
                x = self._smooth_gs(rhs, x, transpose=transpose)
            return x

        elif self.smoother == "asw":
            for _ in range(nsweeps):
                x = self._smooth_asw(rhs, x)
            return x

        raise RuntimeError(f"Unsupported smoother '{self.smoother}'")

    def solve(self, rhs):
        """
        One AMG V-cycle:
          1) pre-smooth nsmooth times
          2) coarse-grid correction once
          3) post-smooth nsmooth times
        """
        x = np.zeros_like(rhs)
        self.iters += 1

        # pre-smoothing
        x = self._apply_smoother(rhs, x, self.nsmooth, transpose=False)

        # coarse-grid correction
        r = rhs - self.A.dot(x)
        rc = self.R.dot(r)
        ec = self.coarse_solver.solve(rc)
        x += self.P.dot(ec)

        # post-smoothing
        x = self._apply_smoother(rhs, x, self.nsmooth, transpose=True)

        return x

    @property
    def total_nnz(self) -> int:
        if isinstance(self.coarse_solver, ClassicalAMG_BSRSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else:
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, ClassicalAMG_BSRSolver):
            return self.coarse_solver.num_levels + 1
        else:
            return 2

    @property
    def num_nodes_list(self) -> str:
        bs = self.A.data.shape[-1]
        nnodes_f = self.A.shape[0] // bs
        nnodes_c = self.Ac.shape[0] // bs
        if isinstance(self.coarse_solver, ClassicalAMG_BSRSolver):
            return str(nnodes_f) + "," + self.coarse_solver.num_nodes_list
        else:
            return f"{nnodes_f},{nnodes_c}"

    @property
    def total_vcycles(self) -> int:
        return self.iters
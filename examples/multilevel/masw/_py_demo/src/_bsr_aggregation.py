import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _smoothers import block_gauss_seidel_6dof, block_gauss_seidel_6dof_transpose

def strength_matrix_bsr(A:sp.bsr_matrix, threshold:float=0.25):
    """
    Compute strength of connections C_{ij} for sparse BSR matrix (slightly different than above strength and strength^T version)
    produces only one matrix

    comes from this paper on aggregation
    https://epubs.siam.org/doi/epdf/10.1137/110838844
    """

    assert sp.isspmatrix_bsr(A)

    # strength of connection graph
    N = A.shape[0]
    block_dim = A.data.shape[-1]
    nnodes = N // block_dim
    
    # get diagonal norms
    diag_nrms = []
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            if i == j:
                diag_nrms += [np.linalg.norm(A.data[jp])]
    
    STRENGTH = [[] for i in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            norm = np.linalg.norm(A.data[jp])
            lb = threshold * np.sqrt(diag_nrms[i] * diag_nrms[j])
            if norm >= lb:
                STRENGTH[i] += [j]
    # can be converted into CSR style NZ pattern (though has no float data, just NZ pattern C_{ij})
    return STRENGTH


def greedy_serial_aggregation_bsr(A:sp.bsr_matrix, threshold:float=0.25):
    """from paper https://epubs.siam.org/doi/epdf/10.1137/110838844, with parallel versions discussed also
    and GPU version here https://www.sciencedirect.com/science/article/pii/S0898122114004143"""

    # greedy serial aggregation
    assert sp.isspmatrix_bsr(A)
    N = A.shape[0]
    block_dim = A.data.shape[-1]
    nnodes = N // block_dim
    STRENGTH = strength_matrix_bsr(A, threshold)
    # print(f"{STRENGTH=}")

    # almost the same code as greedy_serial_aggregation_csr BTW
    # print(f"aggreg {nnodes=} {block_dim=}")
    aggregate_ind = np.full(nnodes, -1) # keep track of aggregate indices (-1 is unpicked, >= 0 is picked and which aggregate it belongs to)\
    aggregate_groups = []
    _ct = 0

    # first phase creates all the aggregates
    for i in range(nnodes):
        strong_neighbors = np.array(STRENGTH[i])
        # if i and each of its strong neighbors not picked, form aggregate:
        picked_neighbors = [ii for ii in strong_neighbors if aggregate_ind[ii] != -1]
        if len(picked_neighbors) == 0:
            # print(f"aggregate {_ct} from node {i=} and {strong_neighbors=}")
            aggregate_groups += [list(strong_neighbors)]
            aggregate_ind[strong_neighbors] = _ct
            _ct += 1 # increment aggregate counter

    # second phase adds all remaining nodes to nearby aggregates
    for i in range(nnodes):
        if aggregate_ind[i] != -1: continue # only look at unpicked nodes
        strong_neighbors = np.array(STRENGTH[i])
        nb_agg_ind = aggregate_ind[strong_neighbors]
        valid_nb_agg_ind = nb_agg_ind[nb_agg_ind != -1]
        # sweep into an aggregate with smallest size
        nb_agg_sizes = np.array([len(aggregate_groups[ind]) for ind in valid_nb_agg_ind])
        min_size_ind = np.argmin(nb_agg_sizes)
        agg_ind = valid_nb_agg_ind[min_size_ind]

        # now add into that aggregate
        aggregate_groups[agg_ind] += [i]
        aggregate_ind[i] = agg_ind

    assert(not np.any(aggregate_ind == -1))
    return aggregate_ind


def get_rigid_body_modes(xpts, th:float=1.0, vpn:int=6, th_flip:bool=False):
    """get rigid body modes of particular mesh"""

    _x = xpts[0::3]; _y = xpts[1::3]; _z = xpts[2::3]
    nnodes = _x.shape[0]

    

    if vpn == 6:

        Bpred = np.zeros((nnodes, 6, 6))

        # first three modes just as translation
        for imode in range(3):
            Bpred[:, imode, imode] = 1.0

        # vw mode 3
        Bpred[:, 1, 3] = -th * _z
        Bpred[:, 2, 3] = th * _y
        Bpred[:, 3, 3] = th

        # uw mode 4
        Bpred[:, 0, 4] = th * _z
        Bpred[:, 2, 4] = -th * _x
        Bpred[:, 4, 4] = th

        # uv mode 5
        Bpred[:, 0, 5] = -th * _y
        Bpred[:, 1, 5] = th * _x
        Bpred[:, 5, 5] = th

    elif vpn == 3 and not th_flip:

        Bpred = np.zeros((nnodes, 3, 3))

        # w for plate
        Bpred[:, 0, 0] = 1.0

        # thx mode
        Bpred[:, 0, 1] = th * _y
        Bpred[:, 1, 1] = th

        # uw mode 4
        Bpred[:, 0, 2] = -th * _x
        Bpred[:, 2, 2] = th

    elif vpn == 3 and th_flip:
        # means that shear strains are instead (w_{,x} + th_x) and (w_{,y} + th_y)

        Bpred = np.zeros((nnodes, 3, 3))

        # w for plate
        Bpred[:, 0, 0] = 1.0

        # thx mode
        Bpred[:, 0, 1] = th * _x
        Bpred[:, 1, 1] = -th

        # uw mode 4
        Bpred[:, 0, 2] = th * _y
        Bpred[:, 2, 2] = -th
    
    return Bpred

def get_coarse_rigid_body_modes(B:np.ndarray, xpts:np.ndarray, aggregate_ind:np.ndarray, vpn:int=6):
    """helper method for constructing tentative prolongator"""
    # nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    # 2) compute coarse rigid body modes R for aggregates (average over nodes)
    R = np.zeros((num_agg, vpn, vpn))
    xpts_c = np.zeros((num_agg, 3))
    for iagg in range(num_agg):
        agg_nodes = aggregate_ind == iagg
        Bk = B[agg_nodes] #.reshape(-1, 6)  # (num_nodes_in_agg * 6) x 6
        # print(f"{Bk.shape=} {Bk=}")
        R[iagg] = np.mean(Bk, axis=0)
        # print(f"{R[iagg]=}")
        xpts_c[iagg] = np.mean(xpts[agg_nodes], axis=0)
    return R, xpts_c

def tentative_prolongator_bsr(B: np.ndarray, aggregate_ind: np.ndarray, bc_flags:np.ndarray, vpn:int=6):
    nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    rowp = np.arange(0, nnodes + 1)
    cols = np.zeros(nnodes, dtype=np.int32)
    for iagg in range(num_agg):
        cols[aggregate_ind == iagg] = iagg

    # B = get_rigid_body_modes(xpts.reshape(3 * nnodes))  # (nnodes, 6, 6)
    # R, xpts_c = get_coarse_rigid_body_modes(B, xpts, aggregate_ind)
    # print(f"{R[0]=}")
    R = np.zeros((num_agg, vpn, vpn))

    data = np.zeros((nnodes, vpn, vpn))

    # QR decomposition of B * Qk = Rk (which also discovers aggregate coarse rigid body modes)
    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        nk = np.sum(agg_mask)

        # Stack rigid body modes for this aggregate
        Bk = B[agg_mask].reshape(nk * vpn, vpn)

        # Thin QR factorization
        Qk, Rk = np.linalg.qr(Bk, mode="reduced")

        # Qk^T Qk = I   → tentative prolongator
        data[agg_mask] = Qk.reshape(nk, vpn, vpn)

        # Store Rk if you need the coarse nullspace
        R[iagg] = Rk

    # since tentative prolongator has only one block nonzero per fine node
    # was missing this before, need this with AMG (otherwise will get NZ deflection in fine prolong)
    for inode in range(nnodes):
        for ii in range(vpn):
            idof = vpn * inode + ii
            if bc_flags[idof]: # apply dirichlet bcs
                data[inode,ii,:] = 0.0

    P0 = sp.bsr_matrix((data, cols, rowp), blocksize=(vpn, vpn))
    return P0, R

def get_bc_flags_bsr(A: sp.bsr_matrix, tol=1e-14) -> np.ndarray:
    """Return True for constrained DOFs in a BSR matrix (diagonal≈I, off-diagonal≈0)"""
    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    flags = np.zeros(A.shape[0], dtype=bool)

    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i+1]
        idx = A.indices[start:end]
        blk = A.data[start:end]

        diag = blk[idx == i]
        off = blk[idx != i]

        flags[i*b:(i+1)*b] = (
            np.all(np.abs(diag - np.eye(b)) < tol) and
            (off.size == 0 or np.all(np.abs(off) < tol))
        )

    return flags

def orthog_nullspace_projector(P: sp.bsr_matrix, Bc: np.ndarray, bcs: np.ndarray):
    """Apply the orthogonal projector to prevent nullspace modes (BSR version)"""

    Pnew = sp.bsr_matrix(
        (P.data.copy(), P.indices.copy(), P.indptr.copy()),
        shape=P.shape,
        blocksize=P.blocksize
    )

    block_dim = P.blocksize[0]
    nblocks_row = P.shape[0] // block_dim

    for brow in range(nblocks_row):
        # construct Fi: mask constrained DOFs for this fine node
        Fi_bcs = np.array([_ for _ in range(block_dim) if bcs[block_dim*brow + _]])
        Fi = np.eye(block_dim)
        if Fi_bcs.size > 0:
            Fi[Fi_bcs, :] = 0.0

        # precompute U and store bcols for this block row
        ncols = P.indptr[brow+1] - P.indptr[brow]
        bcols = P.indices[P.indptr[brow]:P.indptr[brow+1]]

        # this method did not work
        # U = np.hstack([Bc[bcols[j]] @ Fi for j in range(ncols)])  # shape (b, ncols*b)
        # dP_row = np.hstack([P.data[P.indptr[brow]+j] for j in range(ncols)])  # shape (b, ncols*b)
        # # orthogonal projector applied in one shot
        # UTU_inv = np.linalg.pinv(U.T @ U)
        # dP_row_proj = dP_row - dP_row @ U @ UTU_inv @ U.T
        # # unstack back into BSR blocks
        # for j in range(ncols):
        #     Pnew.data[P.indptr[brow]+j] = dP_row_proj[:, j*block_dim:(j+1)*block_dim]

        # this method doesn't quite work either..
        U_list = []
        for j in range(ncols):
            U_list.append(Bc[bcols[j]] @ Fi)
        # compute PU and UTU
        PU = sum(P.data[P.indptr[brow]+j] @ U_list[j] for j in range(ncols))
        UTU = sum(U_list[j].T @ U_list[j] for j in range(ncols))
        UTU_inv = np.linalg.pinv(UTU)

        # apply projector to each P block
        for j in range(ncols):
            Pnew.data[P.indptr[brow]+j] -= PU @ UTU_inv @ U_list[j].T

        # DEBUG : double check linear system..
        # new PU should be zero in this row
        # new_PU = sum(Pnew.data[P.indptr[brow]+j] @ U_list[j] for j in range(ncols))
        # new_PU_nrm = np.linalg.norm(new_PU)
        # print(f'{brow=} : {new_PU_nrm=:.4e}')

    # exit()

    # -----------------------------
    # Verification: check nullspace orthogonality
    # -----------------------------
    # unconstrained_dofs = np.where(bcs == 0)[0]
    # print(f"{unconstrained_dofs=}")
    # print(f"{bcs=}")
    # exit()

    # # form Uc = Bc * Fi
    # Uc = Bc.copy()
    # Uc[bcs // 6] = 0.0
    # for i in range(6):
    #     bc_vec = Uc[:,:,i].reshape(Pnew.shape[-1])
    #     p_vec = Pnew.dot(bc_vec)
    #     # print(f"mode {i=} : {p_vec=}")
    #     p_uc_vec = p_vec[unconstrained_dofs]
    #     # print(f"\t{p_uc_vec=}")
    #     nrm = np.max(np.abs(p_vec))
    #     nrm2 = np.max(np.abs(p_uc_vec))
    #     print(f"{nrm=:.4e} {nrm2=:.4e}")
    # exit()

    return Pnew

def spectral_radius_block_DinvA_bsr(
    A: sp.bsr_matrix,
    maxiter=200,
    tol=1e-8,
):
    """
    Estimate spectral radius of block Jacobi operator D_b^{-1} A.

    Parameters
    ----------
    A : scipy.sparse.bsr_matrix
    maxiter : int
    tol : float

    Returns
    -------
    rho : float
        Estimated spectral radius
    """

    if not sp.isspmatrix_bsr(A):
        raise TypeError("A must be a BSR matrix")

    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    n = A.shape[0]

    # --------------------------------------------------
    # Precompute block diagonal inverses
    # --------------------------------------------------
    Dinv_blocks = np.zeros((nblocks, b, b))
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i + 1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        Dblock = A.data[start + diag_idx[0]]
        Dinv_blocks[i] = np.linalg.inv(Dblock)

    # --------------------------------------------------
    # Matrix-free operator y = D^{-1} A x
    # --------------------------------------------------
    def matvec(x):
        y = A @ x
        y = y.reshape(nblocks, b)

        for i in range(nblocks):
            y[i] = Dinv_blocks[i] @ y[i]

        return y.reshape(n)

    M = spla.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=A.dtype,
    )

    eigval = spla.eigs(
        M,
        k=1,
        which="LM",
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=False,
    )

    return np.abs(eigval[0])


def smooth_prolongator_bsr(T: sp.bsr_matrix, A: sp.bsr_matrix, Bc: np.ndarray,
                           bc_flags: np.ndarray = None, omega: float = None, near_kernel: bool = True):
    """
    Energy-smoothing single-step for tentative prolongator (BSR)
    P = T - omega * Dinv * A * T
    Nullspace components removed from the update Dinv*AT before applying omega.
    """
    if A.blocksize != T.blocksize:
        raise ValueError("A and T must have the same blocksize")
    b = A.blocksize[0]
    nblocks = A.shape[0] // b

    # --- omega ---
    if omega is None:
        rho = spectral_radius_block_DinvA_bsr(A)
        omega = 2.0 / rho * 0.9
        print(f"{omega=}")

    # --- block diagonal inverse ---
    Dinv_blocks = np.zeros((nblocks, b, b))
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i+1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        D_block = A.data[start + diag_idx[0]]
        Dinv_blocks[i] = np.linalg.inv(D_block)

    # --- compute update ---
    AT = A @ T
    dP = AT.copy()
    for i in range(nblocks):
        start, end = dP.indptr[i], dP.indptr[i+1]
        dP.data[start:end] = Dinv_blocks[i] @ dP.data[start:end]

    # --- nullspace removal ---
    if near_kernel:
        dP = orthog_nullspace_projector(dP, Bc, bc_flags)

    # --- zero Dirichlet DOFs in dP ---
    if bc_flags is not None:
        ndof = dP.blocksize[0]
        for i in range(dP.shape[0] // ndof):
            start, end = dP.indptr[i], dP.indptr[i+1]
            for j in range(ndof):
                if bc_flags[i*ndof + j]:
                    dP.data[start:end, j, :] = 0.0

    # --- final Jacobi update ---
    P = T - omega * dP

    return P


class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class MASWLevel_BSR:
    level: int
    A_free: sp.bsr_matrix
    A: sp.bsr_matrix
    B: np.ndarray

    # stored ASW / block-Jacobi inverse data
    block_inv: list[np.ndarray]
    block_ranges: list[tuple[int, int]]

    # one-level transfer operators: coarse -> current
    P: sp.spmatrix = None
    R: sp.spmatrix = None

    # global transfer operators: level -> finest
    P_global: sp.spmatrix = None
    R_global: sp.spmatrix = None

def build_asw_block_inverses_bsr(A: sp.bsr_matrix, asw_sd_size: int):
    """
    Build dense block-Jacobi inverses from contiguous diagonal blocks of A.

    For simplicity (demo code), the BSR matrix is temporarily converted
    to CSR so scalar DOF slicing works.

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

    # Convert once for slicing
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


def enforce_symmetric_dirichlet_bcs_bsr(A: sp.bsr_matrix, tol: float = 1e-12) -> sp.bsr_matrix:
    """
    Symmetrize Dirichlet BC treatment for a BSR matrix without changing
    the BSR nonzero pattern.

    Assumes constrained scalar DOFs already appear as identity rows
    numerically:
        diag = 1
        all off-diagonal entries in that scalar row are zero (up to tol)

    This routine:
      1. Detects constrained scalar DOFs from the existing BSR values
      2. Zeros both the corresponding rows and columns numerically
         while preserving the exact BSR sparsity pattern
      3. Restores 1 on constrained scalar diagonal entries

    Parameters
    ----------
    A : bsr_matrix
        Square BSR matrix with row-only Dirichlet BC treatment.
    tol : float
        Tolerance for detecting unit diagonal / zero off-diagonals.

    Returns
    -------
    A_sym : bsr_matrix
        BSR matrix with same sparsity pattern as A, but with symmetric
        Dirichlet treatment suitable for PCG.
    """
    assert sp.isspmatrix_bsr(A)
    assert A.shape[0] == A.shape[1]

    bs_row, bs_col = A.blocksize
    assert bs_row == bs_col
    bs = bs_row

    n = A.shape[0]
    assert n % bs == 0
    nblock = n // bs

    indptr = A.indptr
    indices = A.indices
    data = A.data

    # --------------------------------------------------
    # 1) Detect constrained scalar DOFs from scalar rows
    #    inside the BSR structure
    # --------------------------------------------------
    bc_dofs = np.zeros(n, dtype=bool)

    for ib in range(nblock):
        row_start = indptr[ib]
        row_end = indptr[ib + 1]

        # check each scalar row inside this BSR block-row
        for rloc in range(bs):
            ig = ib * bs + rloc

            diag_val = 0.0
            found_diag = False
            offdiag_max = 0.0

            for k in range(row_start, row_end):
                jb = indices[k]
                blk = data[k]

                for cloc in range(bs):
                    jg = jb * bs + cloc
                    val = blk[rloc, cloc]

                    if ig == jg:
                        diag_val = val
                        found_diag = True
                    else:
                        offdiag_max = max(offdiag_max, abs(val))

            if found_diag and abs(diag_val - 1.0) <= tol and offdiag_max <= tol:
                bc_dofs[ig] = True

    if not np.any(bc_dofs):
        return A.copy()
    
    print("FOUND BC DOFs, zeroing cols")

    # --------------------------------------------------
    # 2) Zero corresponding rows/cols numerically
    #    but preserve exact BSR pattern
    # --------------------------------------------------
    A_sym = A.copy()

    for ib in range(nblock):
        row_start = A_sym.indptr[ib]
        row_end = A_sym.indptr[ib + 1]

        for k in range(row_start, row_end):
            jb = A_sym.indices[k]
            blk = A_sym.data[k]

            for rloc in range(bs):
                ig = ib * bs + rloc
                row_is_bc = bc_dofs[ig]

                for cloc in range(bs):
                    jg = jb * bs + cloc
                    col_is_bc = bc_dofs[jg]

                    if row_is_bc or col_is_bc:
                        if ig == jg and row_is_bc:
                            blk[rloc, cloc] = 1.0
                        else:
                            blk[rloc, cloc] = 0.0

    return A_sym

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class MASWLevel_BSR:
    level: int
    A_free: sp.bsr_matrix
    A: sp.bsr_matrix
    B: np.ndarray
    xpts: np.ndarray

    # stored ASW / block-Jacobi inverse data
    block_inv: list[np.ndarray]
    block_ranges: list[tuple[int, int]]

    # optional direct coarse solve
    direct_inv: np.ndarray = None
    is_direct: bool = False

    # one-level transfer operators: coarse -> current
    P: sp.csr_matrix = None
    R: sp.csr_matrix = None

    # global transfer operators: level -> finest/original ordering
    P_global: sp.csr_matrix = None
    R_global: sp.csr_matrix = None


class MASW_BSRSolver:
    """
    Fixed-level multilevel additive Schwarz solver for BSR matrices.
    """

    def __init__(self,
                 A_free: sp.bsr_matrix,
                 A: sp.bsr_matrix,
                 B: np.ndarray,
                 xpts: np.ndarray,
                 num_levels: int, # or None
                 asw_sd_size: int,
                 threshold: float = 0.25,
                 omega: float = 0.7,
                 near_kernel: bool = True,
                 coarsening_fcn=None,
                 smooth_prolongator: bool = True,
                 nsmooth: int = 2,
                 omegaJac: float = 0.7,
                 node_sort_fcn=None,
                 galerkin_fcn=None,
                 morton_sort: bool = True,
                 coarse_dof_threshold: int = 100,
                 coarse_nnz_frac: float = 0.4):
        """
        Parameters
        ----------
        num_levels : int or None
            Total number of levels, including finest level.
            If None, coarsen until the next coarse grid is small enough,
            then use a direct solve on the coarsest level.
        coarse_dof_threshold : int
            If num_levels is None, stop when the next coarse level has
            <= coarse_dof_threshold scalar DOFs.
        coarse_nnz_frac : float
            Also stop if nnz(next coarse) >= coarse_nnz_frac * (n_coarse^2).
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)
        assert A.shape[0] == A.shape[1]
        assert A_free.shape == A.shape
        assert num_levels is None or num_levels >= 1
        assert asw_sd_size >= 1

        self.A_free = A_free.tobsr()
        self.A = A.tobsr()
        self.B = np.asarray(B)
        self.xpts = np.asarray(xpts, dtype=float)

        self.num_levels_requested = num_levels
        self.asw_sd_size = asw_sd_size
        self.threshold = threshold
        self.omega = omega
        self.near_kernel = near_kernel
        self.coarsening_fcn = coarsening_fcn
        self.smooth_prolongator = smooth_prolongator
        self.node_sort_fcn = node_sort_fcn
        self.galerkin_fcn = galerkin_fcn
        self.block_dim = A_free.data.shape[-1]
        self.morton_sort = morton_sort
        self.coarse_dof_threshold = coarse_dof_threshold
        self.coarse_nnz_frac = coarse_nnz_frac

        self.nsmooth = nsmooth
        self.omegaJac = omegaJac
        self.iters = 0

        bs = self.block_dim
        nnodes = self.A.shape[0] // bs
        assert self.xpts.shape[0] == nnodes, (
            f"xpts.shape[0]={self.xpts.shape[0]} must equal nnodes={nnodes}"
        )
        assert self.B.shape[0] == nnodes, (
            f"B.shape[0]={self.B.shape[0]} must equal nnodes={nnodes}"
        )

        self.levels: list[MASWLevel_BSR] = []

        self._build_hierarchy()

        print(f"Multilevel ASW-BSR with {self.num_levels=} and {self.operator_complexity=:.3f}")
        print(f"\tnum DOF per level = {self.num_dof_list}")
        print(f"\tnum nodes per level = {self.num_nodes_list}")
        print(f"\tcoarsest direct solve = {self.levels[-1].is_direct}")

    # -------------------------------------------------------------------------
    # Morton helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _spread_bits_2d(v: np.ndarray) -> np.ndarray:
        x = v.astype(np.uint64)
        x = (x | (x << 16)) & np.uint64(0x0000FFFF0000FFFF)
        x = (x | (x << 8))  & np.uint64(0x00FF00FF00FF00FF)
        x = (x | (x << 4))  & np.uint64(0x0F0F0F0F0F0F0F0F)
        x = (x | (x << 2))  & np.uint64(0x3333333333333333)
        x = (x | (x << 1))  & np.uint64(0x5555555555555555)
        return x

    @classmethod
    def morton_perm_from_xpts(cls, xpts: np.ndarray, nbits: int = 16) -> np.ndarray:
        xpts = np.asarray(xpts, dtype=float)
        assert xpts.ndim == 2
        assert xpts.shape[1] >= 2, "Need at least 2 coordinate columns for Morton sorting."

        xy = xpts[:, :2].copy()

        xmin = xy.min(axis=0)
        xmax = xy.max(axis=0)
        span = xmax - xmin
        span[span == 0.0] = 1.0

        scale = (2**nbits - 1) / span
        q = np.floor((xy - xmin) * scale + 0.5).astype(np.uint32)

        xq = q[:, 0]
        yq = q[:, 1]
        codes = cls._spread_bits_2d(xq) | (cls._spread_bits_2d(yq) << np.uint64(1))

        perm = np.lexsort((xy[:, 1], xy[:, 0], codes))
        return perm.astype(np.int64)

    @staticmethod
    def node_perm_to_dof_perm(perm_nodes: np.ndarray, block_dim: int) -> np.ndarray:
        perm_nodes = np.asarray(perm_nodes, dtype=np.int64)
        return (perm_nodes[:, None] * block_dim + np.arange(block_dim, dtype=np.int64)[None, :]).ravel()

    @staticmethod
    def perm_matrix_from_perm(perm: np.ndarray) -> sp.csr_matrix:
        n = len(perm)
        data = np.ones(n)
        rows = perm
        cols = np.arange(n)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def reorder_level_by_nodes(self,
                               A_free: sp.bsr_matrix,
                               A: sp.bsr_matrix,
                               B: np.ndarray,
                               xpts: np.ndarray,
                               perm_nodes: np.ndarray):
        block_dim = A.data.shape[-1]
        block_shape = (block_dim, block_dim)

        perm_dof = self.node_perm_to_dof_perm(perm_nodes, block_dim)

        A_free_sorted = A_free.tocsr()[perm_dof, :][:, perm_dof].tobsr(block_shape)
        A_sorted = A.tocsr()[perm_dof, :][:, perm_dof].tobsr(block_shape)
        B_sorted = np.asarray(B)[perm_nodes, :]
        xpts_sorted = np.asarray(xpts)[perm_nodes, :]

        S_nodes = self.perm_matrix_from_perm(perm_nodes)
        S_dof = self.perm_matrix_from_perm(perm_dof)

        return A_free_sorted, A_sorted, B_sorted, xpts_sorted, S_nodes, S_dof, perm_nodes, perm_dof

    @staticmethod
    def average_xpts_by_aggregate(xpts: np.ndarray, aggregate_ind: np.ndarray) -> np.ndarray:
        xpts = np.asarray(xpts, dtype=float)
        aggregate_ind = np.asarray(aggregate_ind, dtype=np.int64)

        n_agg = int(np.max(aggregate_ind)) + 1
        ndim = xpts.shape[1]

        xpts_c = np.zeros((n_agg, ndim), dtype=float)
        counts = np.zeros(n_agg, dtype=np.int64)

        for i, a in enumerate(aggregate_ind):
            xpts_c[a] += xpts[i]
            counts[a] += 1

        if np.any(counts == 0):
            raise RuntimeError("Empty aggregate encountered while averaging xpts.")

        xpts_c /= counts[:, None]
        return xpts_c

    # -------------------------------------------------------------------------
    # direct / ASW apply helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def build_direct_inverse_bsr(A: sp.bsr_matrix) -> np.ndarray:
        """
        Build a dense inverse for the coarsest direct level.
        """
        Ad = A.toarray()
        return np.linalg.inv(Ad)

    def apply_level_solver(self, lev: MASWLevel_BSR, rhs_level: np.ndarray) -> np.ndarray:
        """
        Apply either ASW blocks or direct inverse on one level.
        """
        if lev.is_direct:
            return lev.direct_inv @ rhs_level
        return apply_asw_blocks_bsr(rhs_level, lev.block_inv, lev.block_ranges)

    def _should_stop_with_direct(self, A_next: sp.bsr_matrix) -> bool:
        """
        Automatic stop criterion when num_levels is None.
        """
        ncoarse_dof = A_next.shape[0]
        max_coarse_nnz = ncoarse_dof ** 2

        return (
            ncoarse_dof <= self.coarse_dof_threshold
            or A_next.nnz >= self.coarse_nnz_frac * max_coarse_nnz
        )

    # -------------------------------------------------------------------------
    # Main hierarchy build
    # -------------------------------------------------------------------------

    def _build_hierarchy(self):
        """
        Build all levels up front.
        """
        A_free_l = self.A_free
        A_l = self.A
        B_l = self.B
        xpts_l = self.xpts

        block_dim = self.block_dim
        block_shape = (block_dim, block_dim)

        # global maps from current internal ordering -> finest/original ordering
        P_global_l = sp.eye(A_l.shape[0], format="csr")
        R_global_l = sp.eye(A_l.shape[0], format="csr")

        level = 0
        while True:
            # --------------------------------------------------
            # 1) Optional Morton sorting / node reordering
            # --------------------------------------------------
            if self.morton_sort:
                if self.node_sort_fcn is not None:
                    perm_nodes = np.asarray(self.node_sort_fcn(xpts_l), dtype=np.int64)
                else:
                    perm_nodes = self.morton_perm_from_xpts(xpts_l)

                A_free_sorted, A_sorted, B_sorted, xpts_sorted, _, S_dof, _, _ = \
                    self.reorder_level_by_nodes(A_free_l, A_l, B_l, xpts_l, perm_nodes)

                P_global_l = (P_global_l @ S_dof).tocsr()
                R_global_l = P_global_l.T.tocsr()
            else:
                A_free_sorted = A_free_l
                A_sorted = A_l
                B_sorted = B_l
                xpts_sorted = xpts_l

            A_free_sorted = A_free_sorted.tobsr(block_shape)
            A_sorted = A_sorted.tobsr(block_shape)
            B_sorted = np.asarray(B_sorted)
            xpts_sorted = np.asarray(xpts_sorted)

            # --------------------------------------------------
            # 2) If fixed num_levels and this is final level,
            #    store as ASW level and stop.
            # --------------------------------------------------
            if self.num_levels_requested is not None and level == self.num_levels_requested - 1:
                block_inv, block_ranges = build_asw_block_inverses_bsr(
                    A_sorted, self.asw_sd_size
                )

                lev = MASWLevel_BSR(
                    level=level,
                    A_free=A_free_sorted,
                    A=A_sorted,
                    B=B_sorted,
                    xpts=xpts_sorted,
                    block_inv=block_inv,
                    block_ranges=block_ranges,
                    direct_inv=None,
                    is_direct=False,
                    P=None,
                    R=None,
                    P_global=P_global_l,
                    R_global=R_global_l,
                )
                self.levels.append(lev)
                break

            # --------------------------------------------------
            # 3) Build/store ASW block inverses on this level
            # --------------------------------------------------
            block_inv, block_ranges = build_asw_block_inverses_bsr(
                A_sorted, self.asw_sd_size
            )

            lev = MASWLevel_BSR(
                level=level,
                A_free=A_free_sorted,
                A=A_sorted,
                B=B_sorted,
                xpts=xpts_sorted,
                block_inv=block_inv,
                block_ranges=block_ranges,
                direct_inv=None,
                is_direct=False,
                P=None,
                R=None,
                P_global=P_global_l,
                R_global=R_global_l,
            )
            self.levels.append(lev)

            # --------------------------------------------------
            # 4) Build P and R for next coarsening step
            # --------------------------------------------------
            if self.coarsening_fcn is None:
                aggregate_ind = greedy_serial_aggregation_bsr(
                    A_free_sorted, threshold=self.threshold
                )
            else:
                aggregate_ind = self.coarsening_fcn(
                    A_free_sorted, threshold=self.threshold
                )

            bc_flags = get_bc_flags_bsr(A_sorted)
            block_dim = A_sorted.data.shape[-1]

            T_l, Bc_l = tentative_prolongator_bsr(
                B_sorted,
                aggregate_ind,
                bc_flags,
                vpn=block_dim,
            )

            if self.smooth_prolongator:
                P_unsorted = smooth_prolongator_bsr(
                    T_l,
                    A_sorted,
                    Bc_l,
                    bc_flags,
                    omega=self.omega,
                    near_kernel=self.near_kernel,
                )
            else:
                P_unsorted = T_l

            P_unsorted = P_unsorted.tocsr()
            R_unsorted = P_unsorted.T.tocsr()

            # --------------------------------------------------
            # 5) Average xpts over aggregates for coarse nodes
            # --------------------------------------------------
            xpts_c_unsorted = self.average_xpts_by_aggregate(xpts_sorted, aggregate_ind)

            # --------------------------------------------------
            # 6) Build next coarse matrix in unsorted aggregate order
            # --------------------------------------------------
            if self.galerkin_fcn is None:
                A_free_next_unsorted = (R_unsorted @ (A_free_sorted @ P_unsorted)).tobsr(block_shape)
                A_next_unsorted = (R_unsorted @ (A_sorted @ P_unsorted)).tobsr(block_shape)
            else:
                A_free_next_unsorted = self.galerkin_fcn(
                    A_free_sorted, P_unsorted, R_unsorted
                ).tobsr(block_shape)
                A_next_unsorted = self.galerkin_fcn(
                    A_sorted, P_unsorted, R_unsorted
                ).tobsr(block_shape)

            # --------------------------------------------------
            # 7) Sort coarse level by Morton order and update P_l
            # --------------------------------------------------
            if self.morton_sort:
                if self.node_sort_fcn is not None:
                    perm_nodes_c = np.asarray(self.node_sort_fcn(xpts_c_unsorted), dtype=np.int64)
                else:
                    perm_nodes_c = self.morton_perm_from_xpts(xpts_c_unsorted)

                perm_dof_c = self.node_perm_to_dof_perm(perm_nodes_c, block_dim)
                S_dof_c = self.perm_matrix_from_perm(perm_dof_c)

                A_free_next = A_free_next_unsorted.tocsr()[perm_dof_c, :][:, perm_dof_c].tobsr(block_shape)
                A_next = A_next_unsorted.tocsr()[perm_dof_c, :][:, perm_dof_c].tobsr(block_shape)
                B_next = np.asarray(Bc_l)[perm_nodes_c, :]
                xpts_next = xpts_c_unsorted[perm_nodes_c, :]

                # P_unsorted maps coarse_unsorted -> current_sorted
                # convert to P_l mapping coarse_sorted -> current_sorted
                P_l = (P_unsorted @ S_dof_c).tocsr()
                R_l = P_l.T.tocsr()
            else:
                A_free_next = A_free_next_unsorted
                A_next = A_next_unsorted
                B_next = Bc_l
                xpts_next = xpts_c_unsorted
                P_l = P_unsorted
                R_l = R_unsorted

            # coarse -> current
            self.levels[level].P = P_l
            self.levels[level].R = R_l

            # --------------------------------------------------
            # 8) Build cumulative/global transfers for next level
            # --------------------------------------------------
            P_global_next = (P_global_l @ P_l).tocsr()
            R_global_next = P_global_next.T.tocsr()

            # --------------------------------------------------
            # 9) If adaptive depth, maybe stop and make next level direct
            # --------------------------------------------------
            if self.num_levels_requested is None and self._should_stop_with_direct(A_next):
                direct_inv = self.build_direct_inverse_bsr(A_next)

                coarse_lev = MASWLevel_BSR(
                    level=level + 1,
                    A_free=A_free_next,
                    A=A_next,
                    B=B_next,
                    xpts=xpts_next,
                    block_inv=None,
                    block_ranges=None,
                    direct_inv=direct_inv,
                    is_direct=True,
                    P=None,
                    R=None,
                    P_global=P_global_next,
                    R_global=R_global_next,
                )
                self.levels.append(coarse_lev)
                break

            # continue to next level
            P_global_l = P_global_next
            R_global_l = R_global_next
            A_free_l = A_free_next
            A_l = A_next
            B_l = B_next
            xpts_l = xpts_next
            level += 1

    # -------------------------------------------------------------------------
    # Info / diagnostics
    # -------------------------------------------------------------------------

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def block_size(self) -> int:
        return self.A.data.shape[-1]

    @property
    def num_dof_list(self) -> list[int]:
        return [lev.A.shape[0] for lev in self.levels]

    @property
    def num_nodes_list(self) -> list[int]:
        bs = self.block_size
        return [lev.A.shape[0] // bs for lev in self.levels]

    @property
    def nnz_list(self) -> list[int]:
        return [lev.A.nnz for lev in self.levels]

    @property
    def total_nnz(self) -> int:
        return sum(lev.A.nnz for lev in self.levels)

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.levels[0].A.nnz

    @property
    def total_vcycles(self) -> int:
        return self.iters

    # -------------------------------------------------------------------------
    # Apply / solve
    # -------------------------------------------------------------------------

    def apply_level_asw(self, level_id: int, rhs_level: np.ndarray) -> np.ndarray:
        lev = self.levels[level_id]
        return self.apply_level_solver(lev, rhs_level)

    def single_solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply multilevel additive Schwarz preconditioner.
        """
        rhs = np.asarray(rhs)
        assert rhs.ndim == 1
        assert rhs.shape[0] == self.levels[0].P_global.shape[0]

        z = np.zeros_like(rhs)

        for lev in self.levels:
            rhs_l = lev.R_global @ rhs
            z_l = self.apply_level_solver(lev, rhs_l)
            z += lev.P_global @ z_l

        return z

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply self.nsmooth defect-correction smoothing steps using the
        multilevel additive Schwarz operator.

        Iteration:
            x_{k+1} = x_k + omega * M^{-1} (rhs - A x_k)
        """
        rhs = np.asarray(rhs)
        assert rhs.ndim == 1
        assert rhs.shape[0] == self.levels[0].P_global.shape[0]

        A = self.A
        omega = self.omegaJac

        x = np.zeros_like(rhs)

        self.iters += self.nsmooth

        for _ in range(self.nsmooth):
            r = rhs - A @ x

            z = np.zeros_like(rhs)
            for lev in self.levels:
                r_l = lev.R_global @ r
                z_l = self.apply_level_solver(lev, r_l)
                z += lev.P_global @ z_l

            x += omega * z

        return x
import numpy as np
import scipy.sparse as sp

def structured_aggregation_csr(A:sp.csr_matrix, threshold:float=None, supernode_size:int=2):
    nnodes = A.shape[0]
    nx = int(np.sqrt(nnodes))
    # assert nx % 2 == 0 # need even # nodes
    nx_c = np.ceil(nx / 2.0)

    # construct 2x2 supernodes.. or 4x4
    aggregate_ind = np.zeros(nnodes, dtype=np.int32)
    for inode in range(nnodes):
        ix, iy = inode % nx, inode // nx
        ix_c = ix // 2; iy_c = iy // 2
        inode_c = nx_c * iy_c + ix_c

        aggregate_ind[inode] = inode_c

    return aggregate_ind

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _smoothers import gauss_seidel_csr, gauss_seidel_csr_transpose

def strength_matrix_csr(A:sp.csr_matrix, threshold:float=0.25):
    """
    Compute strength of connections C_{ij} for sparse CSR matrix (slightly different than above strength and strength^T version)
    produces only one matrix

    comes from this paper on aggregation
    https://epubs.siam.org/doi/epdf/10.1137/110838844
    """

    assert sp.isspmatrix_csr(A)

    # strength of connection graph
    nnodes = A.shape[0]
    
    # get diagonals
    diags = []
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            if i == j:
                diags += [np.abs(A.data[jp])]
    
    STRENGTH = [[] for i in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            value = np.abs(A.data[jp])
            lb = threshold * np.sqrt(diags[i] * diags[j])
            if value >= lb:
                STRENGTH[i] += [j]
    # can be converted into CSR style NZ pattern (though has no float data, just NZ pattern C_{ij})
    return STRENGTH


def greedy_serial_aggregation_csr(A:sp.csr_matrix, threshold:float=0.25):
    """from paper https://epubs.siam.org/doi/epdf/10.1137/110838844, with parallel versions discussed also
    and GPU version here https://www.sciencedirect.com/science/article/pii/S0898122114004143"""

    # greedy serial aggregation
    assert sp.isspmatrix_csr(A)
    nnodes = A.shape[0]
    STRENGTH = strength_matrix_csr(A, threshold)
    # print(f"{STRENGTH=}")

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

def plot_plate_aggregation(aggregate_ind, nx, ny, Lx, Ly):
    """plot aggregation groups with color + numbers"""
    num_agg = np.max(aggregate_ind) + 1

    # plot color aggregates
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Xf = X.ravel()
    Yf = Y.ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    # optional: still color by aggregate
    colors = plt.cm.jet(np.linspace(0.0, 1.0, num_agg + 1))

    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        # print(f"{agg_mask=}")
        # print(f"{Xf[:10]=}\n{Yf[:10]=}")
        ax.scatter(
            Xf[agg_mask], Yf[agg_mask],
            color=colors[iagg], s=120, edgecolors="k"
        )

    # write aggregate index inside each node
    for i in range(len(Xf)):
        ax.text(
            Xf[i], Yf[i],
            str(int(aggregate_ind[i])),
            ha="center", va="center",
            fontsize=8, color="white", weight="bold"
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Greedy serial aggregation (aggregate id per node)")

    plt.show()

def tentative_prolongator_csr(aggregate_ind:np.ndarray):
    # construct tentative prolongator T such that T * Bc = B and T^T T = I
    # however first I'm just following instructions from energy minimization, https://link.springer.com/article/10.1007/s006070050022 paper
    # for scalar PDEs
    nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    # construct CSR sparsity pattern.. first construct rowp
    row_cts = np.zeros(nnodes, dtype=np.int32)
    for iagg in range(num_agg):
        row_cts[aggregate_ind == iagg] += 1
    # print(f"{row_cts=}")
    # then build rowp
    rowp = np.zeros(nnodes + 1, dtype=np.int32)
    for i in range(nnodes):
        rowp[i+1] = rowp[i] + row_cts[i]
    nnz = rowp[-1]
    # print(f"{rowp=}")
    # could do this simpler way since each row only contains one NZ in this way..
    cols = np.zeros(nnz, dtype=np.int32)
    assert nnz == nnodes # assumption to make cols part easier
    for iagg in range(num_agg):
        cols[aggregate_ind == iagg] = iagg
    
    data = np.ones(nnz, dtype=np.double)
    return sp.csr_matrix((data, cols, rowp), shape=(nnodes, num_agg))

def spectral_radius_DinvA_csr(A: sp.spmatrix, maxiter=200, tol=1e-8):
    """
    Estimate spectral radius of D^{-1} A for a CSR matrix A.
    Uses a matrix-free LinearOperator.

    Parameters
    ----------
    A : scipy.sparse matrix (CSR preferred)
    maxiter : int
    tol : float

    Returns
    -------
    rho : float
        Estimated spectral radius
    """
    if not sp.sparse.isspmatrix_csr(A):
        A = A.tocsr()

    n = A.shape[0]

    # Diagonal inverse
    D = A.diagonal()
    if np.any(D == 0.0):
        raise ValueError("Zero diagonal entry in A")

    Dinv = 1.0 / D

    def matvec(x):
        return Dinv * (A @ x)

    M = spla.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=A.dtype
    )

    # Largest magnitude eigenvalue
    eigval = spla.eigs(
        M,
        k=1,
        which="LM",
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=False
    )

    return np.abs(eigval[0])

def get_bc_flags(A: sp.csr_matrix, tol=1e-14) -> np.ndarray:
    """Return True for constrained DOFs in CSR matrix A"""
    return np.array([
        # row i
        np.all(np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] != i]) < tol)  # off-diagonals ~ 0
        and np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] == i][0] - 1.0) < tol  # diagonal ~1
        for i in range(A.shape[0])
    ], dtype=bool)


def scalar_orthog_projector(dP:sp.csr_matrix, bc_flags):
    """scalar orthogonal projector from energy opt paper https://link.springer.com/article/10.1007/s006070050022"""
    # compute and subtract away row-sums from dP the update to prolongation matrix
    for i in range(dP.shape[0]):
        if bc_flags[i]: continue
        row_ips = np.arange(dP.indptr[i], dP.indptr[i+1])
        row_vals = dP.data[row_ips]
        dP.data[row_ips] -= np.mean(row_vals)
    return dP



def smooth_prolongator_csr(T:sp.csr_matrix, A:sp.csr_matrix, bc_flags:np.ndarray, omega:float=0.7, near_kernel:bool=True):
    # single step jacobi smoothing of tentative prolongator
    # now have P = (I - omega * Dinv * A) * T where T is tentative and P is smoothed prolong
    # this introduces some fillin from T to P
    Dinv = 1.0 / A.diagonal()
    AT = A @ T
    if near_kernel:
        AT = scalar_orthog_projector(AT, bc_flags)
        AT = AT.tocsr()
    DAT = AT.multiply(Dinv[:, None])
    newP = T - omega * DAT
    return newP

# def galerkin_coarse_grid_csr(R:sp.csr_matrix, A_fine:sp.csr_matrix, P:sp.csr_matrix):
#     return R @ A_fine @ P    
    

class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x


from scipy.sparse.csgraph import reverse_cuthill_mckee

def rcm_sort_fcn(A_free: sp.csr_matrix, A: sp.csr_matrix):
    """
    Symmetric Reverse Cuthill-McKee reorder of CSR matrices.

    Returns
    -------
    A_free_rcm, A_rcm, perm
    where
        A_rcm = A[perm,:][:,perm]
    """
    assert sp.isspmatrix_csr(A_free)
    assert sp.isspmatrix_csr(A)
    assert A_free.shape == A.shape
    assert A.shape[0] == A.shape[1]

    perm = reverse_cuthill_mckee(A_free, symmetric_mode=True)
    perm = np.asarray(perm, dtype=np.int32)

    A_free_rcm = A_free[perm, :][:, perm].tocsr()
    A_rcm = A[perm, :][:, perm].tocsr()

    return A_free_rcm, A_rcm, perm

from dataclasses import dataclass


@dataclass
class MASWLevel:
    level: int
    A_free: sp.csr_matrix
    A: sp.csr_matrix

    # stored ASW / block-Jacobi inverse data
    block_inv: list[np.ndarray]
    block_ranges: list[tuple[int, int]]

    # one-level transfer operators: coarse -> current
    P: sp.csr_matrix = None
    R: sp.csr_matrix = None

    # global transfer operators: level -> finest
    P_global: sp.csr_matrix = None
    R_global: sp.csr_matrix = None


def build_asw_block_inverses(A: sp.csr_matrix, asw_sd_size: int):
    """
    Build dense block-Jacobi inverses from contiguous diagonal blocks of A.

    Parameters
    ----------
    A : csr_matrix
        Square system matrix on this level.
    asw_sd_size : int
        Block size in DOFs. For example, if asw_sd_size = 32, then blocks are
        [0:32], [32:64], [64:96], ...

    Returns
    -------
    block_inv : list[np.ndarray]
        Dense inverses of each diagonal block.
    block_ranges : list[(int,int)]
        Half-open DOF index ranges (start, end) for each block.
    """
    assert sp.isspmatrix_csr(A)
    assert A.shape[0] == A.shape[1]
    assert asw_sd_size >= 1

    n = A.shape[0]
    block_inv = []
    block_ranges = []

    start = 0
    while start < n:
        end = min(start + asw_sd_size, n)

        Ablock = A[start:end, start:end].toarray()
        block_inv.append(np.linalg.inv(Ablock))
        block_ranges.append((start, end))

        start = end

    return block_inv, block_ranges


def apply_asw_blocks(rhs: np.ndarray, block_inv: list[np.ndarray],
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


class MASW_Solver:
    """
    Fixed-level multilevel additive Schwarz solver.

    Main idea
    ---------
    Build all levels up front for a user-prescribed number of levels.

    For each coarsening step:
      1. Optional Morton reorder
      2. Build/store ASW block inverses on that level
      3. Build prolongator P and restriction R = P^T
      4. Build next coarse matrix
      5. Build/store cumulative P_global and R_global

    Apply preconditioner:
        z = M_0^{-1} rhs
          + P_1 M_1^{-1} P_1^T rhs
          + P_2 M_2^{-1} P_2^T rhs + ...
    where P_l means prolongation from level l to finest.
    """

    def __init__(self,
                 A_free: sp.csr_matrix,
                 A: sp.csr_matrix,
                 num_levels: int,
                 asw_sd_size: int,
                 threshold: float = 0.25,
                 omega: float = 0.7,
                 near_kernel: bool = True,
                 coarsening_fcn=None,
                 smooth_prolongator: bool = True,
                 nsmooth:int=2,
                 omegaJac:float=0.7,
                 node_sort_fcn=None,
                 galerkin_fcn=None):
        """
        Parameters
        ----------
        A_free : csr_matrix
            Unconstrained fine-grid operator for coarsening indicators.
        A : csr_matrix
            Constrained fine-grid operator used in the preconditioner.
        num_levels : int
            Total number of levels, including finest level.
        asw_sd_size : int
            Dense block size in DOFs for each ASW / block-Jacobi block.
        threshold : float
            Coarsening threshold.
        omega : float
            Damping for prolongation smoothing.
        near_kernel : bool
            Passed to smooth_prolongator_csr.
        coarsening_fcn : callable
            Function like aggregate_ind = coarsening_fcn(A_free, threshold=threshold).
        smooth_prolongator : bool
            Whether to smooth the tentative prolongator.
        morton_sort_fcn : callable or None
            Optional reordering function:
                A_free_sorted, A_sorted, perm = morton_sort_fcn(A_free, A)
            If None, matrices are left unchanged.
        galerkin_fcn : callable or None
            Optional coarse-grid construction:
                Ac = galerkin_fcn(A, P, R)
            If None, uses R @ (A @ P).
        """
        assert sp.isspmatrix_csr(A_free)
        assert sp.isspmatrix_csr(A)
        assert A.shape[0] == A.shape[1]
        assert A_free.shape == A.shape
        assert num_levels >= 1
        assert asw_sd_size >= 1

        self.A_free = A_free.tocsr()
        self.A = A.tocsr()

        self.num_levels_requested = num_levels
        self.asw_sd_size = asw_sd_size
        self.threshold = threshold
        self.omega = omega
        self.near_kernel = near_kernel
        self.coarsening_fcn = coarsening_fcn
        self.smooth_prolongator = smooth_prolongator
        self.node_sort_fcn = node_sort_fcn
        self.galerkin_fcn = galerkin_fcn

        self.nsmooth = nsmooth
        self.omegaJac = omegaJac

        self.levels: list[MASWLevel] = []

        self._build_hierarchy()

        print(f"Multilevel ASW with {self.num_levels=} and {self.operator_complexity=:.3f}")
        print(f"\tnum DOF per level = {self.num_dof_list}")

    def _build_hierarchy(self):
        """
        Build all levels up front.
        """
        A_free_l = self.A_free
        A_l = self.A

        # finest level -> finest level
        P_global_l = sp.eye(A_l.shape[0], format="csr")
        R_global_l = sp.eye(A_l.shape[0], format="csr")

        for level in range(self.num_levels_requested):
            # --------------------------------------------------
            # 1) Optional Morton or RCM sorting / reordering
            # --------------------------------------------------
            if self.node_sort_fcn is not None:
                out = self.node_sort_fcn(A_free_l, A_l)
                if len(out) == 3:
                    A_free_sorted, A_sorted, _perm = out
                else:
                    A_free_sorted, A_sorted = out
            else:
                A_free_sorted = A_free_l
                A_sorted = A_l

            A_free_sorted = A_free_sorted.tocsr()
            A_sorted = A_sorted.tocsr()

            plt.spy(A_sorted)
            plt.show()

            # --------------------------------------------------
            # 2) Build/store ASW block inverses on this level
            # --------------------------------------------------
            block_inv, block_ranges = build_asw_block_inverses(
                A_sorted, self.asw_sd_size
            )

            lev = MASWLevel(
                level=level,
                A_free=A_free_sorted,
                A=A_sorted,
                block_inv=block_inv,
                block_ranges=block_ranges,
                P=None,
                R=None,
                P_global=P_global_l,
                R_global=R_global_l,
            )
            self.levels.append(lev)

            # stop at final requested level
            if level == self.num_levels_requested - 1:
                break

            # --------------------------------------------------
            # 3) Build P and R for next coarsening step
            # --------------------------------------------------
            if self.coarsening_fcn is None:
                raise ValueError("coarsening_fcn must be provided for multilevel construction")

            aggregate_ind = self.coarsening_fcn(A_free_sorted, threshold=self.threshold)
            T = tentative_prolongator_csr(aggregate_ind)

            if self.smooth_prolongator:
                bc_flags = get_bc_flags(A_sorted)
                P_l = smooth_prolongator_csr(
                    T,
                    A_sorted,
                    bc_flags,
                    omega=self.omega,
                    near_kernel=self.near_kernel,
                ).tocsr()
            else:
                P_l = T.tocsr()

            R_l = P_l.T.tocsr()

            # coarse -> current
            self.levels[level].P = P_l
            self.levels[level].R = R_l

            # --------------------------------------------------
            # 4) Build next coarse matrix
            # --------------------------------------------------
            if self.galerkin_fcn is None:
                A_free_next = (R_l @ (A_free_sorted @ P_l)).tocsr()
                A_next = (R_l @ (A_sorted @ P_l)).tocsr()
            else:
                A_free_next = self.galerkin_fcn(A_free_sorted, P_l, R_l).tocsr()
                A_next = self.galerkin_fcn(A_sorted, P_l, R_l).tocsr()

            # --------------------------------------------------
            # 5) Build cumulative/global transfers
            #
            # P_l maps coarse_{l+1} -> current_l
            # P_global_l maps current_l -> fine_0
            # so next global prolongator is:
            #   P_global_{l+1} = P_global_l @ P_l
            #
            # Similarly:
            #   R_global_{l+1} = P_global_{l+1}^T
            # --------------------------------------------------
            P_global_l = (P_global_l @ P_l).tocsr()
            R_global_l = P_global_l.T.tocsr()

            A_free_l = A_free_next
            A_l = A_next

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def num_dof_list(self) -> list[int]:
        return [lev.A.shape[0] for lev in self.levels]

    @property
    def nnz_list(self) -> list[int]:
        return [lev.A.nnz for lev in self.levels]

    @property
    def total_nnz(self) -> int:
        return sum(lev.A.nnz for lev in self.levels)

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.levels[0].A.nnz

    def apply_level_asw(self, level_id: int, rhs_level: np.ndarray) -> np.ndarray:
        """
        Apply stored ASW block-Jacobi inverse on one level.
        """
        lev = self.levels[level_id]
        return apply_asw_blocks(rhs_level, lev.block_inv, lev.block_ranges)

    def single_solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply multilevel additive Schwarz preconditioner.

        Returns
        -------
        z = M^{-1} rhs

        with
            M^{-1} = M_0^{-1}
                   + P_1 M_1^{-1} P_1^T
                   + P_2 M_2^{-1} P_2^T + ...

        where P_l is the cumulative/global prolongator from level l to finest.
        """
        rhs = np.asarray(rhs)
        assert rhs.ndim == 1
        assert rhs.shape[0] == self.levels[0].A.shape[0]

        z = np.zeros_like(rhs)

        for lev in self.levels:
            if lev.level == 0:
                # fine-level additive correction
                z += apply_asw_blocks(rhs, lev.block_inv, lev.block_ranges)
            else:
                # restrict to level l
                rhs_l = lev.R_global @ rhs

                # apply ASW inverse on level l
                z_l = apply_asw_blocks(rhs_l, lev.block_inv, lev.block_ranges)

                # prolong back to fine level
                z += lev.P_global @ z_l

        return z
    
    def solve(self, rhs: np.ndarray):
        """
        Apply self.nsmooth defect-correction smoothing steps using the
        multilevel additive Schwarz operator.

        Iteration:
            x_{k+1} = x_k + omega * M^{-1} (rhs - A x_k)
        """
        rhs = np.asarray(rhs)
        assert rhs.ndim == 1
        assert rhs.shape[0] == self.levels[0].A.shape[0]

        A = self.levels[0].A
        omega = self.omegaJac

        x = np.zeros_like(rhs)

        for _ in range(self.nsmooth):
            r = rhs - A @ x

            z = np.zeros_like(rhs)
            for lev in self.levels:
                if lev.level == 0:
                    z += apply_asw_blocks(r, lev.block_inv, lev.block_ranges)
                else:
                    r_l = lev.R_global @ r
                    z_l = apply_asw_blocks(r_l, lev.block_inv, lev.block_ranges)
                    z += lev.P_global @ z_l

            x += omega * z
        return x
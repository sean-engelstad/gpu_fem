
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

    # one-level transfer operators: coarse -> current
    P: sp.csr_matrix = None
    R: sp.csr_matrix = None

    # global transfer operators: level -> finest/original
    P_global: sp.csr_matrix = None

class MASW_BSRSolver_V1:
    """
    Fixed-level multilevel additive Schwarz solver for BSR matrices.

    Main idea
    ---------
    Build all levels up front for a user-prescribed number of levels.

    For each coarsening step:
      1. Optional node sorting / reordering
      2. Build/store ASW block inverses on that level
      3. Build BSR tentative prolongator T and energy-smoothed prolongator P
      4. Build next coarse matrix
      5. Build/store cumulative P_global and R_global

    Apply preconditioner:
        z = M_0^{-1} rhs
          + P_1 M_1^{-1} P_1^T rhs
          + P_2 M_2^{-1} P_2^T rhs + ...
    where P_l means prolongation from level l to finest.
    """

    def __init__(self,
                 A_free: sp.bsr_matrix,
                 A: sp.bsr_matrix,
                 B: np.ndarray,
                 num_levels: int,
                 asw_sd_size: int,
                 threshold: float = 0.25,
                 omega: float = 0.7,
                 near_kernel: bool = True,
                 coarsening_fcn=None,
                 smooth_prolongator: bool = True,
                 nsmooth: int = 2,
                 omegaJac: float = 0.7,
                 node_sort_fcn=None,
                 galerkin_fcn=None):
        """
        Parameters
        ----------
        A_free : bsr_matrix
            Unconstrained fine-grid operator for coarsening indicators.
        A : bsr_matrix
            Constrained fine-grid operator used in the preconditioner.
        B : ndarray
            Fine-grid near-kernel modes / RBMs in scalar-DOF form.
        num_levels : int
            Total number of levels, including finest level.
        asw_sd_size : int
            Dense block size in scalar DOFs for each ASW / block-Jacobi block.
        threshold : float
            Coarsening threshold.
        omega : float
            Damping for prolongation smoothing.
        near_kernel : bool
            Passed to smooth_prolongator_bsr.
        coarsening_fcn : callable or None
            Function like:
                aggregate_ind = coarsening_fcn(A_free, threshold=threshold)
            If None, greedy_serial_aggregation_bsr is used.
        smooth_prolongator : bool
            Whether to smooth the tentative prolongator.
        nsmooth : int
            Number of defect-correction smoothing steps in solve().
        omegaJac : float
            Damping for outer defect-correction iteration.
        node_sort_fcn : callable or None
            Optional reordering function:
                A_free_sorted, A_sorted, B_sorted, perm = node_sort_fcn(A_free, A, B)
            or
                A_free_sorted, A_sorted, B_sorted = node_sort_fcn(A_free, A, B)
        galerkin_fcn : callable or None
            Optional coarse-grid construction:
                Ac = galerkin_fcn(A, P, R)
            If None, uses R @ (A @ P).
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)
        assert A.shape[0] == A.shape[1]
        assert A_free.shape == A.shape
        assert num_levels >= 1
        assert asw_sd_size >= 1

        self.A_free = A_free.tobsr()
        self.A = A.tobsr()
        self.B = np.asarray(B)

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

        self.nsmooth = nsmooth
        self.omegaJac = omegaJac
        self.iters = 0

        self.levels: list[MASWLevel_BSR] = []

        self._build_hierarchy()

        print(f"Multilevel ASW-BSR with {self.num_levels=} and {self.operator_complexity=:.3f}")
        print(f"\tnum DOF per level = {self.num_dof_list}")
        print(f"\tnum nodes per level = {self.num_nodes_list}")

    def _build_hierarchy(self):
        """
        Build all levels up front.
        """
        A_free_l = self.A_free
        A_l = self.A
        B_l = self.B
        block_dim = self.block_dim
        block_shape = (block_dim, block_dim)

        # finest level -> finest level
        P_global_l = sp.eye(A_l.shape[0], format="csr")
        R_global_l = sp.eye(A_l.shape[0], format="csr")

        for level in range(self.num_levels_requested):
            # --------------------------------------------------
            # 1) Optional sorting / reordering
            # --------------------------------------------------
            if self.node_sort_fcn is not None:
                out = self.node_sort_fcn(A_free_l, A_l, B_l)
                if len(out) == 4:
                    A_free_sorted, A_sorted, B_sorted, _perm = out
                else:
                    A_free_sorted, A_sorted, B_sorted = out
            else:
                A_free_sorted = A_free_l
                A_sorted = A_l
                B_sorted = B_l

            A_free_sorted = A_free_sorted.tobsr(block_shape)
            A_sorted = A_sorted.tobsr(block_shape)
            B_sorted = np.asarray(B_sorted)

            print(f"{A_sorted.data.shape=}")

            # --------------------------------------------------
            # 2) Build/store ASW block inverses on this level
            # --------------------------------------------------
            block_inv, block_ranges = build_asw_block_inverses_bsr(
                A_sorted, self.asw_sd_size
            )

            lev = MASWLevel_BSR(
                level=level,
                A_free=A_free_sorted,
                A=A_sorted,
                B=B_sorted,
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
            # print(f"{A_free_sorted.shape=}")
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
            # print(f"{block_dim=}")

            # print(f"{level=} {self.B.shape=} {B_sorted.shape=} {aggregate_ind=} {aggregate_ind.shape=}")

            T_l, Bc_l = tentative_prolongator_bsr(
                B_sorted,
                aggregate_ind,
                bc_flags,
                vpn=block_dim,
            )

            # print(f"{T_l.shape=}")

            if self.smooth_prolongator:
                P_l = smooth_prolongator_bsr(
                    T_l,
                    A_sorted,
                    Bc_l,
                    bc_flags,
                    omega=self.omega,
                    near_kernel=self.near_kernel,
                )
            else:
                P_l = T_l

            P_l = P_l.tocsr()
            R_l = P_l.T.tocsr()

            # coarse -> current
            self.levels[level].P = P_l
            self.levels[level].R = R_l

            # --------------------------------------------------
            # 4) Build next coarse matrix
            # --------------------------------------------------
            if self.galerkin_fcn is None:
                A_free_next = (R_l @ (A_free_sorted @ P_l)).tobsr(block_shape)
                A_next = (R_l @ (A_sorted @ P_l)).tobsr(block_shape)
            else:
                A_free_next = self.galerkin_fcn(A_free_sorted, P_l, R_l).tobsr(block_shape)
                A_next = self.galerkin_fcn(A_sorted, P_l, R_l).tobsr(block_shape)

            # --------------------------------------------------
            # 5) Build cumulative/global transfers
            # --------------------------------------------------
            P_global_l = (P_global_l @ P_l).tocsr()
            R_global_l = P_global_l.T.tocsr()

            A_free_l = A_free_next
            A_l = A_next
            B_l = Bc_l

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

    def apply_level_asw(self, level_id: int, rhs_level: np.ndarray) -> np.ndarray:
        """
        Apply stored ASW block-Jacobi inverse on one level.
        """
        lev = self.levels[level_id]
        return apply_asw_blocks_bsr(rhs_level, lev.block_inv, lev.block_ranges)

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
                z += apply_asw_blocks_bsr(rhs, lev.block_inv, lev.block_ranges)
            else:
                rhs_l = lev.R_global @ rhs
                z_l = apply_asw_blocks_bsr(rhs_l, lev.block_inv, lev.block_ranges)
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
        assert rhs.shape[0] == self.levels[0].A.shape[0]

        A = self.levels[0].A
        omega = self.omegaJac

        x = np.zeros_like(rhs)

        self.iters += self.nsmooth

        for _ in range(self.nsmooth):
            r = rhs - A @ x

            z = np.zeros_like(rhs)
            for lev in self.levels:
                if lev.level == 0:
                    z += apply_asw_blocks_bsr(r, lev.block_inv, lev.block_ranges)
                else:
                    r_l = lev.R_global @ r
                    z_l = apply_asw_blocks_bsr(r_l, lev.block_inv, lev.block_ranges)
                    z += lev.P_global @ z_l

            x += omega * z

        return x

    R_global: sp.csr_matrix = None


class MASW_BSRSolver_V2:
    """
    Fixed-level multilevel additive Schwarz solver for BSR matrices.

    Main idea
    ---------
    Build all levels up front for a user-prescribed number of levels.

    For each coarsening step:
      1. Optional Morton sorting of nodes using xpts
      2. Build/store ASW block inverses on that level
      3. Build BSR tentative prolongator T and energy-smoothed prolongator P
      4. Average xpts over aggregates to define coarse node coordinates
      5. Build next coarse matrix
      6. Build/store cumulative P_global and R_global

    Apply preconditioner:
        z = M_0^{-1} rhs
          + P_1 M_1^{-1} P_1^T rhs
          + P_2 M_2^{-1} P_2^T rhs + ...
    where P_l means prolongation from level l to finest/original ordering.
    """

    def __init__(self,
                 A_free: sp.bsr_matrix,
                 A: sp.bsr_matrix,
                 B: np.ndarray,
                 xpts: np.ndarray,
                 num_levels: int,
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
                 morton_sort: bool = True):
        """
        Parameters
        ----------
        A_free : bsr_matrix
            Unconstrained fine-grid operator for coarsening indicators.
        A : bsr_matrix
            Constrained fine-grid operator used in the preconditioner.
        B : ndarray
            Fine-grid near-kernel modes / RBMs in scalar-DOF form, shape (ndof, nmodes).
        xpts : ndarray
            Fine-grid nodal coordinates, shape (nnodes, ndim).
        num_levels : int
            Total number of levels, including finest level.
        asw_sd_size : int
            Dense block size in scalar DOFs for each ASW / block-Jacobi block.
        threshold : float
            Coarsening threshold.
        omega : float
            Damping for prolongation smoothing.
        near_kernel : bool
            Passed to smooth_prolongator_bsr.
        coarsening_fcn : callable or None
            Function like:
                aggregate_ind = coarsening_fcn(A_free, threshold=threshold)
            If None, greedy_serial_aggregation_bsr is used.
        smooth_prolongator : bool
            Whether to smooth the tentative prolongator.
        nsmooth : int
            Number of defect-correction smoothing steps in solve().
        omegaJac : float
            Damping for outer defect-correction iteration.
        node_sort_fcn : callable or None
            Optional node permutation function:
                perm_nodes = node_sort_fcn(xpts)
            If None and morton_sort=True, Morton sorting is used.
        galerkin_fcn : callable or None
            Optional coarse-grid construction:
                Ac = galerkin_fcn(A, P, R)
            If None, uses R @ (A @ P).
        morton_sort : bool
            Whether to Morton-sort nodes on each level.
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)
        assert A.shape[0] == A.shape[1]
        assert A_free.shape == A.shape
        assert num_levels >= 1
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

    # -------------------------------------------------------------------------
    # Morton helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _spread_bits_2d(v: np.ndarray) -> np.ndarray:
        """
        Spread lower 32 bits into even bit positions for Morton coding.
        """
        x = v.astype(np.uint64)
        x = (x | (x << 16)) & np.uint64(0x0000FFFF0000FFFF)
        x = (x | (x << 8))  & np.uint64(0x00FF00FF00FF00FF)
        x = (x | (x << 4))  & np.uint64(0x0F0F0F0F0F0F0F0F)
        x = (x | (x << 2))  & np.uint64(0x3333333333333333)
        x = (x | (x << 1))  & np.uint64(0x5555555555555555)
        return x

    @classmethod
    def morton_perm_from_xpts(cls, xpts: np.ndarray, nbits: int = 16) -> np.ndarray:
        """
        Compute a Morton/Z-order node permutation from nodal coordinates.

        Nearby nodes in physical space tend to get nearby indices.
        """
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

        # stable tie break by x then y
        perm = np.lexsort((xy[:, 1], xy[:, 0], codes))
        return perm.astype(np.int64)

    @staticmethod
    def node_perm_to_dof_perm(perm_nodes: np.ndarray, block_dim: int) -> np.ndarray:
        """
        Expand a node permutation into a scalar-DOF permutation.
        Assumes node-major ordering with block_dim DOFs per node.
        """
        perm_nodes = np.asarray(perm_nodes, dtype=np.int64)
        return (perm_nodes[:, None] * block_dim + np.arange(block_dim, dtype=np.int64)[None, :]).ravel()

    @staticmethod
    def perm_matrix_from_perm(perm: np.ndarray) -> sp.csr_matrix:
        """
        Build permutation matrix S such that
            x_sorted = S^T x_old = x_old[perm]
            A_sorted = S^T A_old S = A_old[perm][:, perm]
        """
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
        """
        Reorder one level by a node permutation.
        """
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
        """
        Compute coarse node coordinates by averaging fine node coordinates in each aggregate.
        """
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

        for level in range(self.num_levels_requested):
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

                # update global maps because this level is now internally sorted
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
            # 2) Build/store ASW block inverses on this level
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
            # 4) Average xpts over aggregates for coarse nodes
            # --------------------------------------------------
            xpts_c_unsorted = self.average_xpts_by_aggregate(xpts_sorted, aggregate_ind)

            # --------------------------------------------------
            # 5) Build next coarse matrix in unsorted aggregate order
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
            # 6) Sort coarse level by Morton order and update P_l
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
            # 7) Build cumulative/global transfers
            # --------------------------------------------------
            P_global_l = (P_global_l @ P_l).tocsr()
            R_global_l = P_global_l.T.tocsr()

            A_free_l = A_free_next
            A_l = A_next
            B_l = B_next
            xpts_l = xpts_next

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
        """
        Apply stored ASW block-Jacobi inverse on one level.
        """
        lev = self.levels[level_id]
        return apply_asw_blocks_bsr(rhs_level, lev.block_inv, lev.block_ranges)

    def single_solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply multilevel additive Schwarz preconditioner.

        Returns
        -------
        z = M^{-1} rhs
        """
        rhs = np.asarray(rhs)
        assert rhs.ndim == 1
        assert rhs.shape[0] == self.levels[0].P_global.shape[0]

        z = np.zeros_like(rhs)

        for lev in self.levels:
            rhs_l = lev.R_global @ rhs
            z_l = apply_asw_blocks_bsr(rhs_l, lev.block_inv, lev.block_ranges)
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
                z_l = apply_asw_blocks_bsr(r_l, lev.block_inv, lev.block_ranges)
                z += lev.P_global @ z_l

            x += omega * z

        return x
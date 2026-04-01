import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import sys
sys.path.append("../1_sa_amg/_src/")
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


def greedy_rn_serial_aggregation_csr(A: sp.csr_matrix, C_nodes: np.ndarray, threshold: float = 0.25):
    """Greedy root-node serial aggregation.
    
    Unassigned nodes may only be added to aggregates whose ROOT NODE
    is a strong connection of that node.
    """

    assert sp.isspmatrix_csr(A)
    nnodes = A.shape[0]
    STRENGTH = strength_matrix_csr(A, threshold)

    # aggregate_ind[i] = aggregate id for node i, or -1 if unassigned
    aggregate_ind = np.full(nnodes, -1, dtype=int)
    aggregate_groups = []

    # list of root/coarse nodes
    C_node_list = [i for i in range(nnodes) if C_nodes[i]]

    # root_of_agg[k] = root node for aggregate k
    root_of_agg = []

    # initialize one aggregate per root node
    for iagg, root_node in enumerate(C_node_list):
        # print(f"{root_node=}")
        aggregate_groups.append([root_node])
        aggregate_ind[root_node] = iagg
        root_of_agg.append(root_node)

    # second phase: add all remaining nodes to nearby aggregates,
    # but ONLY if the aggregate's root is a strong neighbor of the node
    for i in range(nnodes):
        if aggregate_ind[i] != -1:
            continue  # already assigned

        strong_neighbors = np.array(STRENGTH[i], dtype=int)

        # candidate aggregates are only those whose root node is a strong neighbor of i
        candidate_aggs = []
        for iagg, root_node in enumerate(root_of_agg):
            if root_node in strong_neighbors:
                candidate_aggs.append(iagg)

        if len(candidate_aggs) == 0:
            continue  # leave unassigned for now

        # among eligible aggregates, choose the smallest one
        candidate_sizes = np.array([len(aggregate_groups[iagg]) for iagg in candidate_aggs])
        min_size_ind = np.argmin(candidate_sizes)
        agg_ind = candidate_aggs[min_size_ind]

        aggregate_groups[agg_ind].append(i)
        aggregate_ind[i] = agg_ind

    # second phase adds all remaining nodes to nearby aggregates
    for i in range(nnodes):
        if aggregate_ind[i] != -1: continue # only look at unpicked nodes
        strong_neighbors = np.array(STRENGTH[i])
        nb_agg_ind = aggregate_ind[strong_neighbors]
        valid_nb_agg_ind = nb_agg_ind[nb_agg_ind != -1]
        # sweep into an aggregate with smallest size
        nb_agg_sizes = np.array([len(aggregate_groups[ind]) for ind in valid_nb_agg_ind])
        if nb_agg_sizes.shape[0] == 0: continue
        min_size_ind = np.argmin(nb_agg_sizes)
        agg_ind = valid_nb_agg_ind[min_size_ind]

        # now add into that aggregate
        aggregate_groups[agg_ind] += [i]
        aggregate_ind[i] = agg_ind

    # # if this fails, some nodes were not strongly connected to any root node
    # assert not np.any(aggregate_ind == -1), \
    #     "Some nodes could not be assigned: not strongly connected to any root node."

    return aggregate_ind

def plot_plate_aggregation(aggregate_ind, C_nodes, nx, ny, Lx, Ly):
    """plot aggregation groups with color + numbers, highlight coarse nodes"""

    num_agg = np.max(aggregate_ind) + 1

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Xf = X.ravel()
    Yf = Y.ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, num_agg + 1))

    # --- plot all nodes colored by aggregate ---
    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        ax.scatter(
            Xf[agg_mask], Yf[agg_mask],
            color=colors[iagg], s=120, edgecolors="k"
        )

    # --- overlay coarse nodes (root nodes) in black ---
    coarse_mask = C_nodes.astype(bool)

    ax.scatter(
        Xf[coarse_mask], Yf[coarse_mask],
        color="black",
        s=140,               # slightly bigger
        edgecolors="white",  # contrast edge
        linewidths=1.5,
        zorder=3             # draw on top
    )

    # --- write aggregate index ---
    for i in range(len(Xf)):
        is_coarse = coarse_mask[i]

        ax.text(
            Xf[i], Yf[i],
            str(int(aggregate_ind[i])),
            ha="center", va="center",
            fontsize=8,
            color="white" if is_coarse else "white",
            weight="bold",
            zorder=4
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Aggregation (root nodes highlighted)")

    plt.show()

# def tentative_prolongator_csr(aggregate_ind: np.ndarray, C_nodes: np.ndarray):
#     """
#     Construct a scalar root-node AMG tentative prolongator T.

#     For each aggregate:
#       - the coarse/root node (C-node) is injected 1-to-1 into its own coarse DOF
#       - only the F-nodes in that aggregate participate in the local QR construction

#     Current scalar start:
#       - near-nullspace is the constant vector
#       - QR on F-nodes only reduces to normalizing a vector of ones over the F-nodes
#         in each aggregate

#     Parameters
#     ----------
#     aggregate_ind : (nnodes,) int ndarray
#         aggregate_ind[i] = aggregate index of fine node i
#     C_nodes : (nnodes,) bool ndarray
#         C_nodes[i] = True if node i is a coarse/root node

#     Returns
#     -------
#     T : scipy.sparse.csr_matrix, shape (nnodes, num_agg)
#         Tentative prolongator
#     """
#     nnodes = aggregate_ind.shape[0]
#     assert C_nodes.shape == (nnodes,)
#     assert C_nodes.dtype == bool

#     num_agg = np.max(aggregate_ind) + 1

#     # Root-node AMG assumption for this starter version:
#     # exactly one C-node per aggregate.
#     agg_root = np.full(num_agg, -1, dtype=np.int32)

#     for inode in range(nnodes):
#         if C_nodes[inode]:
#             iagg = aggregate_ind[inode]
#             if agg_root[iagg] != -1:
#                 raise ValueError(
#                     f"Aggregate {iagg} has more than one C-node/root node"
#                 )
#             agg_root[iagg] = inode

#     if np.any(agg_root < 0):
#         missing = np.where(agg_root < 0)[0]
#         raise ValueError(
#             f"Each aggregate must contain exactly one C-node. Missing roots in aggregates {missing}"
#         )

#     # Count one nz per row in scalar root-node tentative prolongator
#     rowp = np.arange(nnodes + 1, dtype=np.int32)
#     cols = np.empty(nnodes, dtype=np.int32)
#     data = np.empty(nnodes, dtype=np.double)

#     # Precompute F-node counts per aggregate
#     F_counts = np.zeros(num_agg, dtype=np.int32)
#     for iagg in range(num_agg):
#         in_agg = (aggregate_ind == iagg)
#         F_counts[iagg] = np.count_nonzero(in_agg & (~C_nodes))

#     # Build rows
#     for inode in range(nnodes):
#         iagg = aggregate_ind[inode]
#         cols[inode] = iagg

#         if C_nodes[inode]:
#             # Pure injection on root/coarse nodes
#             data[inode] = 1.0
#         else:
#             # QR on F-nodes only for scalar constant mode:
#             # q = 1/sqrt(nF) * ones
#             nF = F_counts[iagg]
#             if nF <= 0:
#                 raise ValueError(
#                     f"Aggregate {iagg} has no F-nodes, but node {inode} is marked F."
#                 )
#             data[inode] = 1.0 / np.sqrt(nF)

#     T = sp.csr_matrix((data, cols, rowp), shape=(nnodes, num_agg))
#     return T

import numpy as np
import scipy.sparse as sp


def tentative_prolongator_csr(
    aggregate_ind: np.ndarray,
    C_nodes: np.ndarray,
    B: np.ndarray,
    Bc: np.ndarray = None,
):
    """
    Construct a scalar root-node AMG tentative prolongator T so that
    T * Bc reproduces the fine candidate B aggregate-by-aggregate.

    In this scalar / single-candidate case:
      - exactly one coarse DOF per aggregate
      - root/C-node row is injected exactly: T[root, agg] = 1
      - F-node rows are chosen so that T[:, agg] * Bc[agg] matches B in
        least-squares sense on the aggregate

    With one scalar coarse DOF per aggregate, the local LS fit is simply
        T[i, agg] = B[i] / Bc[agg]
    for all F-nodes in the aggregate.

    Parameters
    ----------
    aggregate_ind : (nnodes,) int ndarray
        aggregate_ind[i] = aggregate index of fine node i
    C_nodes : (nnodes,) bool ndarray
        C_nodes[i] = True if node i is the root/C-node of its aggregate
    B : (nnodes,) ndarray
        Fine-grid candidate vector to be reproduced
    Bc : (num_agg,) ndarray, optional
        Coarse candidate values. If None, set from root injection:
            Bc[agg] = B[root_of_agg]

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (nnodes, num_agg)
        Tentative prolongator satisfying T @ Bc = B exactly (up to FP roundoff)
        in this scalar one-mode case, assuming Bc[agg] != 0.
    Bc : (num_agg,) ndarray
        Coarse candidate vector used to define the fit
    """
    nnodes = aggregate_ind.shape[0]
    assert C_nodes.shape == (nnodes,)
    assert C_nodes.dtype == bool
    assert B.shape == (nnodes,)

    num_agg = np.max(aggregate_ind) + 1

    # exactly one root per aggregate
    agg_root = np.full(num_agg, -1, dtype=np.int32)

    for inode in range(nnodes):
        if C_nodes[inode]:
            iagg = aggregate_ind[inode]
            if agg_root[iagg] != -1:
                raise ValueError(
                    f"Aggregate {iagg} has more than one C-node/root node"
                )
            agg_root[iagg] = inode

    if np.any(agg_root < 0):
        missing = np.where(agg_root < 0)[0]
        raise ValueError(
            "Each aggregate must contain exactly one C-node. "
            f"Missing roots in aggregates {missing}"
        )

    # default coarse candidate from root injection
    if Bc is None:
        Bc = B[agg_root].copy()
    else:
        Bc = np.asarray(Bc, dtype=B.dtype)
        if Bc.shape != (num_agg,):
            raise ValueError(f"Bc must have shape ({num_agg},), got {Bc.shape}")

    # one nz per row
    rowp = np.arange(nnodes + 1, dtype=np.int32)
    cols = aggregate_ind.astype(np.int32).copy()
    data = np.empty(nnodes, dtype=np.double)

    for inode in range(nnodes):
        iagg = aggregate_ind[inode]

        if C_nodes[inode]:
            # exact root injection
            data[inode] = 1.0
        else:
            if abs(Bc[iagg]) < 1e-30:
                raise ValueError(
                    f"Bc[{iagg}] is zero (or too small), so cannot fit "
                    f"T[i,{iagg}] * Bc[{iagg}] = B[i] for scalar root-node AMG. "
                    "You need a different coarse candidate choice or more than one coarse DOF."
                )

            # scalar LS/exact-fit coefficient
            data[inode] = B[inode] / Bc[iagg]

    T = sp.csr_matrix((data, cols, rowp), shape=(nnodes, num_agg))
    return T, Bc


# def orthog_nullspace_projector_csr(
#     P: sp.csr_matrix,
#     Bc: np.ndarray,
#     bcs: np.ndarray,
# ):
#     """
#     Apply the orthogonal projector row-wise for a scalar CSR prolongator.
#     Ensures each unconstrained row is orthogonal to the coarse nullspace:
#         Pnew[i, :] @ Bc = 0

#     Parameters
#     ----------
#     P : sp.csr_matrix
#         Scalar CSR matrix of shape (nfine, ncoarse)
#     Bc : np.ndarray
#         Coarse nullspace modes, shape (ncoarse, nmodes)
#     bcs : np.ndarray
#         Boolean array of shape (nfine,), where True means constrained DOF

#     Returns
#     -------
#     Pnew : sp.csr_matrix
#         Projected CSR matrix
#     """

#     assert sp.isspmatrix_csr(P)
#     assert Bc.ndim == 2
#     assert bcs.shape[0] == P.shape[0]
#     assert Bc.shape[0] == P.shape[1]

#     Pnew = sp.csr_matrix(
#         (P.data.copy(), P.indices.copy(), P.indptr.copy()),
#         shape=P.shape
#     )

#     for irow in range(P.shape[0]):
#         start = P.indptr[irow]
#         end = P.indptr[irow + 1]

#         if start == end:
#             continue

#         cols = P.indices[start:end]
#         prow = Pnew.data[start:end]

#         # scalar mask: constrained row => nullspace removed entirely
#         Fi = 0.0 if bcs[irow] else 1.0

#         U = Bc[cols, :] * Fi   # shape (nnz_row, nmodes)

#         # PU = prow @ U, shape (nmodes,)
#         PU = prow @ U

#         # UTU = U^T U, shape (nmodes, nmodes)
#         UTU = U.T @ U
#         UTU_inv = np.linalg.pinv(UTU)

#         # projected row
#         prow -= (PU @ UTU_inv) @ U.T

#         Pnew.data[start:end] = prow

#         # DEBUG
#         # new_PU = Pnew.data[start:end] @ U
#         # print(f"{irow=} {np.linalg.norm(new_PU)=:.4e}")

#     Pnew.eliminate_zeros()
#     return Pnew

def orthog_nullspace_projector_csr(
    P: sp.csr_matrix,
    Bc: np.ndarray,
):
    """
    Apply the orthogonal projector row-wise for a scalar CSR prolongator.

    Ensures each row satisfies
        Pnew[i, :] @ Bc = 0
    in the local least-squares / orthogonal projection sense.

    Parameters
    ----------
    P : sp.csr_matrix
        Scalar CSR matrix of shape (nfine, ncoarse)
    Bc : np.ndarray
        Coarse nullspace modes, shape (ncoarse, nmodes)

    Returns
    -------
    Pnew : sp.csr_matrix
        Projected CSR matrix
    """

    assert sp.isspmatrix_csr(P)
    assert Bc.ndim == 2
    assert Bc.shape[0] == P.shape[1]

    Pnew = sp.csr_matrix(
        (P.data.copy(), P.indices.copy(), P.indptr.copy()),
        shape=P.shape
    )

    for irow in range(P.shape[0]):
        start = P.indptr[irow]
        end = P.indptr[irow + 1]

        if start == end:
            continue

        cols = P.indices[start:end]
        prow = Pnew.data[start:end]

        # local coarse nullspace restricted to this row sparsity
        U = Bc[cols, :]   # shape (nnz_row, nmodes)

        # project prow onto orthogonal complement of span(U)
        PU = prow @ U                 # shape (nmodes,)
        UTU = U.T @ U                 # shape (nmodes, nmodes)
        UTU_inv = np.linalg.pinv(UTU)

        prow -= (PU @ UTU_inv) @ U.T

        Pnew.data[start:end] = prow

        # DEBUG
        # new_PU = Pnew.data[start:end] @ U
        # print(f"{irow=} {np.linalg.norm(new_PU)=:.4e}")

    Pnew.eliminate_zeros()
    return Pnew

import numpy as np
import scipy.sparse as sp


def energy_smooth_vector_csr(
    B: np.ndarray,
    A: sp.csr_matrix,
    omega: float = 0.7,
    nsmooth: int = 1,
):
    """
    Pure Jacobi energy smoothing of a single dense fine-grid candidate vector.

    Parameters
    ----------
    B : np.ndarray
        Dense fine-grid candidate vector, shape (n,)
    A : sp.csr_matrix
        Fine-grid operator, shape (n, n)
    omega : float
        Jacobi damping parameter
    nsmooth : int
        Number of Jacobi smoothing steps

    Returns
    -------
    Bnew : np.ndarray
        Smoothed candidate vector
    """
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()

    assert B.ndim == 1
    assert B.shape[0] == A.shape[0]
    assert A.shape[0] == A.shape[1]

    Dinv = 1.0 / A.diagonal()
    Bnew = B.copy()

    for _ in range(nsmooth):
        Bnew -= omega * Dinv * (A @ Bnew)

    return Bnew

def spectral_radius_DinvA_csr(A: sp.spmatrix, maxiter=200, tol=1e-8):
    """
    Estimate spectral radius of D^{-1} A for a CSR matrix A.
    Uses a matrix-free LinearOperator.
    """
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()

    n = A.shape[0]

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
        np.all(np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] != i]) < tol)
        and np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] == i][0] - 1.0) < tol
        for i in range(A.shape[0])
    ], dtype=bool)


def scalar_orthog_projector(dP: sp.csr_matrix, bc_flags, C_nodes=None):
    """
    Scalar orthogonal projector from energy optimization paper.

    For root-node AMG:
      - do not modify BC rows
      - optionally do not modify C-node/root rows
    """
    for i in range(dP.shape[0]):
        if bc_flags[i]:
            continue
        if C_nodes is not None and C_nodes[i]:
            continue

        row_ips = np.arange(dP.indptr[i], dP.indptr[i+1])
        row_vals = dP.data[row_ips]
        if row_vals.size > 0:
            dP.data[row_ips] -= np.mean(row_vals)

    return dP


# def smooth_prolongator_csr(
#     T: sp.csr_matrix,
#     A: sp.csr_matrix,
#     Bc:np.ndarray,
#     C_nodes: np.ndarray,
#     omega: float = 0.7,
#     near_kernel: bool = True,
# ):
#     """
#     Single-step Jacobi smoothing of tentative prolongator for root-node AMG.

#     Root-node modification:
#       - C-node rows are kept fixed as pure injection rows
#       - only F-node rows are smoothed
#     """
#     if not sp.isspmatrix_csr(T):
#         T = T.tocsr()
#     if not sp.isspmatrix_csr(A):
#         A = A.tocsr()

#     Dinv = 1.0 / A.diagonal()
#     AT = A @ T

#     if near_kernel:
#         AT = orthog_nullspace_projector_csr(AT.tocsr(), Bc).tocsr()
#     else:
#         AT = AT.tocsr()

#     DAT = AT.multiply(Dinv[:, None])
#     P = (T - omega * DAT).tolil()

#     # Enforce root-node injection exactly on C-node rows
#     root_rows = np.where(C_nodes)[0]
#     for i in root_rows:
#         row_start = T.indptr[i]
#         row_end = T.indptr[i + 1]

#         if row_end - row_start != 1:
#             raise ValueError(
#                 f"Expected exactly one nonzero in tentative row for C-node {i}"
#             )

#         col = T.indices[row_start]
#         val = T.data[row_start]

#         P.rows[i] = [int(col)]
#         P.data[i] = [float(val)]

#     return P.tocsr()

import numpy as np
import scipy.sparse as sp


def enforce_csr_sparsity_mask(X: sp.csr_matrix, mask: sp.csr_matrix):
    """
    Enforce fixed sparsity on CSR matrix X using the sparsity pattern of `mask`.

    Returns X .* mask_pattern, where only entries in the mask pattern survive.
    """
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    if not sp.isspmatrix_csr(mask):
        mask = mask.tocsr()

    Xout = X.multiply(mask)
    Xout.sum_duplicates()
    Xout.eliminate_zeros()
    return Xout.tocsr()


def enforce_rootnode_injection_csr(P: sp.csr_matrix, T: sp.csr_matrix, C_nodes: np.ndarray):
    """
    Replace root-node rows in P with the original injection rows from T.

    Assumes each C-node row in T has exactly one nonzero.
    """
    if not sp.isspmatrix_csr(P):
        P = P.tocsr()
    if not sp.isspmatrix_csr(T):
        T = T.tocsr()

    Plil = P.tolil()

    root_rows = np.where(C_nodes)[0]
    for i in root_rows:
        row_start = T.indptr[i]
        row_end = T.indptr[i + 1]

        if row_end - row_start != 1:
            raise ValueError(
                f"Expected exactly one nonzero in tentative row for C-node {i}"
            )

        col = T.indices[row_start]
        val = T.data[row_start]

        Plil.rows[i] = [int(col)]
        Plil.data[i] = [float(val)]

    return Plil.tocsr()

def smooth_prolongator_csr(
    T: sp.csr_matrix,
    A: sp.csr_matrix,
    Bc: np.ndarray,
    C_nodes: np.ndarray,
    omega: float = 0.7,
    near_kernel: bool = True,
    nsmooth: int = 1,
):
    """
    Multi-step Jacobi smoothing of tentative prolongator for root-node AMG.

    Root-node modification:
      - C-node rows are restored to pure injection after each step
      - smoothing uses fixed sparsity pattern of A @ T
    """
    if not sp.isspmatrix_csr(T):
        T = T.tocsr()
    else:
        T = T.copy()

    if not sp.isspmatrix_csr(A):
        A = A.tocsr()

    Dinv = 1.0 / A.diagonal()

    # fixed sparsity pattern from A @ T
    AT0 = (A @ T).tocsr()
    AT0.sum_duplicates()
    AT0.eliminate_zeros()

    mask = AT0.copy()
    mask.data[:] = 1.0

    P = T.copy()

    for _ in range(nsmooth):
        AP = (A @ P).tocsr()
        AP = enforce_csr_sparsity_mask(AP, mask)

        if near_kernel:
            AP = orthog_nullspace_projector_csr(AP, Bc)

        DAP = AP.multiply(Dinv[:, None])
        P = (P - omega * DAP).tocsr()
        P = enforce_rootnode_injection_csr(P, T, C_nodes)

    P.sum_duplicates()
    P.eliminate_zeros()
    return P


class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spolve(self.A, rhs)
        return x
import numpy as np
import scipy.sparse as sp


class DirectCSRSolver:
    def __init__(self, A_csr: sp.csr_matrix):
        assert sp.isspmatrix_csr(A_csr)
        self.A = A_csr.tocsc()

    def solve(self, rhs: np.ndarray):
        return sp.linalg.spsolve(self.A, rhs)


class RootNodeAMGSolver:
    """General multilevel scalar root-node AMG solver."""
    def __init__(
        self,
        A_free: sp.csr_matrix,
        A: sp.csr_matrix,
        threshold: float = 0.25,
        omega: float = 0.7,
        pre_smooth: int = 1,
        post_smooth: int = 1,
        level: int = 0,
        near_kernel: bool = True,
        coarsening_fcn=None,
        aggregation_fcn=greedy_rn_serial_aggregation_csr,
        rbm_omega: float = 0.5,
        rbm_nsmooth: int = 2,
        prol_nsmooth: int = 1,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        A_free : sp.csr_matrix
            Fine-grid operator without Dirichlet enforcement, used for coarsening/aggregation.
        A : sp.csr_matrix
            Fine-grid operator with BC treatment, used for smoothing / Galerkin products.
        threshold : float
            Strength/coarsening threshold.
        omega : float
            Jacobi damping for prolongation smoothing.
        pre_smooth, post_smooth : int
            Stored for later V-cycle use.
        level : int
            Current AMG level.
        near_kernel : bool
            Whether to apply row-wise orthogonal projector during prolongation smoothing.
        coarsening_fcn : callable
            Should return (C_nodes, F_nodes) from A_free.
        aggregation_fcn : callable
            Should return aggregate_ind from (A_free, C_nodes, threshold).
        rbm_omega : float
            Jacobi damping used to smooth the scalar rigid-body / constant candidate.
        rbm_nsmooth : int
            Number of smoothing steps for the rigid-body candidate.
        prol_nsmooth : int
            Number of smoothing steps for prolongator smoothing.
        verbose : bool
            Print small diagnostics.
        """
        assert sp.isspmatrix_csr(A_free)
        assert sp.isspmatrix_csr(A)

        self.A_free = A_free
        self.A = A
        self.threshold = threshold
        self.omega = omega
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.level = level
        self.near_kernel = near_kernel
        self.verbose = verbose

        self.nfine = A.shape[0]
        self.fine_nnz = A.nnz

        # --------------------------------------------------
        # 1. Coarsening: get C/F splitting
        # --------------------------------------------------
        self.C_nodes, self.F_nodes = coarsening_fcn(A_free, threshold=threshold)

        # --------------------------------------------------
        # 2. Aggregation using the chosen C-nodes
        # --------------------------------------------------
        self.aggregate_ind = aggregation_fcn(A_free, self.C_nodes, threshold=threshold)
        self.num_agg = np.max(self.aggregate_ind) + 1

        # --------------------------------------------------
        # 3. Energy-smooth scalar fine-grid RBM / constant mode
        # --------------------------------------------------
        B0 = np.ones(self.nfine, dtype=np.double)
        self.B = energy_smooth_vector_csr(
            B0, A, omega=rbm_omega, nsmooth=rbm_nsmooth
        )

        # --------------------------------------------------
        # 4. Tentative prolongator and coarse candidate
        # --------------------------------------------------
        self.T, self.Bc = tentative_prolongator_csr(
            self.aggregate_ind, self.C_nodes, self.B
        )
        self.Bc_stack = self.Bc.reshape((self.Bc.shape[0], 1))

        if verbose:
            B_resid = self.T @ self.Bc - self.B
            print(
                f"level {level}: num_agg={self.num_agg}, "
                f"||T Bc - B||_2 = {np.linalg.norm(B_resid):.4e}, "
                f"||T Bc - B||_inf = {np.linalg.norm(B_resid, ord=np.inf):.4e}"
            )

        # --------------------------------------------------
        # 5. Optional small check for orthogonal projector
        # --------------------------------------------------
        if verbose and near_kernel:
            AT = (A @ self.T).tocsr()
            U = orthog_nullspace_projector_csr(AT, self.Bc_stack)
            proj_resid = U @ self.Bc
            print(
                f"level {level}: ||Proj(A T) Bc||_2 = {np.linalg.norm(proj_resid):.4e}, "
                f"||Proj(A T) Bc||_inf = {np.linalg.norm(proj_resid, ord=np.inf):.4e}"
            )

        # --------------------------------------------------
        # 6. Energy-smooth the tentative prolongator
        # --------------------------------------------------
        self.P = smooth_prolongator_csr(
            self.T,
            A,
            self.Bc_stack,
            self.C_nodes,
            omega=omega,
            near_kernel=near_kernel,
            nsmooth=prol_nsmooth,
        )
        self.R = self.P.T.tocsr()

        if verbose:
            B_resid_post = self.P @ self.Bc - self.B
            print(
                f"level {level}: ||P Bc - B||_2 = {np.linalg.norm(B_resid_post):.4e}, "
                f"||P Bc - B||_inf = {np.linalg.norm(B_resid_post, ord=np.inf):.4e}"
            )

        # --------------------------------------------------
        # 7. Galerkin coarse operators
        # --------------------------------------------------
        self.Ac = (self.R @ (A @ self.P)).tocsr()
        self.Ac_free = (self.R @ (A_free @ self.P)).tocsr()

        self.coarse_nnz = self.Ac.nnz
        self.ncoarse = self.Ac.shape[0]

        # --------------------------------------------------
        # 8. Build next coarse solver
        # --------------------------------------------------
        self.coarse_solver = None
        max_coarse_nnz = self.ncoarse ** 2

        if level == 0 and verbose:
            print("level 0 is AMG solver..")

        if self.coarse_nnz >= 0.4 * max_coarse_nnz or self.ncoarse <= 100:
            if verbose:
                print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            if verbose:
                print(f"level {level+1} building AMG solver")
            self.coarse_solver = RootNodeAMGSolver(
                self.Ac_free,
                self.Ac,
                threshold=threshold,
                omega=omega,
                pre_smooth=pre_smooth,
                post_smooth=post_smooth,
                level=level + 1,
                near_kernel=near_kernel,
                coarsening_fcn=coarsening_fcn,
                aggregation_fcn=aggregation_fcn,
                rbm_omega=rbm_omega,
                rbm_nsmooth=rbm_nsmooth,
                prol_nsmooth=prol_nsmooth,
                verbose=verbose,
            )

        # --------------------------------------------------
        # 9. Some hierarchy stats
        # --------------------------------------------------

        if level == 0 and verbose:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=:.4f}")
            print(f"\tnum nodes per level = {self.num_nodes_list}")

    @property
    def total_nnz(self) -> int:
        # get total nnz across all levels
        if isinstance(self.coarse_solver, RootNodeAMGSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else: # direct solver
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz     

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, RootNodeAMGSolver):
            return self.coarse_solver.num_levels + 1
        else: # direct solver
            return 2
        
    @property
    def num_nodes_list(self) -> str:
        if isinstance(self.coarse_solver, RootNodeAMGSolver):
            return str(self.A.shape[0]) + "," + self.coarse_solver.num_nodes_list
        else: # direct solver
            return f"{self.A.shape[0]},{self.Ac.shape[0]}"

    def solve(self, rhs):
        """
        One AMG V-cycle
        """
        # initial guess
        x = np.zeros_like(rhs)

        # -------- pre-smoothing --------
        if self.pre_smooth > 0:
            x = gauss_seidel_csr(
                self.A, rhs,
                x0=x,
                num_iter=self.pre_smooth
            )

        # fine residual
        r = rhs - self.A @ x

        # restrict
        rc = self.R @ r

        # coarse solve
        ec = self.coarse_solver.solve(rc)

        # prolong correction
        x += self.P @ ec

        # -------- post-smoothing --------
        if self.post_smooth > 0:
            r = rhs - self.A @ x
            dx = gauss_seidel_csr_transpose(
                self.A, r,
                x0=np.zeros_like(rhs),
                num_iter=self.post_smooth
            )
            x += dx

        return x
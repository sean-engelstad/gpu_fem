# Reissner-Mindlin cylinder with MITC4 shell elements, 6x6 DOF per node BSR matrix
# using root-node AMG

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys

# cylinder case imports
sys.path.append("../../../milu/")
sys.path.append("../1_sa_amg/")
from _cylinder import make_cylinder_case
from __src import plot_cylinder_vec, sort_vis_maps

sys.path.append("../../../adv_elem/1_beam/src/")
from smoothers import right_pcg2, right_pgmres2

# SA / BSR utilities
sys.path.append("../1_sa_amg/_src/")
from bsr_aggregation import enforce_symmetric_dirichlet_bcs_bsr, get_rigid_body_modes

# CF AMG imports
sys.path.append("../2_cf_amg/_src/")
from bsr_coarsening import (
    standard_bsr_coarsening,
    aggressive_A2_bsr_coarsening,
    aggressive_A1_bsr_coarsening,
    RS_bsr_coarsening,
)

# root-node AMG imports
sys.path.append("_src/")
from bsr_coarse import greedy_rn_serial_aggregation_bsr, RootNodeAMG_BSRSolver


def plot_cylinder_rootnode_aggregation(
    xpts0: np.ndarray,
    aggregate_ind: np.ndarray,
    C_mask: np.ndarray,
    title: str = "Root-Node AMG Aggregates",
    flat: bool = True,
):
    """
    Plot cylinder node aggregates and root/C nodes.

    Parameters
    ----------
    xpts0 : (3*nnodes,) or (nnodes, 3) ndarray
        Cylinder nodal coordinates
    aggregate_ind : (nnodes,) int ndarray
        Aggregate index per node
    C_mask : (nnodes,) bool ndarray
        Root/C-node mask
    flat : bool
        If True, unwrap cylinder by plotting theta vs x
    """
    if xpts0.ndim == 1:
        xyz = xpts0.reshape((-1, 3))
    else:
        xyz = xpts0.copy()

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    theta = np.arctan2(z, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    if flat:
        # unwrap cylinder to (x, theta)
        ax.scatter(
            x,
            theta,
            c=aggregate_ind,
            cmap="tab20",
            s=35,
            alpha=0.9,
        )

        ax.scatter(
            x[C_mask],
            theta[C_mask],
            c="k",
            s=85,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            label="Root / C nodes",
            zorder=3,
        )

        ax.set_xlabel("x")
        ax.set_ylabel(r"$\theta$")
    else:
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            x,
            y,
            z,
            c=aggregate_ind,
            cmap="tab20",
            s=20,
            alpha=0.9,
        )

        ax.scatter(
            x[C_mask],
            y[C_mask],
            z[C_mask],
            c="k",
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            label="Root / C nodes",
            zorder=3,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_title(title)
    ax.legend()
    plt.show()


# ====================================================
# 1) make cylinder case
# ====================================================

import argparse
parser = argparse.ArgumentParser()

# AMG settings
parser.add_argument("--nmatrix", type=int, default=2, help="num prolongation smoothing steps")
parser.add_argument("--nsmooth", type=int, default=2, help="num V-cycle smoothing sweeps")
parser.add_argument("--threshold", type=float, default=0.1, help="coarsening / aggregation threshold")
parser.add_argument("--omega", type=float, default=0.5, help="Jacobi prolongation smoothing coeff")
parser.add_argument("--crs", type=str, default="standard", help="coarsening type: [rs, standard, A1, A2]")
parser.add_argument("--thick", type=float, default=1e-3)

# problem / run settings
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="random ordering")
parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="disable plots")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="disable preconditioner")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="apply one V-cycle only")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="debug multilevel process")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="disable near-kernel projector")
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level (unused here)")
parser.add_argument("--nxe", type=int, default=16, help="num axial elems")
args = parser.parse_args()

complex_load = True

_, _, A, rhs, perm, xpts0 = make_cylinder_case(args, complex_load=complex_load, apply_bcs=True)
_, _, A_free, _, _, _ = make_cylinder_case(args, complex_load=complex_load, apply_bcs=False)

print("ENFORCE SYM BCs")
A = enforce_symmetric_dirichlet_bcs_bsr(A)

N = A.shape[0]
nnodes = N // 6

assert not args.random

# ====================================================
# 2) direct solve baseline
# ====================================================

x_direct = sp.linalg.spsolve(A.copy(), rhs.copy())

# ====================================================
# 3) choose coarsening + aggregation
# ====================================================

if args.crs == "rs":
    coarsening_fcn = RS_bsr_coarsening
elif args.crs == "standard":
    coarsening_fcn = standard_bsr_coarsening
elif args.crs == "A1":
    coarsening_fcn = aggressive_A1_bsr_coarsening
elif args.crs == "A2":
    coarsening_fcn = aggressive_A2_bsr_coarsening
else:
    raise ValueError(f"Unknown coarsening type: {args.crs}")

aggregation_fcn = greedy_rn_serial_aggregation_bsr

# ====================================================
# 4) debug root-node aggregates on cylinder
# ====================================================

if args.debug:
    C_mask, F_mask = coarsening_fcn(A_free, threshold=args.threshold)
    aggregate_ind = aggregation_fcn(A_free, C_mask, threshold=args.threshold)
    num_agg = np.max(aggregate_ind) + 1

    num_C_nodes = np.sum(C_mask)
    num_F_nodes = np.sum(F_mask)
    print(f"{num_C_nodes=} {num_F_nodes=} {num_agg=}")

    plot_cylinder_rootnode_aggregation(
        xpts0=xpts0,
        aggregate_ind=aggregate_ind,
        C_mask=C_mask,
        title=f"Cylinder Root-Node Aggregates ({args.crs})",
        flat=True,
    )

# ====================================================
# 5) build root-node AMG preconditioner
# ====================================================

B = get_rigid_body_modes(xpts0)

# smoother settings
smoother, omegas, overlap = "gs", 0.8, 0
# smoother, omegas, overlap = "asw", 0.5, 0
# smoother, omegas, overlap = "asw", 0.3, 1

pc = RootNodeAMG_BSRSolver(
    A_free,
    A,
    B,
    coarsening_fcn=coarsening_fcn,
    aggregation_fcn=aggregation_fcn,
    threshold=args.threshold,
    omega=args.omega,
    near_kernel=not args.nokernel,
    smoother=smoother,
    omegaSmooth=omegas,
    prol_nsmooth=args.nmatrix,
    nsmooth=args.nsmooth,
    asw_overlap=overlap,
    asw_sd_size=2,
    
    rbm_nsmooth=3,
    # rbm_nsmooth=1,
    # rbm_nsmooth=0,
)

# ====================================================
# 6) solve
# ====================================================

if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2, iters = right_pcg2(
        A,
        b=rhs,
        x0=None,
        M=pc if not args.noprec else None,
        rtol=1e-6,
        atol=1e-12,
    )
    print(f"{iters=}")

# ====================================================
# 7) plot
# ====================================================

if not args.noplot:
    print("plot fine soln using direct vs iterative solver")

    flat = True
    nxe = args.nxe
    sort_fw = np.arange(0, N)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    plot_cylinder_vec(nxe, x_direct.copy(), ax, sort_fw, nodal_dof=2, flat=flat)

    ax = fig.add_subplot(122, projection="3d")
    plot_cylinder_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2, flat=flat)
    plt.show()
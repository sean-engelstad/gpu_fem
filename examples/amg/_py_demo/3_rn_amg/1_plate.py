# Reissner-Mindlin plate with MITC4 shell elements, 6x6 DOF per node BSR matrix and using SA-AMG

"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
# plate case imports from milu python cases
sys.path.append("../../../milu/")
from _plate import make_plate_case
from __src import plot_plate_vec, sort_vis_maps
# from __linalg import right_pgmres, right_pcg

sys.path.append("../../../adv_elem/1_beam/src/")
from smoothers import right_pcg2, right_pgmres2

# SA AMG imports
sys.path.append("../1_sa_amg/_src/")
from csr_aggregation import plot_plate_aggregation
from bsr_aggregation import greedy_serial_aggregation_bsr, tentative_prolongator_bsr, smooth_prolongator_bsr
from _smoothers import block_gauss_seidel_6dof
from bsr_aggregation import enforce_symmetric_dirichlet_bcs_bsr

# CF AMG imports
sys.path.append("../2_cf_amg/_src/")
from bsr_coarsening import standard_bsr_coarsening, aggressive_A2_bsr_coarsening, aggressive_A1_bsr_coarsening, RS_bsr_coarsening

# root node AMG imports
sys.path.append("_src/")
from bsr_coarse import greedy_rn_serial_aggregation_bsr, RootNodeAMG_BSRSolver

# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
# AMG settings
parser.add_argument("--nmatrix", type=int, default=2, help="num energy-opt iters (if iter == 1 same as SA-AMG)")
parser.add_argument("--nsmooth", type=int, default=2, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--threshold", type=float, default=0.1, help="aggregation sparsity threshold, higher threshold increases operator complexity and takes fewer GMRES iters, but inc memory and more time per iteration (tradeoff)")
parser.add_argument("--omega", type=float, default=0.5, help="jacobi smoother update coeff, default is None and uses spectral radius")
parser.add_argument("--crs", type=str, default="standard", help="string: type of coarsening [rs, standard, A1, A2]")


# parser.add_argument("--nxe", type=int, default=32, help="num elems each direction x and y")
parser.add_argument("--nxe", type=int, default=8, help="num elems each direction x and y")

parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-3) # 2e-3
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, no: solve with GMRES")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="whether to debug multilevel process")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="whether to turnoff kernel or not")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
args = parser.parse_args()

complex_load = True
# complex_load = False

_, _, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load, apply_bcs=True)
_, _, A_free, _, _, _ = make_plate_case(args, complex_load=complex_load, apply_bcs=False)


print("ENFORCE SYM BCs")
A = enforce_symmetric_dirichlet_bcs_bsr(A)

N = A.shape[0]
nnodes = N // 6

assert not(args.random) # just regular ordering.. 

# ====================================================
# 2) direct solve baseline
# ====================================================

# equiv solution with no reorder
x_direct = sp.linalg.spsolve(A.copy(), rhs.copy())



# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

from bsr_aggregation import get_rigid_body_modes #, get_coarse_rigid_body_modes
B = get_rigid_body_modes(xpts0)
# for i in range(nnodes):
#     print(f"xpt[{i=}] : {xpts0[3*i:(3*i+3)]}")

# Gauss-Seidel actually performs better than block-Jacobi style non-overlapping ASWs smoother (not quite elem-based + hence, not much better prob)
smoother, omegas, overlap = 'gs', 0.8, 0
# smoother, omegas, overlap = 'asw', 0.5, 0
# smoother, omegas, overlap = 'asw', 0.3, 1

# choose coarsening method
if args.crs == "rs":
    coarsening_fcn = RS_bsr_coarsening
elif args.crs == "standard":
    coarsening_fcn = standard_bsr_coarsening
elif args.crs == "A1":
    coarsening_fcn = aggressive_A1_bsr_coarsening
elif args.crs == "A2":
    coarsening_fcn = aggressive_A2_bsr_coarsening

aggregation_fcn = greedy_rn_serial_aggregation_bsr

if args.debug:
    Lx = Ly = 1.0
    nx = ny = args.nxe + 1

    # root-node coarse/fine splitting
    C_mask, F_mask = coarsening_fcn(A_free, threshold=args.threshold)

    # root-node aggregation
    aggregate_ind = aggregation_fcn(A_free, C_mask, threshold=args.threshold)
    num_agg = np.max(aggregate_ind) + 1

    # plate grid
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xf = X.ravel()
    Yf = Y.ravel()

    num_C_nodes = np.sum(C_mask)
    num_F_nodes = np.sum(F_mask)
    print(f"{num_C_nodes=} {num_F_nodes=} {num_agg=}")

    fig, ax = plt.subplots(figsize=(7, 6))

    # plot aggregates
    sc = ax.scatter(
        Xf, Yf,
        c=aggregate_ind,
        cmap="tab20",
        s=35,
        alpha=0.9,
    )

    # overlay root/C nodes separately
    ax.scatter(
        Xf[C_mask], Yf[C_mask],
        c="k",
        s=80,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="Root / C nodes",
        zorder=3,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Root-Node Aggregates ({args.crs})")
    ax.legend()

    plt.show()


pc = RootNodeAMG_BSRSolver(
    A_free, A, B, 
    coarsening_fcn=coarsening_fcn,
    aggregation_fcn=aggregation_fcn,
    threshold=args.threshold, omega=args.omega, 
    near_kernel=not(args.nokernel),
    smoother=smoother,
    omegaSmooth=omegas,
    prol_nsmooth=args.nmatrix,
    nsmooth=args.nsmooth,
    # asw_overlap=0,
    asw_overlap=overlap,
    # asw_sd_size=4,
    # asw_sd_size=3,
    asw_sd_size=2,
    # nmatrix=args.nmatrix,
    # rbm_nsmooth=3,
    rbm_nsmooth=2,
    # rbm_nsmooth=1,
    # rbm_nsmooth=0,
)

# # check V-cycle symmetry
# x = np.random.rand(N)
# y = np.random.rand(N)
# lhs_ = x @ pc.solve(y)
# rhs_ = y @ pc.solve(x)
# print(f"{lhs_=} {rhs_=}")
# print(np.allclose(lhs_, rhs_))


if args.justpc:
    x2 = pc.solve(rhs)
else:
    # x2 = right_pgmres2(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)
    # coarse grid matrix is not exactly numerically symmetric rn.. (due to spsolve function?) so GMRES is doing better..
    # after fix orthog projector may work again (cause less near-zero pivots)?
    # NOPE : issue was that only BCs applied to rows not cols (fixed above) => now can use PCG
    x2, iters = right_pcg2(A, b=rhs, x0=None, M=pc if not(args.noprec) else None, rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

if not args.noplot:
    print("plot fine soln using direct vs iterative solver")

    # for plotting
    nxe = int(nnodes**0.5)-1
    sort_fw = np.arange(0, N)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plot_plate_vec(nxe, x_direct.copy(), ax, sort_fw, nodal_dof=2)

    # plot right-precond solution
    ax = fig.add_subplot(122, projection='3d')
    plot_plate_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2)
    plt.show()
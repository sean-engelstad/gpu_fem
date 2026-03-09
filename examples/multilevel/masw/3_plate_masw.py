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

# AMG imports
sys.path.append("../../../amg/_py_demo/_src/")
from csr_aggregation import plot_plate_aggregation
# , tentative_prolongator_bsr, smooth_prolongator_bsr
from _smoothers import block_gauss_seidel_6dof

sys.path.append("src/")
from _bsr_aggregation import MASW_BSRSolver
from _bsr_aggregation import greedy_serial_aggregation_bsr, enforce_symmetric_dirichlet_bcs_bsr

# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
parser.add_argument("--nlevels", type=int, default=None, help="# MASW levels (1 or more)")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, no: solve with GMRES")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="whether to debug multilevel process")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="whether to turnoff kernel or not")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=32, help="num elems each direction x and y")
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
parser.add_argument("--iters", type=int, default=1, help="num energy-opt iters (if iter == 1 same as SA-AMG)")
parser.add_argument("--nsmooth", type=int, default=4, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--threshold", type=float, default=0.13, help="aggregation sparsity threshold, higher threshold increases operator complexity and takes fewer GMRES iters, but inc memory and more time per iteration (tradeoff)")
parser.add_argument("--omega", type=float, default=None, help="jacobi smoother update coeff, default is None and uses spectral radius")
args = parser.parse_args()

complex_load = True
# complex_load = False

_, _, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load, apply_bcs=True)
_, _, A_free, _, _, _ = make_plate_case(args, complex_load=complex_load, apply_bcs=False)

print("ENFORCE SYM BCs")
A = enforce_symmetric_dirichlet_bcs_bsr(A)

# plt.spy(A)
# plt.show()

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
pc = MASW_BSRSolver(
    A_free, A, 
    B=B, # rigid body modes..
    xpts=xpts0.reshape((nnodes, 3)),
    # num_levels=1, 
    # num_levels=2,
    num_levels=args.nlevels,
    threshold=0.1,
    # omega=0.7,
    omega=0.5,
    near_kernel=not args.nokernel, 
    coarsening_fcn=greedy_serial_aggregation_bsr,
    # node_sort_fcn=rcm_sort_fcn, # this is bad ordering doesn't lead small node groups to be consecutive..
    node_sort_fcn=None,
    morton_sort=True,
    # morton_sort=False,
    smooth_prolongator=True,
    # smooth_prolongator=False,
    # asw_sd_size=2,
    asw_sd_size=3,
    # asw_sd_size=4,
    # asw_sd_size=8,
    # asw_sd_size=16,
    # asw_sd_size=32,
    # asw_sd_size=8,
    omegaJac=0.25,
    # omegaJac=0.5,
    nsmooth=args.nsmooth,
   # nsmooth=4,
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
    # x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)
    # coarse grid matrix is not exactly numerically symmetric rn.. (due to spsolve function?) so GMRES is doing better..
    # after fix orthog projector may work again (cause less near-zero pivots)?
    x2, iters = right_pcg2(A, b=rhs, x0=None, M=pc if not(args.noprec) else None)


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
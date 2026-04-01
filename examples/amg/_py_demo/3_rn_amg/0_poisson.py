"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../milu/")
# from __src import right_pgmres
from __linalg import right_pcg

sys.path.append("../1_sa_amg/_src")
from poisson import poisson_2d_csr, plot_poisson_surface, poisson_apply_bcs

sys.path.append("_src/")
from csr_coarse import greedy_rn_serial_aggregation_csr, plot_plate_aggregation
# from csr_aggregation import tentative_prolongator_csr, smooth_prolongator_csr
# from csr_aggregation import AggregationAMGSolver, gauss_seidel_csr
from csr_coarse import RootNodeAMGSolver

sys.path.append("../2_cf_amg/_src")
from cf_coarsening import standard_csr_coarsening, aggressive_A2_csr_coarsening, aggressive_A1_csr_coarsening

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=50, help="num nodes each direction")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="enforce near kernel constr on prolong update")
# parser.add_argument("--smoothP", type=int, default=1, help="whether to smooth prolongation or not")
parser.add_argument("--debug", type=int, default=0, help="debug printouts")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, default of no: solve with PCG")
args = parser.parse_args()

# ------------------------------------------------------------
# 0) Problem setup
# ------------------------------------------------------------
nx, ny = args.nxe + 1, args.nxe + 1

Lx, Ly = 1.0, 1.0
hx, hy = Lx/(nx-1), Ly/(ny-1)

A_free = poisson_2d_csr(nx, ny, hx, hy)

# RHS: smooth forcing
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# f = np.sin(np.pi * X) * np.sin(np.pi * Y)
f = np.sin(5 * np.pi * X) * np.sin(np.pi * (Y**2 + X**2))
f = f.ravel()

# Apply homogeneous Dirichlet BCs (simple row overwrite)
A = poisson_apply_bcs(A_free, f, nx, ny)

# ------------------------------------------------------------
# 1) Direct Solve (for true soln)
# ------------------------------------------------------------
import time
start_direct = time.time()
u_truth = spla.spsolve(A.copy(), f)
time_direct = time.time() - start_direct

# ------------------------------------------------------------
# 2) aggregation and tentative prolongator construction
# ------------------------------------------------------------

nnodes = A.shape[0]

# based on papers
"""
1) A GPU accelerated aggregation algebraic multigrid method, https://www.sciencedirect.com/science/article/pii/S0898122114004143
2) EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS, https://epubs.siam.org/doi/epdf/10.1137/110838844
also somewhat tips from this RN AMG and pics (though this is more advanced method), https://epubs.siam.org/doi/epdf/10.1137/16M1082706
"""

# ==========================================================================================
if args.debug:
    # C_nodes, F_nodes = standard_csr_coarsening(A_free, threshold=0.25)
    # C_nodes, F_nodes = aggressive_A1_csr_coarsening(A_free, threshold=0.25)
    C_nodes, F_nodes = aggressive_A2_csr_coarsening(A_free, threshold=0.25)

    # compute node aggregate sets
    # make sure to use unconstrained matrix for aggregation indicators originally
    aggregate_ind = greedy_rn_serial_aggregation_csr(A_free, C_nodes, threshold=0.25)
    num_agg = np.max(aggregate_ind) + 1

    # print(f"{aggregate_ind=}")
    print(f"{num_agg=}")
    plot_plate_aggregation(aggregate_ind, C_nodes, nx, ny, Lx, Ly)

# ==========================================================================================

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

# A_free = A.copy() # if A was not BC version.. # makes it much slower, more memory and more DOF

# precond = None
# threshold = 0.25 is default..
# needed a lower threshold to get the coarsening to work right on one of the levels.. (would coarsen from 454 to 454 nodes... then next time would work better)
precond = RootNodeAMGSolver(A_free, A, threshold=0.1,
                            coarsening_fcn=aggressive_A2_csr_coarsening,
                            aggregation_fcn=greedy_rn_serial_aggregation_csr,
                            omega=0.7, pre_smooth=1, post_smooth=1, prol_nsmooth=3,
                            near_kernel=not(args.nokernel))

if args.justpc:
    u2 = precond.solve(f)
else:
    # needed to do M and then M^T pre and post smooths to make AMG precond symmetric for PCG..
    # u2 = right_pcg_v0(A, b=f, x0=None, M=precond)
    start_pcg = time.time()
    u2 = right_pcg(A, b=f, x0=None, M=precond)
    time_pcg = time.time() - start_pcg
    print(f"{time_direct=:.4e} {time_pcg=:.4e}")

# u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u_truth, u2, nx, ny, Lx, Ly)
"""
root-node AMG demo for Poisson problem
based on these two papers:
1. A Root-Node Based AMG method, https://arxiv.org/pdf/1610.03154
2. A General Interpolation Strategy for Algebraic Multigrid Using Energy Minimization, https://epubs.siam.org/doi/epdf/10.1137/100803031
    * this one is more general + krylov based form of energy min..
"""

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
from csr_coarse import energy_smooth_vector_csr, tentative_prolongator_csr, orthog_nullspace_projector_csr
from csr_coarse import smooth_prolongator_csr

sys.path.append("../2_cf_amg/_src")
from cf_coarsening import standard_csr_coarsening, aggressive_A2_csr_coarsening, aggressive_A1_csr_coarsening

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=5, help="num nodes each direction")
parser.add_argument("--kernel", action=argparse.BooleanOptionalAction, default=False, help="enforce near kernel constr on prolong update")
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

# C_nodes, F_nodes = standard_csr_coarsening(A_free, threshold=0.25)
# C_nodes, F_nodes = aggressive_A1_csr_coarsening(A_free, threshold=0.25)
C_nodes, F_nodes = aggressive_A2_csr_coarsening(A_free, threshold=0.1)

# compute node aggregate sets
# make sure to use unconstrained matrix for aggregation indicators originally
aggregate_ind = greedy_rn_serial_aggregation_csr(A_free, C_nodes, threshold=0.1)
num_agg = np.max(aggregate_ind) + 1

# print(f"{aggregate_ind=}")
print(f"{num_agg=}")
plot_plate_aggregation(aggregate_ind, C_nodes, nx, ny, Lx, Ly)


# energy-smooth the rigid body modes
B0 = np.ones(nnodes)
omega = 0.5
B = energy_smooth_vector_csr(B0, A, omega=omega, nsmooth=2)
# plot_poisson_surface(B0, B, nx, ny, Lx, Ly)

# construct tentative prolongation
T, Bc = tentative_prolongator_csr(
    aggregate_ind, C_nodes, B
)
# plt.imshow(T.toarray())
# plt.show()

# test that the initial RBM constraints are satisfied by the tentative prolongation
B_resid = T @ Bc - B
print(f"{B_resid=}")

# test that the orthog projector ensures U * Bc = 0 exactly..
AT = A @ T
Bc_stack = Bc.reshape((Bc.shape[0], 1))
U = orthog_nullspace_projector_csr(AT, Bc_stack)
B_resid2 = U @ Bc
print(f"{B_resid2=}")

# energy-smooth the prolongation (with projector)
P = smooth_prolongator_csr(T, A, Bc_stack, C_nodes, omega, near_kernel=True, nsmooth=1)
# near kernel = True is actually helpful here..

# now check several things..
# P * Bc = B
B_resid3 = P @ Bc - B
print(f"post energy-smooth: {B_resid3=}")

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(T.toarray())
ax[1].imshow(P.toarray())
plt.show()

# # create tentative prolongator then smooth it
# T = tentative_prolongator_csr(aggregate_ind)
# P = smooth_prolongator_csr(T, A, omega=0.7) # single damped jacobi step, so only one step of fillin
# R = P.T # sym matrix so restriction is transpose prolong

# # galerkin coarse grid construction
# Ac = R @ (A @ P)

# print(f"{Ac.shape=}")

# # ------------------------------------------------------------
# # DEBUG: demo steps in V-cycle
# # ------------------------------------------------------------

# # ignore pre-smooth for prelim demo (ADD LATER)

# # restrict fine residual
# rhs = f.copy()
# rhs_coarse = R.dot(rhs)
# # print(f"{rhs.shape=} {rhs_coarse.shape=}")

# # coarse grid direct solve
# soln_coarse = spla.spsolve(Ac, rhs_coarse)

# # prolong solution to fine
# pr_fine = P.dot(soln_coarse)
# r_fine = rhs - A.dot(pr_fine) # new residual

# # smooth solution with Gauss-seidel
# dx_fine = gauss_seidel_csr(A, r_fine, x0=np.zeros(nnodes), num_iter=1)
# x_fine = pr_fine + dx_fine
# r_fine2 = rhs - A.dot(x_fine)

# # plot soln comparison
# plot_poisson_surface(u_truth, x_fine, nx, ny, Lx, Ly)
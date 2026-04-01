"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../milu/")
from __linalg import right_pgmres
from __linalg import right_pcg
sys.path.append("../../../amg/_py_demo/_src/")
from poisson import poisson_2d_csr, plot_poisson_surface, poisson_apply_bcs
from csr_aggregation import greedy_serial_aggregation_csr, plot_plate_aggregation
from csr_aggregation import tentative_prolongator_csr, smooth_prolongator_csr
from csr_aggregation import AMGSolver, gauss_seidel_csr, get_bc_flags
sys.path.append("src/")
from _csr_aggregation import structured_aggregation_csr, rcm_sort_fcn, MASW_Solver

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=50, help="num nodes each direction")
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
MAS: Multilevel additive schwarz
- [ ] [A GPU-based multilevel additive schwarz preconditioner for cloth and deformable body simulation](https://dl.acm.org/doi/10.1145/3528223.3530085)
   - [ ] Nicolaides coarsening [Deflation of Conjugate Gradients with Applications to Boundary Value Problems](https://epubs.siam.org/doi/abs/10.1137/0724027?journalCode=sjnaam)
"""

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

# A_free = A.copy() # if A was not BC version.. # makes it much slower, more memory and more DOF

# precond = None
precond = MASW_Solver(
    A_free, A, 
    # num_levels=1, 
    # num_levels=2,
    num_levels=3,
    threshold=0.1,
    omega=0.7,
    near_kernel=args.kernel, 
    # coarsening_fcn=structured_aggregation_csr,
    coarsening_fcn=greedy_serial_aggregation_csr,
    # node_sort_fcn=rcm_sort_fcn, # this is bad ordering doesn't lead small node groups to be consecutive..
    node_sort_fcn=None,
    smooth_prolongator=True,
    # smooth_prolongator=False,
    # asw_sd_size=2,
    asw_sd_size=4,
    # asw_sd_size=8,
    # asw_sd_size=16,
    # asw_sd_size=8,
    # omegaJac=0.25,
    omegaJac=0.2,
    nsmooth=2,
)

if args.justpc:
    u2 = precond.solve(f)
else:
    # needed to do M and then M^T pre and post smooths to make AMG precond symmetric for PCG..
    # u2 = right_pcg_v0(A, b=f, x0=None, M=precond)
    start_pcg = time.time()
    # u2 = right_pgmres(A, b=f, x0=None, M=precond)
    u2 = right_pcg(A, b=f, x0=None, M=precond)
    time_pcg = time.time() - start_pcg
    print(f"{time_direct=:.4e} {time_pcg=:.4e}")

# u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u_truth, u2, nx, ny, Lx, Ly)
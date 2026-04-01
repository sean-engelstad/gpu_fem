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
from cf_coarsening import RS_csr_coarsening, standard_csr_coarsening, aggressive_A2_csr_coarsening, aggressive_A1_csr_coarsening
from cf_amg import direct_csr_interpolation, standard_csr_interpolation
from cf_amg import ClassicalAMGSolver

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=20, help="num elements each direction")
parser.add_argument("--smoothP", type=int, default=1, help="whether to smooth prolongation or not")
parser.add_argument("--debug", type=int, default=0, help="debug printouts")
parser.add_argument("--crs", type=str, default="standard", help="string: type of coarsening [rs, standard, A1, A2]")
parser.add_argument("--interp", type=str, default="standard", help="string: type of coarsening [rs, standard, A1, A2]")
args = parser.parse_args()

# ------------------------------------------------------------
# 0) Problem setup
# ------------------------------------------------------------
nx, ny = args.nxe, args.nxe

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
u = spla.spsolve(A.copy(), f)
U = u.reshape((nx, ny))


# ------------------------------------------------------------
# 2) CF_AMG aka classical AMG
# ------------------------------------------------------------

# choose coarsening method
if args.crs == "rs":
    coarsening_fcn = RS_csr_coarsening
elif args.crs == "standard":
    coarsening_fcn = standard_csr_coarsening
elif args.crs == "A1":
    coarsening_fcn = aggressive_A1_csr_coarsening
elif args.crs == "A2":
    coarsening_fcn = aggressive_A2_csr_coarsening

if args.interp == "direct":
    interpolation_fcn = direct_csr_interpolation
elif args.interp == "standard":
    interpolation_fcn = standard_csr_interpolation
# elif args.interp == "A1":
#     coarsening_fcn = aggressive_A1_csr_coarsening
# elif args.interp == "A2":
#     coarsening_fcn = aggressive_A2_csr_coarsening



pc = ClassicalAMGSolver(
    A_free, A, threshold=0.1, omega=0.7, pre_smooth=1, post_smooth=1,
    coarsening_fcn=coarsening_fcn,
    interpolation_fcn=interpolation_fcn,
)

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

precond = None

# u2 = precond.solve(f)
# u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)
u2 = right_pcg(A, b=f, x0=None, M=pc)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
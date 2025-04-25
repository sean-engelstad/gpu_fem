import numpy as np
import matplotlib.pyplot as plt
from _gmres_util import gmres, get_laplace_system
import scipy.sparse as spp

# N = 4 # 2 nodes
N = 16 # 16, 900

A, b = get_laplace_system(N)

Adense = A.toarray()
plt.figure()
plt.imshow(Adense)
plt.savefig("Adense.png", dpi=400)

use_precond = True
precond_case = 2

# really nice ref on GMRES and python PGMRES (preconditioned)
# https://relate.cs.illinois.edu/course/CS556-f16/file-version/38da5c7d14e22927a994eeba759f0974b2169fb2/demos/ILU%20preconditioning.html
# algebraic multigrid solver is by far the fastest..

if precond_case == 1:
    # ILU threshold preconditioner with scipy (not ILU(0))
    M = spp.linalg.spilu(A) if use_precond else None
elif precond_case == 2:
    # almost ILU(0) through scipy
    fill_factor = 1 # 1, 2, 3 ILU(k)
    M = spp.linalg.spilu(A, drop_tol=1e-12, fill_factor=fill_factor) if use_precond else None

x_gmres = gmres(A, b, M=M, restart=100, max_iter=100)
if N < 100: print(f"{x_gmres=}")
import numpy as np
import matplotlib.pyplot as plt
from _gmres_util import gmres, get_laplace_system
import scipy.sparse as spp

# N = 4 # 2 nodes
N = 64 # 16, 900

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

# visualize the loads
plt.figure()
n = np.int32(N**0.5)
b_mat = b.reshape((n,n))
plt.imshow(b_mat)
print(f"{b_mat=}")
plt.savefig("bmat.png", dpi=400)

soln_mat = x_gmres.reshape((n,n))
plt.imshow(soln_mat)
print(f"{soln_mat=}")
plt.savefig("soln_mat.png", dpi=400)
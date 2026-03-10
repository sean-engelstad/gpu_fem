# demo the FETI-DP method.. with constrained lagrange multipliers..

"""
two-subdomain neumann neumann preconditioner (example from section 3.1 of [FETI–DP, BDDC, and Block Cholesky Method](https://cs.nyu.edu/~widlund/li_widlund_041211.pdf) )
"""

import numpy as np, scipy.sparse as sp
from src import FETIDP_Assembler, MITCPlateElement
from src import RichardsonSolver, ILU0Preconditioner, ExactSparseSolver, ILUTPreconditioner

import sys
sys.path.append("../_poiss_dd/src/")
from krylov import right_pcg_matfree

# fairly stable to higher DOF and thick-independent!
# only thing is would maybe need true multi-level BDDC or multi-level FETI
# to get fully h-independent convergence I bet.. iter count is going up a bit more for higher DOF 

# nxe, nxs = 256, 64
nxe, nxs = 128, 32
# nxe, nxs = 128, 16
# nxe, nxs = 128, 8
# nxe, nxs = 64, 16
# nxe, nxs = 32, 4
# nxe, nxs = 32, 8
# nxe, nxs = 4, 2

# thick = 1e-1
# thick = 1e-2
thick = 1e-3


m, n = 2, 2
# m, n = 2, 3
def load_fcn(_x,_y):
    import math
    theta = math.atan2(_y, _x)
    r = np.sqrt(_x**2 + _y**2)
    return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta)

ELEMENT = MITCPlateElement(
    prolong_mode='standard',
)
assembler = FETIDP_Assembler(
    ELEMENT=ELEMENT,
    thick=thick,
    nxe=nxe, nxs=nxs, nys=nxs,
    # nxe=4, nxs=2, nys=2, # DEBUG inputs
    # nxe=6, nxs=3, nys=3,
    clamped=True,
    # clamped=False,
    # load_fcn=lambda x, y : np.sin(5 * np.pi * x) * np.sin(np.pi * (y**2 + x**2)),
    # load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y),
    load_fcn=load_fcn,
    # would be nice if coarse space could be solved local (but that doesn't seem to work well, need to assemble it with sum of local Schur complements)
    coarse_mode="assembled", 
    # coarse_mode="local",
    geometry="plate",
    subdomain_solver_cls=ExactSparseSolver,
    # subdomain_solver_cls=RichardsonSolver,
    # subdomain_solver_kwargs=dict(
    #     # precond_cls=ILU0Preconditioner,
    #     precond_cls=ILUTPreconditioner,
    #     # nsteps=1, omega=1.0,
    #     nsteps=4, omega=0.5,
    #     # nsteps=10, omega=0.5,
    #     # precond_cls=ExactSparseSolver, nsteps=1, omega=1.0,
    # ),
)

print(f"{assembler.num_subdomains=}")

assembler.assemble_all()
lam_rhs = assembler.get_lam_rhs() # g_G rhs 
print(f"{np.linalg.norm(lam_rhs)=:.4e}")

def norm(x): 
    # return np.max(np.abs(x))
    return np.linalg.norm(x)

# assembler.neumann_solve(rhs)
norm_hist = []
lam_soln, nsteps = right_pcg_matfree(
    assembler, b=lam_rhs,
    rtol=1e-6, 
    atol=1e-20,
    max_iter=200,
    print_freq=3,
    norm_hist=norm_hist,
)

# check lambda linear system is solved
interface_res = lam_rhs - assembler.mat_vec(lam_soln)
res_nrm = norm(interface_res)
rhs_nrm = norm(lam_rhs)
rel_solve_nrm = res_nrm / rhs_nrm
print(f"{rel_solve_nrm=:.4e}")

# reconstruct global solution
global_soln = assembler.get_global_solution(lam_soln)

# compare to direct solve
direct_soln = assembler.direct_solve(assembly=False)

# compute solution error..
diff = global_soln - direct_soln
err_nrm = np.max(np.abs(diff))
orig_nrm = np.max(np.abs(direct_soln))
rel_nrm = err_nrm / orig_nrm
print(f"{err_nrm=:.4e} {orig_nrm=:.4e} {rel_nrm=:.4e}")


# then plot the solution
assembler.u = global_soln.copy() # store in assembler for plot
assembler.plot_disp()

# DEBUG the FETI-DP system
# ========================================

# # DEBUG
# assembler.u = diff
# assembler.plot_disp()
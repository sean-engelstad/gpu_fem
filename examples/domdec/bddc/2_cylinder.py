# demo the FETI-DP method.. with constrained lagrange multipliers..

"""
two-subdomain neumann neumann preconditioner (example from section 3.1 of [FETI–DP, BDDC, and Block Cholesky Method](https://cs.nyu.edu/~widlund/li_widlund_041211.pdf) )
"""

import numpy as np, scipy.sparse as sp

import sys
sys.path.append("../feti_dp/")
from src import BDDC_Assembler, MITCShellElement, BDDC_EdgeAvg_Assembler
from src import RichardsonSolver, ILU0Preconditioner, ExactSparseSolver, ILUTPreconditioner

import sys
sys.path.append("../_poiss_dd/src/")
from krylov import right_pcg_matfree

# fairly stable to higher DOF and thick-independent!
# only thing is would maybe need true multi-level BDDC or multi-level FETI
# to get fully h-independent convergence I bet.. iter count is going up a bit more for higher DOF 

# nxe, nxs = 128, 8
# nxe, nxs = 64, 8
# nxe, nxs = 32, 4
# nxe, nxs = 16, 2
# nxe, nxs = 16, 4
# nxe, nxs = 32, 8
nxe, nxs = 16, 4

# thick = 1e-1
# thick = 1e-2
# thick = 1e-3
thick = 1e-4

# extra constraints
# edge_averages = False
edge_averages = True

if edge_averages:
    ASSEMBLER = BDDC_EdgeAvg_Assembler
else:
    ASSEMBLER = BDDC_Assembler


m, n = 2, 2
# m, n = 2, 3

radius = 1.0
xs_load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
def xyz_load_fcn(x,y,z):
    th = np.arctan2(y, z)
    dth = th - np.arctan2(-1.0,0)
    s = radius * dth
    return xs_load_fcn(x, s)

ELEMENT = MITCShellElement()
assembler = ASSEMBLER(
    ELEMENT=ELEMENT,
    thick=thick,
    nxe=nxe, nxs=nxs, nys=nxs,
    # nxe=4, nxs=2, nys=2, # DEBUG inputs
    # nxe=6, nxs=3, nys=3,
    clamped=True,
    # cgamped=False,
    load_fcn=xyz_load_fcn,    
    # would be nice if coarse space could be solved local (but that doesn't seem to work well, need to assemble it with sum of local Schur complements)
    coarse_mode="assembled", 
    # coarse_mode="local",
    geometry="cylinder",
    radius=1.0,
)


assembler.assemble_all()

# plot direct solve first
direct_soln = assembler.direct_solve(assembly=False) # cause already assembled
assembler.plot_disp()

gam_rhs = assembler.get_gam_rhs() # g_G rhs 
print(f"{np.linalg.norm(gam_rhs)=:.4e}")

# assembler.neumann_solve(rhs)
norm_hist = []
gam_soln, nsteps = right_pcg_matfree(
    assembler, b=gam_rhs,
    rtol=1e-6, atol=1e-20,
    max_iter=200,
    print_freq=3,
    norm_hist=norm_hist,
)

# reconstruct global solution
global_soln = assembler.get_global_solution(gam_soln)



# compute solution error..
diff = global_soln - direct_soln
err_nrm = np.max(np.abs(diff))
orig_nrm = np.max(np.abs(direct_soln))
rel_nrm = err_nrm / orig_nrm
print(f"{err_nrm=:.4e} {orig_nrm=:.4e} {rel_nrm=:.4e}")

# then plot the solution
# assembler.u = global_soln.copy() # store in assembler for plot
# assembler.plot_disp()

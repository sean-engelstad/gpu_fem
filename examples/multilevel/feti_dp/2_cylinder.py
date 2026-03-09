# demo the FETI-DP method.. with constrained lagrange multipliers..

"""
two-subdomain neumann neumann preconditioner (example from section 3.1 of [FETI–DP, BDDC, and Block Cholesky Method](https://cs.nyu.edu/~widlund/li_widlund_041211.pdf) )
"""

import numpy as np, scipy.sparse as sp
from src import FETIDP_Assembler, MITCShellElement

import sys
sys.path.append("../_poiss_dd/src/")
from krylov import right_pcg_matfree

# fairly stable to higher DOF and thick-independent!
# only thing is would maybe need true multi-level BDDC or multi-level FETI
# to get fully h-independent convergence I bet.. iter count is going up a bit more for higher DOF 

# nxe, nxs = 128, 8
# nxe, nxs = 64, 8
# nxe, nxs = 32, 4
nxe, nxs = 16, 2

# thick = 1e-1
# thick = 1e-2
thick = 1e-3

m, n = 2, 3

radius = 1.0
xs_load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
def xyz_load_fcn(x,y,z):
    th = np.arctan2(y, z)
    dth = th - np.arctan2(-1.0,0)
    s = radius * dth
    return xs_load_fcn(x, s)

ELEMENT = MITCShellElement()
assembler = FETIDP_Assembler(
    ELEMENT=ELEMENT,
    thick=thick,
    nxe=nxe, nxs=nxs, nys=nxs,
    # nxe=4, nxs=2, nys=2, # DEBUG inputs
    # nxe=6, nxs=3, nys=3,
    clamped=True,
    # clamped=False,
    load_fcn=xyz_load_fcn,    
    # would be nice if coarse space could be solved local (but that doesn't seem to work well, need to assemble it with sum of local Schur complements)
    coarse_mode="assembled", 
    # coarse_mode="local",
    geometry="cylinder",
    radius=1.0,
)


assembler.assemble_all()
interface_rhs = assembler.get_interface_rhs() # g_G rhs 
print(f"{np.linalg.norm(interface_rhs)=:.4e}")

# assembler.neumann_solve(rhs)
norm_hist = []
interface_soln, nsteps = right_pcg_matfree(
    assembler, b=interface_rhs,
    rtol=1e-6, atol=1e-20,
    max_iter=200,
    print_freq=3,
    norm_hist=norm_hist,
)

# reconstruct global solution
global_soln = assembler.get_global_solution(interface_soln)

# compare to direct solve
direct_soln = assembler.direct_solve(assembly=False) # cause already assembled

# compute solution error..
diff = global_soln - direct_soln
err_nrm = np.linalg.norm(diff)
print(f"{err_nrm=:.4e}")

# then plot the solution
assembler.u = global_soln.copy() # store in assembler for plot
assembler.plot_disp()
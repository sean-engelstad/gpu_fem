"""
two-subdomain neumann neumann preconditioner (example from section 3.1 of [FETI–DP, BDDC, and Block Cholesky Method](https://cs.nyu.edu/~widlund/li_widlund_041211.pdf) )
"""

import numpy as np, scipy.sparse as sp

from src import PoissonPlateElement, NeumannPlateAssembler
from src import right_pcg_matfree

nxe = 32
# nxe = 64
# nxs = 3
nxs = 4
# nxs = 8

# Neumann-Neumann works only for low # of subdomains.. it's the simplest example
# doesn't have coarse space, no separate treatment of vertices + counting functions
# but definitely a nice start

# NOTE : maye worse convergence with more subdomains because (not h-independent without coarse space)
# but also: Neumann-Neumann can have singular interior subdomains (called floating subdomains) sometimes?
# maybe that's why worse performance? Hence the use of more advanced FETI-DP and BDDC preconditioners

# I didn't demo the FETI method (precursor to FETI-DP cause it is the transpose of Neumann-Neumann precond system, so same performance)

ELEMENT = PoissonPlateElement()
assembler = NeumannPlateAssembler(
    ELEMENT=ELEMENT,
    nxe=nxe, nxs=nxs, nys=nxs,
    # nxe=4, nxs=2, nys=2, # DEBUG inputs
    clamped=False,
    load_fcn=lambda x, y : np.sin(5 * np.pi * x) * np.sin(np.pi * (y**2 + x**2))
)

assembler.assemble_all()
# rhs = assembler.force
interface_rhs = assembler.get_interface_rhs() # g_G rhs 

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
direct_soln = assembler.direct_solve()

# compute solution error..
diff = global_soln - direct_soln
err_nrm = np.linalg.norm(diff)
print(f"{err_nrm=:.4e}")

# then plot the solution
assembler.u = global_soln.copy() # store in assembler for plot
assembler.plot_disp()
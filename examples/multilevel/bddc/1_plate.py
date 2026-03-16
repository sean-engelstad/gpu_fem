# demo the FETI-DP method.. with constrained lagrange multipliers..

"""
two-subdomain neumann neumann preconditioner (example from section 3.1 of [FETI–DP, BDDC, and Block Cholesky Method](https://cs.nyu.edu/~widlund/li_widlund_041211.pdf) )
"""

import numpy as np, scipy.sparse as sp

import sys

sys.path.append("../feti_dp/")
from src import MITCPlateElement, BDDC_Assembler
from src import RichardsonSolver, ILU0Preconditioner, ExactSparseSolver, ILUTPreconditioner

import sys
sys.path.append("../_poiss_dd/src/")
from krylov import right_pcg_matfree

# fairly stable to higher DOF and thick-independent!
# only thing is would maybe need true multi-level BDDC or multi-level FETI
# to get fully h-independent convergence I bet.. iter count is going up a bit more for higher DOF 

# here nxs represents # subdomains not subdomain size
# nxe, nxs = 256, 64
# nxe, nxs = 128, 64 # even works pretty well with small 2x2 subdomains for higher DOF problems! 4x4 fine too..
nxe, nxs = 128, 32 # 4x4 works well also
# nxe, nxs = 128, 16
# nxe, nxs = 128, 8
# nxe, nxs = 64, 16
# nxe, nxs = 64, 8
# nxe, nxs = 32, 4
# nxe, nxs = 32, 8
# nxe, nxs = 16, 4
# nxe, nxs = 6, 3
# nxe, nxs = 4, 2

# thick = 1e-1
thick = 1e-2
# thick = 1e-3


# m, n = 2, 2
# m, n = 2, 3
def load_fcn(_x,_y):
    import math
    theta = math.atan2(_y, _x)
    r = np.sqrt(_x**2 + _y**2)

    return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta)

ELEMENT = MITCPlateElement(
    prolong_mode='standard',
)
assembler = BDDC_Assembler(
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
# print(f"{lam_rhs=}")

# for i in range(lam_rhs.shape[0] // 3):
#     sub_vec = lam_rhs[3*i:(3*i+3)]
#     print(f"lam_rhs, {i=} {sub_vec=}")

# DEBUG
# import matplotlib.pyplot as plt
# for i_sd in range(assembler.num_subdomains):
#     print(f"{i_sd=}/{assembler.num_subdomains=}")
#     sd_kmat = assembler.sd_kmat[i_sd].copy().tobsr(blocksize=(3,3))
#     imap = assembler.sd_node_inv_map[i_sd].copy()
#     for i in range(sd_kmat.indptr.shape[0]-1):
#         for jp in range(sd_kmat.indptr[i], sd_kmat.indptr[i+1]):
#             j = sd_kmat.indices[jp]
#             gi, gj = imap[i], imap[j]
#             print(f"{i_sd=} ({gi},{gj}) {jp=}:")
#             print(sd_kmat.data[jp])

# Svv_bsr = assembler.S_VV.tobsr(blocksize=(3,3))
# print(f"Svv_bsr: \n")
# for i in range(Svv_bsr.indptr.shape[0] - 1):
#     for jp in range(Svv_bsr.indptr[i], Svv_bsr.indptr[i+1]):
#         j = Svv_bsr.indices[jp]
#         gr = assembler.vertex_nodes_global[i]
#         gc = assembler.vertex_nodes_global[j]
#         block = Svv_bsr.data[jp]
#         print(f"Svv_block {jp} ({gr},{gc}) :\n{block}")
# # print(f"{Svv_bsr.data=}")

# for i_sd in range(assembler.num_subdomains):
#     print(f"{i_sd=}/{assembler.num_subdomains=}")
#     sd_force = assembler.sd_force[i_sd].copy()
#     imap = assembler.sd_node_inv_map[i_sd].copy()
#     for inode in range(assembler.sd_nnodes[i_sd]):
#         gnode = imap[inode]
#         sub_vec = sd_force[3*inode:(3*inode+3)]
#         print(f"force {inode=} {gnode=} {sub_vec=}")

# soln1 = assembler.precond_solve(lam_rhs)
# for i in range(soln1.shape[0] // 3):
#     sub_vec = soln1[3*i:(3*i+3)]
#     print(f"pc-solve0, {i=} {sub_vec=}")

def norm(x): 
    # return np.max(np.abs(x))
    return np.linalg.norm(x)

# assembler.neumann_solve(rhs)
norm_hist = []
lam_soln, nsteps = right_pcg_matfree(
    assembler, b=lam_rhs,
    # rtol=1e-6, 
    rtol=1e-9,
    atol=1e-20,
    max_iter=200,
    print_freq=3,
    norm_hist=norm_hist,
)

# for i_sd in range(assembler.num_subdomains):
#     print(f"{i_sd=} {assembler.sd_B_delta[i_sd].toarray()=}")
#     print(f"\t{assembler.sd_edge_dofs[i_sd]=}")
            

# for i in range(lam_soln.shape[0] // 3):
#     sub_vec = lam_soln[3*i:(3*i+3)]
#     print(f"lam_soln, {i=} {sub_vec=}")


# check lambda linear system is solved
interface_res = lam_rhs - assembler.mat_vec(lam_soln)
res_nrm = norm(interface_res)
rhs_nrm = norm(lam_rhs)
rel_solve_nrm = res_nrm / rhs_nrm
print(f"{rel_solve_nrm=:.4e}")

# reconstruct global solution
global_soln = assembler.get_global_solution(lam_soln)

# for i in range(global_soln.shape[0] // 3):
#     sub_vec = global_soln[3*i:(3*i+3)]
#     print(f"global_soln, {i=} {sub_vec=}")

# compare to direct solve
direct_soln = assembler.direct_solve(assembly=False)

# compute solution error..
diff = global_soln - direct_soln
err_nrm = np.max(np.abs(diff))
orig_nrm = np.max(np.abs(direct_soln))
rel_nrm = err_nrm / orig_nrm
print(f"{err_nrm=:.4e} {orig_nrm=:.4e} {rel_nrm=:.4e}")

# compute residual norms
load_nrm = np.linalg.norm(assembler.force.copy())
res_nrm_direct = np.linalg.norm(assembler.force - assembler.kmat.dot(direct_soln))
res_nrm_bddc = np.linalg.norm(assembler.force - assembler.kmat.dot(global_soln))
print(f"{load_nrm=:.4e} {res_nrm_direct=:.4e} {res_nrm_bddc=:.4e}")

# then plot the solution
assembler.u = global_soln.copy() # store in assembler for plot
assembler.plot_disp()

# DEBUG the FETI-DP system
# ========================================

# # DEBUG
# assembler.u = diff
# assembler.plot_disp()
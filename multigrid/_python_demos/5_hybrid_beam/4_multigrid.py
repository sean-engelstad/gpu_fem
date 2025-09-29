# try multigrid with each beam element
import numpy as np
import matplotlib.pyplot as plt
from src import EBAssembler, HybridAssembler, TimoshenkoAssembler
from src import vcycle_solve, block_gauss_seidel_smoother

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=str, default='eb', help="--beam, options: eb, hyb, ts")
parser.add_argument("--nxe", type=int, default=64, help="num max elements in the beam assembler")
parser.add_argument("--SR", type=int, default=100.0, help="beam slenderness")
args = parser.parse_args()

grids = []
nxe_min = 16

# same problem settings for all
# -----------------------------
E = 2e7; b = 1.0; L = 1.0; rho = 1; qmag = 1e4; ys = 4e5; rho_KS = 50.0
# scale by slenderness
thick = L / args.SR
qmag *= thick**3
hvec = np.array([thick] * args.nxe)
load_fcn = lambda x : np.sin(3.0 * np.pi * x / L)


# ----------------------------------
# create all the grids / assemblers
# ----------------------------------

if args.beam == 'eb':
    # make euler-bernoulli beam assemblers for multigrid
    nxe = args.nxe
    while (nxe >= nxe_min):
        eb_grid = EBAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        eb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [eb_grid]
        nxe = nxe // 2


# ----------------------------------
# some prelim multigrid debug
# ----------------------------------

K = grids[0].Kmat.copy()
u = np.zeros_like(grids[0].force)
F = grids[0].force.copy()

# # compare solves of two levels (fine and next coarsest)
# helem0, helem1 = np.array([thick]*grids[0].num_elements), np.array([thick]*grids[1].num_elements)
# u0 = grids[0].solve_forward(helem0)
# u1 = grids[1].solve_forward(helem1)
# fig, ax = plt.subplots(1, 2, figsize=(10, 6))
# i = 0
# # i = 1
# ax[0].plot(grids[0].xvec, u0[i::2])
# ax[1].plot(grids[1].xvec, u1[i::2])
# plt.show()


# F_init_nrm = np.linalg.norm(F)
# u, F = block_gauss_seidel_smoother(K, u, F, num_iter=2, dof_per_node=2)
# F_nrm2 = np.linalg.norm(F)
# print(f"{F_init_nrm=:.2e} {F_nrm2=:.2e}")

# # try restrict defect, good..
# fig, ax = plt.subplots(1, 2, figsize=(10, 6))
# ax[0].plot(grids[0].xvec, F[0::2])
# F_c = grids[1].restrict_defect(F)
# ax[1].plot(grids[1].xvec, F_c[0::2])
# plt.show()

# # try prolongation then smooth
# u_c = grids[1].solve().copy()
# du = grids[0].prolongate(u_c)
# u_f = 

# u, F = block_gauss_seidel_smoother(K, u, F, num_iter=2, dof_per_node=2)
# F_nrm2 = np.linalg.norm(F)
# # print(f"{F_init_nrm=:.2e} {F_nrm2=:.2e}")

# exit()

# ----------------------------------
# solve the multigrid using V-cycle
# ----------------------------------

fine_soln = vcycle_solve(grids, nvcycles=100, pre_smooth=1, post_smooth=1)
# try multigrid with each beam element
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from src import EBAssembler, HybridAssembler, TimoshenkoAssembler
from src import vcycle_solve, block_gauss_seidel_smoother
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=128, help="num max elements in the beam assembler")
parser.add_argument("--nxe_min", type=int, default=16, help="num min elements in the beam assembler")
parser.add_argument("--nsmooth", type=int, default=1, help="num smoothing steps")
args = parser.parse_args()



SR_vec = [1.0, 10.0, 1e2, 1e3, 1e4]
runtimes = {key:[] for key in ['eb', 'ts', 'hyb']}

for SR in SR_vec:

    # same problem settings for all
    # -----------------------------
    E = 2e7; b = 1.0; L = 1.0; rho = 1; qmag = 1e4; ys = 4e5; rho_KS = 50.0
    # scale by slenderness
    
    thick = L / SR
    qmag *= 1e5
    qmag *= thick**3
    hvec = np.array([thick] * args.nxe)
    load_fcn = lambda x : np.sin(3.0 * np.pi * x / L)

    # ----------------------------------
    # create all the grids / assemblers
    # ----------------------------------

    for beam in ['eb', 'ts', 'hyb']:

        grids = []

        if beam == 'eb':
            # make euler-bernoulli beam assemblers for multigrid
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                eb_grid = EBAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                eb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [eb_grid]
                nxe = nxe // 2

        elif beam == 'ts':
            # make timoshenko beam assemblers
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                ts_grid = TimoshenkoAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                ts_grid.red_int = True
                ts_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [ts_grid]
                nxe = nxe // 2

        elif beam == 'hyb':
            # make hybrid beam assemblers
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                hyb_grid = HybridAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                hyb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [hyb_grid]
                nxe = nxe // 2

        # ----------------------------------
        # solve the multigrid using V-cycle
        # ----------------------------------

        start_time = time.time()
        fine_soln, n_iters = vcycle_solve(grids, nvcycles=200, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
            # debug_print=True,
            debug_print=False,
        )
        dt = time.time() - start_time

        # runtimes[beam] += [dt]
        runtimes[beam] += [n_iters]


# now plot each runtimes
import niceplots
plt.style.use(niceplots.get_style())
for beam in ['eb', 'ts', 'hyb']:
    plt.plot(SR_vec, runtimes[beam], 'o-', label=beam)
    
plt.xlabel("Slenderness")
plt.xscale('log')
plt.ylabel("# V-cycles")
# plt.ylabel("Runtime (sec)")
# plt.yscale('log')
plt.legend()
# plt.show()
plt.margins(x=0.05, y=0.05)

# plt.savefig("hybrid-beam-multigrid-times.png", dpi=400)
plt.savefig("hybrid-beam-multigrid-cycles.png", dpi=400)
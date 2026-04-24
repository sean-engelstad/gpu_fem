import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from std_assembler import StandardBeamAssembler
from elem import EulerBernoulliElement, TimoshenkoElement, HierarchicRotHermiteElement, HierarchicDispHermiteElement
# sys.path.append("src/elem")
# from eb_elem import EulerBernoulliElement
from multigrid import VcycleSolver
from smoothers import BlockGaussSeidel, OnedimAddSchwarz, right_pcg2, right_pgmres2, OneDimAddSchwarzVertex2Edges
from multigrid2 import vcycle_solve, VMG

from elem import AlgebraicSubGridScaleElement, OrthogonalSubGridScaleElement
from elem import HellingerReissnerAnsatzElement

# IGA elements
from elem import AsymptoticIsogeometricTimoshenkoElement, HierarchicIsogeometricDispElement
# from iga_assembler import IGABeamAssembler
from iga_assembler2 import IGABeamAssemblerV2


# special vertex-edge DeRham style element
from elem import DeRhamIsogeometricElement
from diga_assembler import DeRhamIGABeamAssembler

from elem import TimoshenkoElement_OptProlong

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=128, help="number of elements")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
parser.add_argument("--nsmooth", type=int, default=1, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
args = parser.parse_args()


# SR_vec = [1.0, 3.0, 10.0, 30.0, 50.0, 1e2, 300.0, 500.0, 1e3, 3e3, 1e4]

SR_vec = [1.0, 3.0, 10.0, 30.0, 50.0, 1e2, 300.0, 500.0, 1e3]
# beam_types = ['eb', 'tsr', 'hhr', 'hhd', 'higd']
# beam_types = ['eb', 'tsr', 'aig', 'hhr', 'hhd', 'higd', 'tsrp']
beam_types = ['eb', 'tsr', 'hhr', 'asgs', 'hra', 'higd', 'tsr-lp']
if args.smoother == 'asw': # only has smoother for asw
    beam_types += ['mig']
runtimes = {key:[] for key in beam_types}

for SR in SR_vec:
    thick = 1.0 / SR

    for elem in beam_types:
            
        is_iga = False
        if elem == 'eb':
            ELEMENT = EulerBernoulliElement()
        elif elem == 'tsr':
            ELEMENT = TimoshenkoElement(reduced_integrated=True)
        elif elem == 'hhr':
            ELEMENT = HierarchicRotHermiteElement(reduced_integrated=False)
        elif elem == 'hhd':
            ELEMENT = HierarchicDispHermiteElement(reduced_integrated=False)
        elif elem == 'higd':
            ELEMENT = HierarchicIsogeometricDispElement(reduced_integrated=False)
            is_iga = True
        elif elem == 'aig':
            # ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=True)
            ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=False)
            is_iga = True
        elif elem == 'mig':
            # derham isogeometric element, special vertex-edge style IGA 2nd order basis that is auto locking-free!
            ELEMENT = DeRhamIsogeometricElement()
            is_iga = True
        elif elem == 'tsr-lp':
            # reduced integrated and with locking-aware prolongation stencil
            ELEMENT = TimoshenkoElement_OptProlong(
                reduced_integrated=True, 
                # locking_aware_prolong=True,
                prolong_mode="local-locking", 
                # locking_aware_prolong=False,
                lam=1e-2
            )
        elif 'asgs' in elem:
            prolong_mode = 'standard'
            ELEMENT = AlgebraicSubGridScaleElement(
                prolong_mode=prolong_mode, omega=0.3
            )
        elif elem == 'hra':
            ELEMENT = HellingerReissnerAnsatzElement(
                schur_complement=True, # at element-level
                # schur_complement=False,
                # these DOF can only be eliminated at global patch level..
                # hra_method = 'disp', 
                # hra_method = 'strain',
                # uses edge DOF for internal strain so can static condensate (idea from this paper Two-field formulations for isogeometric Reissner–Mindlin plates and shells with global and local condensation)
                hra_method='strain-int',
            )

        # ================================
        # make beam assembler
        # ================================

        # clamped = True
        clamped = False # simply supported
        # ASSEMBLER = IGABeamAssembler if is_iga else StandardBeamAssembler
        ASSEMBLER = IGABeamAssemblerV2 if is_iga else StandardBeamAssembler
        if elem == 'mig':
            ASSEMBLER = DeRhamIGABeamAssembler

        
        assembler = ASSEMBLER(
            ELEMENT=ELEMENT,
            nxe=args.nxe,
            thick=thick,
            clamped=clamped,
            split_disp_bc=elem in ['hhd', 'higd'],
            load_fcn = lambda x : np.sin(4.0 * np.pi * x)
        )

        # ================================
        # make multigrid object (optional)
        # ================================

        # make V-cycle solver
        nxe = args.nxe
        nxe_min = args.nxemin
        grids = []
        smoothers = []
        # double_smooth = True
        double_smooth = False

        nsmooth = args.nsmooth
        while (nxe >= nxe_min):
            # print(f"{nxe=}")
            grid = ASSEMBLER(
                ELEMENT=ELEMENT,
                nxe=nxe,
                thick=thick,
                clamped=clamped,
                split_disp_bc=elem in ['hhd', 'higd'],
                load_fcn = lambda x : np.sin(4.0 * np.pi * x)
            )
            grid._assemble_system()
            if elem in ['tsr-lp', 'asgsp', 'hrap']:
                ELEMENT._kmat_cache[nxe // 2] = grid.kmat.tocsr()

            grids += [grid]
            if args.smoother == 'gs':
                smoother = BlockGaussSeidel.from_assembler(
                    grid, omega=args.omega, iters=nsmooth
                )
            elif args.smoother == 'asw':
                omega = args.omega / 2.0 # since 2x smoothing
                if elem == 'mig': # needs custom schwarz smoother with vertex-edge based
                    smoother = OneDimAddSchwarzVertex2Edges.from_assembler(
                        grid, omega=omega, iters=nsmooth,
                        # patch_type="vertex2edges",
                        patch_type="2nodes3edges", # slightly better conv
                    )
                else:
                    smoother = OnedimAddSchwarz.from_assembler(
                        grid, omega=omega, iters=nsmooth, coupled_size=2
                    )
            smoothers += [smoother]
            nxe = nxe // 2
            if double_smooth:
                nsmooth *= 2

        # run MG analysis
        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                            #  line_search=not(args.elem == 'aig'))
                                            # line search sometimes hurts high cond # cases (high defects in prolong)
                                            # line_search=not(elem in ['aig', 'tsr', 'hhd', 'higd']),    
                                            # line_search=False,   
                                            line_search=elem == 'mig', # rarerly helps (but does help for mig elem!)                                      
                                            # line_search=True,
                                            smoothers=smoothers)
        if ncyc >= 99:
            ncyc = np.nan

        runtimes[elem] += [ncyc]

# ==================================
# now plot each runtimes
# ==================================

# import niceplots
# plt.style.use(niceplots.get_style())

# fig, ax = plt.subplots()

# # colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
# colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", 'tab:gray', "k"]
# thick_vec = 1.0 / np.array(SR_vec)

# for ibeam, beam in enumerate(beam_types):
#     ax.plot(thick_vec, runtimes[beam], 'o-' if not(beam == 'hhd') else 'o--', 
#              color=colors[ibeam], label=beam,
#              linewidth=2.5, markersize=7)
    
# # plt.xlabel("Slenderness")
# plt.xlabel("Thickness t/L")
# plt.xscale('log')
# plt.ylabel(f"# V({args.nsmooth},{args.nsmooth})-cycles")
# # plt.ylabel("Runtime (sec)")
# # plt.yscale('log')
# plt.legend()
# # plt.show()
# plt.margins(x=0.05, y=0.05)
# # plt.axis([0.5, 2e3, -1, 30])
# plt.axis([1.0/2e3, 2.0, -1, 50])

# # fix xtick labels
# xticks = ax.get_xticks()
# xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
# ax.set_xticklabels(xlabels)


# ax.set_yticks([0, 10, 20, 30, 40, 50])
# # ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
# ax.set_yticklabels(["0", "10", "20", "30", "40", "50"])


# ax.legend(
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.02),
#     ncol=4,
#     frameon=False,
#     fontsize=12   # ↓ shrink legend font only
# )
# plt.tight_layout()

# # plt.savefig("hybrid-beam-multigrid-times.png", dpi=400)
# plt.savefig(f"beam-multigrid-cycles_{args.smoother}.png", dpi=400)
# plt.savefig(f"beam-multigrid-cycles_{args.smoother}.svg", dpi=400)


import numpy as np
import matplotlib.pyplot as plt
import niceplots

# -----------------------------
# Match style from first plot
# -----------------------------
fs = 22

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

plt.style.use(niceplots.get_style())

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5))

colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1",
          "#3b90b3", "#1a4c60", "tab:gray", "k"]

thick_vec = 1.0 / np.array(SR_vec)

for ibeam, beam in enumerate(beam_types):
    ax.plot(
        thick_vec,
        runtimes[beam],
        # "o-" if beam != "hhd" else "o--",
        'o-',
        color=colors[ibeam],
        label=beam.upper(),
        linewidth=3,
        markersize=8,
    )

ax.set_xlabel(r"Thickness $t/L$")
ax.set_ylabel(rf"# V$({args.nsmooth},{args.nsmooth})$-cycles")

ax.set_xscale("log")
ax.set_xlim(5e-4, 2.0)
ax.set_ylim(-1, 50)

ax.tick_params(axis="both", labelsize=fs)
ax.grid(True)

# set exact x ticks + labels manually
xticks = [1e-3, 1e-2, 1e-1, 1e0]
xlabels = [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

# y ticks
ax.set_yticks([0, 10, 20, 30, 40, 50])

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=True,
    facecolor="white",
    framealpha=1.0,
)

plt.tight_layout()

plt.savefig(f"beam-multigrid-cycles_{args.smoother}.png", dpi=400, bbox_inches="tight")
plt.savefig(f"beam-multigrid-cycles_{args.smoother}.svg", dpi=400, bbox_inches="tight")
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from tacs import pyTACS, constitutive, elements

COMM = MPI.COMM_WORLD
PWD = os.path.dirname(__file__)
OUT_DIR = os.path.join(PWD, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# user inputs
# --------------------------------------------------
BDF_FILE = os.path.join("../../examples/ilu/uCRM", "CRM_box_2nd.bdf")

STRAIN_TYPE = "nonlinear"
ROTATION_TYPE = "linear"

# NLOADS = 5
# NLOADS = 15
NLOADS = 35
LOAD_FACTORS = np.linspace(0.05, 1.0, NLOADS)

# uniform nodal load in global z direction
# LOAD_Z = 10.0
# LOAD_Z = 20.0
# LOAD_Z = 40.0
# LOAD_Z = 1e3
LOAD_Z = 4e3

# material / shell thickness
RHO = 2500.0
E = 70e9
NU = 0.3
YS = 350e6
THICKNESS = 1e-1

# need higher shell thickness to suppress panel buckling + enable larger overall deflections of wingbox..

PRINT = True

# --------------------------------------------------
# choose shell element type
# --------------------------------------------------
elementType = None
if STRAIN_TYPE == "linear":
    if ROTATION_TYPE == "linear":
        elementType = elements.Quad4Shell
    elif ROTATION_TYPE == "quadratic":
        elementType = elements.Quad4ShellModRot
    elif ROTATION_TYPE == "quaternion":
        elementType = elements.Quad4ShellQuaternion
elif STRAIN_TYPE == "nonlinear":
    if ROTATION_TYPE == "linear":
        elementType = elements.Quad4NonlinearShell
    elif ROTATION_TYPE == "quadratic":
        elementType = elements.Quad4NonlinearShellModRot
    elif ROTATION_TYPE == "quaternion":
        elementType = elements.Quad4NonlinearShellQuaternion

if elementType is None:
    raise RuntimeError("Invalid STRAIN_TYPE / ROTATION_TYPE combination.")

both_linear = STRAIN_TYPE == "linear" and ROTATION_TYPE == "linear"

# --------------------------------------------------
# pyTACS setup
# --------------------------------------------------
structOptions = {
    "printtiming": (not both_linear) and PRINT,
}
FEAAssembler = pyTACS(BDF_FILE, options=structOptions, comm=COMM)

def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    matProps = constitutive.MaterialProperties(
        rho=RHO, E=E, nu=NU, ys=YS
    )
    con = constitutive.IsoShellConstitutive(
        matProps,
        t=THICKNESS,
        tNum=dvNum,
        tlb=1e-2 * THICKNESS,
        tub=1e2 * THICKNESS,
    )
    transform = None
    element = elementType(transform, con)
    tScale = [10.0]
    return element, tScale

FEAAssembler.initialize(elemCallBack)

probOptions = {
    "printTiming": (not both_linear) and PRINT,
    "printLevel": 1,
}
newtonOptions = {"useEW": True, "MaxLinIters": 10}
continuationOptions = {
    "CoarseRelTol": 1e-3,
    "InitialStep": 0.2,
    "UsePredictor": True,
    "NumPredictorStates": 7,
}

forceProblem = FEAAssembler.createStaticProblem("CRM_Z_Load", options=probOptions)
try:
    forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)
    forceProblem.nonlinearSolver.setOptions(continuationOptions)
except AttributeError:
    pass

# --------------------------------------------------
# geometry info from BDF
# --------------------------------------------------
bdfInfo = FEAAssembler.getBDFInfo()
bdfInfo.cross_reference()
nodeCoords = bdfInfo.get_xyz_in_coord()
nastranNodeNums = list(bdfInfo.node_ids)

if COMM.rank == 0:
    print(f"Loaded BDF: {BDF_FILE}")
    print(f"nnodes = {len(nastranNodeNums)}")

# --------------------------------------------------
# apply uniform z nodal load
# --------------------------------------------------
for node_id in nastranNodeNums:
    forceProblem.addLoadToNodes(
        [node_id],
        [0.0, 0.0, LOAD_Z, 0.0, 0.0, 0.0],
        nastranOrdering=True,
    )

ForceVec = np.copy(forceProblem.F_array)

# --------------------------------------------------
# choose a response quantity for plotting
# use max |uz| over all nodes
# --------------------------------------------------
max_abs_uz_hist = []
mean_uz_hist = []

# optional: choose a "tip" node as the node with maximum x
xcoords = nodeCoords[:, 0]
tip_local_idx = int(np.argmax(xcoords))
tip_node_id_local = nastranNodeNums[tip_local_idx]
tip_xyz_local = nodeCoords[tip_local_idx].copy()

# --------------------------------------------------
# solve at each load factor + write VTK/F5 each step
# --------------------------------------------------
# for i, LOAD_FACTOR in enumerate(LOAD_FACTORS):
#     Fext = LOAD_FACTOR * ForceVec
#     forceProblem.solve(Fext=Fext)

i = 0
for LOAD_FACTOR in LOAD_FACTORS:
    Fext = (LOAD_FACTOR - LOAD_FACTORS[-1]) * ForceVec
    forceProblem.solve(Fext=Fext)

    if PRINT and COMM.rank == 0:
        print(f"Solved load factor {i+1}/{len(LOAD_FACTORS)} : {LOAD_FACTOR:.4f}")
    i += 1

    baseName = f"crm_nl" #.replace(".", "p")
    forceProblem.writeSolution(outputDir=OUT_DIR, baseName=baseName)

    disps = forceProblem.u_array
    uz = disps[2::6]

    local_max_abs_uz = np.max(np.abs(uz)) if uz.size > 0 else 0.0
    global_max_abs_uz = COMM.reduce(local_max_abs_uz, op=MPI.MAX, root=0)

    local_sum_uz = np.sum(uz)
    local_n = len(uz)
    global_sum_uz = COMM.reduce(local_sum_uz, op=MPI.SUM, root=0)
    global_n = COMM.reduce(local_n, op=MPI.SUM, root=0)

    if COMM.rank == 0:
        max_abs_uz_hist.append(global_max_abs_uz)
        mean_uz_hist.append(global_sum_uz / max(global_n, 1))


# # --------------------------------------------------
# # plots
# # --------------------------------------------------
# if COMM.rank == 0:
#     try:
#         import niceplots
#         plt.style.use(niceplots.get_style())
#     except Exception:
#         pass

#     # -----------------------
#     # shared plotting style
#     # -----------------------
#     fs = 22
#     plt.rcParams.update({
#         "font.size": fs,
#         "axes.titlesize": fs + 2,
#         "axes.labelsize": fs,
#         "xtick.labelsize": fs - 1,
#         "ytick.labelsize": fs - 1,
#         "legend.fontsize": fs - 2,
#         "axes.linewidth": 2.0,
#         "lines.linewidth": 3.0,
#     })

#     def make_response_plot(
#         x,
#         y,
#         xlabel,
#         ylabel,
#         filename,
#         marker="o",
#         label=None,
#     ):
#         fig, ax = plt.subplots(figsize=(9.0, 6.8))

#         ax.plot(
#             x,
#             y,
#             marker=marker,
#             linestyle="-",   # solid line only
#             markersize=8,
#             markeredgewidth=1.2,
#             label=label,
#         )

#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)

#         # cleaner, lighter grid
#         ax.grid(True, which="major", alpha=0.25)
#         ax.grid(False, which="minor")

#         # slightly tighter margins so the data fills the panel better
#         ax.margins(x=0.03, y=0.08)

#         # nicer legend styling
#         if label is not None:
#             leg = ax.legend(
#                 loc="best",
#                 frameon=True,
#                 fancybox=True,
#                 framealpha=0.95,
#                 borderpad=0.4,
#                 handlelength=2.2,
#             )
#             leg.get_frame().set_linewidth(0.0)

#         fig.tight_layout()
#         fig.savefig(os.path.join(OUT_DIR, filename), dpi=400)
#         plt.close(fig)

#     # max displacement vs load factor
#     make_response_plot(
#         x=LOAD_FACTORS,
#         y=max_abs_uz_hist,
#         xlabel="Load Factor",
#         ylabel=r"Maximum $|u_z|$",
#         filename="crm_nonlinear_load_disp.png",
#         marker="o",
#         label=r"max $|u_z|$",
#     )

#     # mean displacement vs load factor
#     make_response_plot(
#         x=LOAD_FACTORS,
#         y=mean_uz_hist,
#         xlabel="Load Factor",
#         ylabel=r"Mean $u_z$",
#         filename="crm_nonlinear_mean_uz.png",
#         marker="s",
#         label=r"mean $u_z$",
#     )

#     print("\n============================\n")
#     print(f"{LOAD_FACTORS=}")
#     print(f"{np.array(max_abs_uz_hist)=}")
#     print(f"{np.array(mean_uz_hist)=}")
#     print(f"VTK/F5 written to: {OUT_DIR}")

# --------------------------------------------------
# plots
# --------------------------------------------------
if COMM.rank == 0:
    try:
        import niceplots
        plt.style.use(niceplots.get_style())
    except Exception:
        pass

    # -----------------------
    # BIGGER fonts (much bigger)
    # -----------------------
    fs = 28
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs + 4,
        "axes.labelsize": fs + 2,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs - 2,
        "axes.linewidth": 2.5,
        "lines.linewidth": 3.5,
    })

    # def make_response_plot(
    #     x,
    #     y,
    #     xlabel,
    #     ylabel,
    #     filename,
    #     marker="o",
    #     label=None,
    # ):
    #     x = np.array(x)
    #     y = np.array(y)

    #     fig, ax = plt.subplots(figsize=(10.5, 7.5))

    #     # -----------------------
    #     # nonlinear curve
    #     # -----------------------
    #     ax.plot(
    #         x,
    #         y,
    #         marker=marker,
    #         linestyle="-",
    #         markersize=9,
    #         markeredgewidth=1.4,
    #         label=label,
    #     )

    #     # -----------------------
    #     # linear estimate (first two points)
    #     # -----------------------
    #     if len(x) >= 2:
    #         # slope from first two points
    #         slope = (y[1] - y[0]) / (x[1] - x[0])

    #         # extend to zero
    #         x_lin = np.concatenate(([0.0], x))
    #         y_lin = slope * x_lin

    #         ax.plot(
    #             x_lin,
    #             y_lin,
    #             linestyle="-",
    #             linewidth=3.0,
    #             alpha=0.9,
    #             label="FEA-LIN",
    #         )

    #     # -----------------------
    #     # axes + grid
    #     # -----------------------
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)

    #     ax.grid(True, which="major", alpha=0.25)
    #     ax.grid(False, which="minor")

    #     ax.margins(x=0.03, y=0.08)

    #     # -----------------------
    #     # legend (clean)
    #     # -----------------------
    #     if label is not None:
    #         leg = ax.legend(
    #             loc="best",
    #             frameon=True,
    #             fancybox=True,
    #             framealpha=0.95,
    #             borderpad=0.4,
    #             handlelength=2.2,
    #         )
    #         leg.get_frame().set_linewidth(0.0)

    #     fig.tight_layout()
    #     fig.savefig(os.path.join(OUT_DIR, filename), dpi=400)
    #     plt.close(fig)

    def make_response_plot(
        x,
        y,
        xlabel,
        ylabel,
        filename,
        marker="o",
        label=None,
    ):
        x = np.array(x)
        y = np.array(y)

        fig, ax = plt.subplots(figsize=(10.5, 7.5))

        # -----------------------
        # linear estimate FIRST (so it's underneath)
        # -----------------------
        if len(x) >= 2:
            slope = (y[1] - y[0]) / (x[1] - x[0])

            x_lin = np.concatenate(([0.0], x))
            y_lin = slope * x_lin

            ax.plot(
                x_lin,
                y_lin,
                linestyle="--",     # dashed as requested
                linewidth=3.0,
                alpha=0.9,
                label="FEA-LIN",
                zorder=1,           # explicitly underneath
            )

        # -----------------------
        # nonlinear curve ON TOP
        # -----------------------
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle="-",
            markersize=9,
            markeredgewidth=1.4,
            label=label,
            zorder=3,              # explicitly on top
        )

        # -----------------------
        # axes + grid
        # -----------------------
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid(True, which="major", alpha=0.25)
        ax.grid(False, which="minor")

        ax.margins(x=0.03, y=0.08)

        ax.set_ylim(bottom=-1.0, top=13.0)

        # -----------------------
        # legend
        # -----------------------
        if label is not None:
            leg = ax.legend(
                loc="best",
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                borderpad=0.4,
                handlelength=2.2,
            )
            leg.get_frame().set_linewidth(0.0)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=400)
        plt.close(fig)

    # -----------------------
    # max displacement plot
    # -----------------------
    make_response_plot(
        x=LOAD_FACTORS,
        y=max_abs_uz_hist,
        xlabel="Load Factor",
        ylabel="Max w Displacement (m)",
        filename="crm_nonlinear_load_disp.png",
        marker="o",
        label="FEA-NL",
    )

    # # -----------------------
    # # mean displacement plot
    # # -----------------------
    # make_response_plot(
    #     x=LOAD_FACTORS,
    #     y=mean_uz_hist,
    #     xlabel="Load Factor",
    #     ylabel=r"Mean $u_z$",
    #     filename="crm_nonlinear_mean_uz.png",
    #     marker="s",
    #     label=r"FEA-NL",
    # )

    print("\n============================\n")
    print(f"{LOAD_FACTORS=}")
    print(f"{np.array(max_abs_uz_hist)=}")
    print(f"{np.array(mean_uz_hist)=}")
    print(f"VTK/F5 written to: {OUT_DIR}")
import sys
sys.path.append("_src/") # contains gpusolver

import platemultigrid
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse

# ndvs_per_side = 4
# ndvs_per_side = 16
ndvs_per_side = 32
# ndvs_per_side = 64
# ndvs_per_side = 128

# solver_type = "gmg_cp"
# solver_type = "gmg_asw"
# solver_type = "bddc_lu"
solver_type = "lu"

omega = None
load_mag = 1.6e7

if solver_type == "gmg_cp":
    SOLVER = platemultigrid.Linear_GMGCP_PlateSolver
    omega, nsmooth = 0.85, 1
    out_file = "out/plate_mg.vtk"
elif solver_type == "gmg_asw":
    SOLVER = platemultigrid.Linear_GMGASW_PlateSolver
    omega, nsmooth = 0.25, 2
    out_file = "out/plate_mg.vtk"
elif solver_type == "bddc_lu":
    SOLVER = platemultigrid.Linear_BDDCLU_PlateSolver
    omega, nsmooth = 1.0, 1
    out_file = "out/plate_bddc.vtk"
    # load_mag *= (2.2 / 8.0)
    load_mag *= 2.0
elif solver_type == "lu":
    SOLVER = platemultigrid.Linear_LU_PlateSolver
    omega, nsmooth = 1.0, 1
    out_file = "out/plate_lu.vtk"

# setup GPU solver
solver = SOLVER(
    rhoKS=100.0,
    safety_factor=1.5,
    # load_mag=5e5,
    # load_mag=1e6,
    # load_mag=2e6,
    # load_mag=4e6,
    # load_mag=8e6,
    load_mag=load_mag,
    # omega=0.3, # much faster
    omega=omega,
    nsmooth=nsmooth,
    # in_plane_frac=0.05,
    # in_plane_frac=0.1,
    in_plane_frac=0.6,
    # in_plane_frac=0.15,
    # in_plane_frac=0.25,
    # in_plane_frac=0.25,
    # in_plane_frac=0.75,
    # nsmooth=4,
    # omega=0.85,
    # nxe=512,
    # nxe=256,
    nxe=128,
    # nxe=64,
    # nxe=32,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
    # SR=50.0, # slenderness
    ORDER=8,
    # ORDER=4,
    rtol=1e-6, # almost want to have high rtol like this only later in the optimization tbh..
    Lx=1.0,
    # Lx=2.0, # both dimensions
    # Lx=4.0,
    # Lx=8.0,
    # Lx=16.0,
    # rtol=1e-5,
    # rtol=1e-4, # faster optimization, but will it converge though?
    print=True, # always on for these kinds of tests..
)

def load_design(filename):
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip first line (mass/ksfail)
    return np.array([float(line.split()[1]) for line in lines])

# linear optimal
filename = "out/lin_gpu_opt.txt"
# nonlinear optimal
# filename = "out/nl_gpu_opt.txt"

# load linear optimal design..
if os.path.exists(filename):
    lin_opt_design = load_design(filename)
    x0 = lin_opt_design.copy()

else:
    # init dvs
    ndvs = solver.get_num_dvs()
    # x0 = np.array([3e-2]*ndvs)
    x0 = np.array([2e-2]*ndvs) # very slender
    # x0 = np.array([2e-3] * ndvs)
    # x0 = np.array([1e-3]*ndvs) # very slender

# debug writing DVs to not same values..
print("set design variables in python script")
solver.set_design_variables(x0)
print("\tdone with set design variables in python script")
solver.solve()
ksfail = solver.evalFunction("ksfailure")
print(f"{ksfail=:.2e}")
solver.writeSolution(out_file)


# # debug writing DVs to not same values..
# solver.set_design_variables(x0)
# solver.solve()
# mass = solver.evalFunction("mass")
# ksfail = solver.evalFunction("ksfailure")
# print(f"{mass=:.4e} {ksfail=:.4e}")
# solver.writeSolution("out/plate_init.vtk")
# # try setting values again, check gradient
# mass_grad = solver.evalFunctionSens("mass")
# print("evalFunctoinSens\n")
# ksfail_grad = solver.evalFunctionSens("ksfailure")
# print(f"{mass_grad=}")
# print(f"{ksfail_grad=}")
# # exit()


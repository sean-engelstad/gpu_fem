"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

from funtofem import *
import numpy as np
import time
start_time = time.time()
import argparse
import shutil, os
from pyNastran.bdf.bdf import BDF

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--level",
    type=int,
    choices=[0, 1, 2, 3, 4, 5],
    required=True,
    help="Mesh refinement level: 0 = coarsest, 1 = finer, 2 etc..",
)
args = parent_parser.parse_args()

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs, pytacs
import os

# two versions of fuselage
# PREFIX = "cylinder_fuselage"
PREFIX = "box_fuselage"

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, f"{PREFIX}.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel('model')
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct",
    active_procs=[0],
    verbosity=1,
)

# finer level 0
if args.level == 0:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=2,
        edge_pt_max=4,
        global_mesh_size=0.2,
        max_surf_offset=0.16,
        max_dihedral_angle=40,
    ).register_to(
        tacs_model
    )

if args.level == 1:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=5,
        edge_pt_max=8,
        global_mesh_size=0.05,
        max_surf_offset=0.03,
        max_dihedral_angle=10,
    ).register_to(
        tacs_model
    )

elif args.level == 2:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=12,
        edge_pt_max=15,
        global_mesh_size=0.01,
        max_surf_offset=0.01,
        max_dihedral_angle=5,
    ).register_to(
        tacs_model
    )

elif args.level == 3:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=26,
        edge_pt_max=30,
        global_mesh_size=0.005,
        max_surf_offset=0.005,
        max_dihedral_angle=3,
    ).register_to(
        tacs_model
    )


elif args.level == 4:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=54,
        edge_pt_max=60,
        global_mesh_size=0.0025,
        max_surf_offset=0.0025,
        max_dihedral_angle=1,
    ).register_to(
        tacs_model
    )

elif args.level == 5:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=108,
        edge_pt_max=120,
        global_mesh_size=0.00125,
        max_surf_offset=0.00125,
        max_dihedral_angle=0.8,
    ).register_to(
        tacs_model
    )


f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim

egads_aim = tacs_model.mesh_aim

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("cylinder", boundary=2)
# aerothermoelastic

null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

def num_to_padstr(mynum):
    # if mynum < 10:
    #     return "0" + str(mynum)
    # else:
    #     return str(mynum)
    return str(mynum) # not really needed tbh.. as long as it does lOML then uOML, then ribs and spars, etc.

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = ["uskin1", "uskin2", "lskin1", "lskin2", "lespar1", "lespar2", "tespar1", "tespar2", "rib"]

# print(f"{component_groups=}")
# exit()

# delim = "-"
delim = "_" # had to change to _ so pynastran doesn't complain about prop to DV definitions (just for this case)

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    panel_length = 1.0
    Variable.structural(
        f"{comp}{delim}" + TacsSteadyInterface.LENGTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    # stiffener pitch variable
    Variable.structural(f"{comp}{delim}spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}{delim}T", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}{delim}sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}{delim}sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    Variable.structural(
        f"{comp}{delim}" + TacsSteadyInterface.WIDTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

# AOB - all clamped DOF, since BDDC only supports clamped nodes rn
caps2tacs.PinConstraint("bndry", dof_constraint=123456).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# -------------------------------------------------
# here we have to use pynastran to combine BDF and DAT file since the TACS C++ MeshLoader (from BDF)
# can't read BDF + DAT file pair like pynastran..

orig_bdf = os.path.join("capsStruct_0", "Scratch", "tacs", "tacs.dat")
final_bdf = f"{PREFIX}_L" + str(args.level) + ".bdf"

model = BDF()
model.read_bdf(orig_bdf, xref=True)

# Write out as a single combined BDF (orig)
model.write_bdf(final_bdf, size=16) #, sort_ids=False)

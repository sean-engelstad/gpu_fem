"""
fro funtofem/examples/framework/1_*py naca wing example
"""

from funtofem import *
from tacs import caps2tacs
import openmdao.api as om
from mpi4py import MPI
import shutil, os
from pyNastran.bdf.bdf import BDF

# --------------------------------------------------------------#
# Setup CAPS Problem and FUNtoFEM model
# --------------------------------------------------------------#
comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("tacs_wing")
wing = Body.aeroelastic(
    "wing"
)  # says aeroelastic but not coupled, may want to make new classmethods later...

# define the Tacs model
tacs_model = caps2tacs.TacsModel.build(csm_file="large_naca_wing.csm", comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=15,
    edge_pt_max=20,
    global_mesh_size=0.01,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
tacs_aim = tacs_model.tacs_aim

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

# setup the thickness design variables + automatic shell properties
# using Composite functions, this part has to go after all funtofem variables are defined...
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

init_thickness = 0.08
for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

# add constraints and loads
caps2tacs.PinConstraint("root").register_to(tacs_model)
# caps2tacs.GridForce("OML", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model) # don't need that here

# run the tacs model setup and register to the funtofem model
f2f_model.structural = tacs_model

# register the funtofem Body to the model
wing.register_to(f2f_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=100)
Function.ksfailure(ks_weight=10.0).optimize(
    scale=30.0, upper=0.267, objective=False, plot=True
).register_to(tacs_scenario)
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
tacs_scenario.register_to(f2f_model)

# make the composite functions for adjacency constraints
variables = f2f_model.get_variables()
adjacency_ratio = 2.0
adjacency_scale = 10.0
for irib in range(
    1, nribs
):  # not (1, nribs+1) bc we want to do one less since we're doing nribs-1 pairs
    left_rib = f2f_model.get_variables(names=f"rib{irib}")
    right_rib = f2f_model.get_variables(names=f"rib{irib+1}")
    # make a composite function for relative diff in rib thicknesses
    adjacency_rib_constr = (left_rib - right_rib) / left_rib
    adjacency_rib_constr.set_name(f"rib{irib}-{irib+1}").optimize(
        lower=-adjacency_ratio, upper=adjacency_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for ispar in range(1, nspars):
    left_spar = f2f_model.get_variables(names=f"spar{ispar}")
    right_spar = f2f_model.get_variables(names=f"spar{ispar+1}")
    # make a composite function for relative diff in spar thicknesses
    adjacency_spar_constr = (left_spar - right_spar) / left_spar
    adjacency_spar_constr.set_name(f"spar{ispar}-{ispar+1}").optimize(
        lower=-adjacency_ratio, upper=adjacency_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for iOML in range(1, nOML):
    left_OML = f2f_model.get_variables(names=f"OML{iOML}")
    right_OML = f2f_model.get_variables(names=f"OML{iOML+1}")
    # make a composite function for relative diff in OML thicknesses
    adj_OML_constr = (left_OML - right_OML) / left_OML
    adj_OML_constr.set_name(f"OML{iOML}-{iOML+1}").optimize(
        lower=-adjacency_ratio, upper=adjacency_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

# make the BDF and DAT file for TACS structural analysis
tacs_aim.setup_aim()
tacs_aim.pre_analysis()


# -------------------------------------------------
# here we have to use pynastran to combine BDF and DAT file since the TACS C++ MeshLoader (from BDF)
# can't read BDF + DAT file pair like pynastran..

orig_bdf = os.path.join("capsStruct_0", "Scratch", "tacs", "tacs.dat")
final_bdf = "naca_wing.bdf"

model = BDF()
model.read_bdf(orig_bdf, xref=True)

# Write out as a single combined BDF
model.write_bdf(final_bdf, size=16)
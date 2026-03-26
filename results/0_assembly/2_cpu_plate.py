# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up using the
# TACS assembler directly as opposed to using the pyTACS user interface.
#
# This version measures only:
#   1) factorization time
#   2) triangular solve time
#   3) matrix-vector product time

import numpy as np
import os
import time
import csv
from mpi4py import MPI
from tacs import TACS, elements, constitutive

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=100, help="# elements along each direction")
parser.add_argument(
    "--fill",
    type=float,
    default=0,
    help="ILU fill level. Use 0 for ILU(0), large value / default-style usage for LU if desired",
)
parser.add_argument(
    "--ordering",
    type=str,
    default="RCM",
    choices=["AMD", "RCM", "NATURAL"],
    help="Matrix ordering for createSchurMat",
)
parser.add_argument(
    "--csv",
    type=str,
    default="timings.csv",
    help="CSV file to append timings to",
)
args = parser.parse_args()


def get_ordering(name):
    if name == "AMD":
        return TACS.AMD_ORDER
    elif name == "RCM":
        return TACS.RCM_ORDER
    elif name == "NATURAL":
        return TACS.NATURAL_ORDER
    raise ValueError(f"Unknown ordering {name}")


# ------------------------------------------------------------
# Mesh generation
# ------------------------------------------------------------
os.system(f"python _gen_plate_mesh.py --nxe {args.nxe}")

# ------------------------------------------------------------
# Load structural mesh from BDF file
# ------------------------------------------------------------
bdfFile = "plate.bdf"
tacs_comm = MPI.COMM_WORLD
rank = tacs_comm.rank

struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile(bdfFile)

# ------------------------------------------------------------
# Constitutive / elements
# ------------------------------------------------------------
rho = 2500.0
E = 70e9
nu = 0.3
ys = 350e6
min_thickness = 0.002
max_thickness = 2.0
thickness = 1.0

num_components = struct_mesh.getNumComponents()
for i in range(num_components):
    descriptor = struct_mesh.getElementDescript(i)

    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    stiff = constitutive.IsoShellConstitutive(
        prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i
    )

    element = None
    transform = None
    if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
        element = elements.Quad4Shell(transform, stiff)

    struct_mesh.setElement(i, element)

# ------------------------------------------------------------
# Create assembler
# ------------------------------------------------------------
tacs = struct_mesh.createTACS(6)

x = tacs.createDesignVec()
tacs.getDesignVars(x)

X = tacs.createNodeVec()
tacs.getNodes(X)
tacs.setNodes(X)

# ------------------------------------------------------------
# Create RHS
# ------------------------------------------------------------
forces = tacs.createVec()
force_array = forces.getArray()
force_array[2::6] += 100.0
tacs.applyBCs(forces)

res = tacs.createVec()
ans = tacs.createVec()

# ------------------------------------------------------------
# Create matrix and preconditioner
# ------------------------------------------------------------
ordering = get_ordering(args.ordering)

mat_start = time.time()
mat = tacs.createSchurMat(ordering)
mat_create_time = time.time() - mat_start

pc_start = time.time()
pc = TACS.Pc(mat, lev_fill=args.fill)
pc_create_time = time.time() - pc_start

# ------------------------------------------------------------
# Assemble Jacobian
# ------------------------------------------------------------
alpha = 1.0
beta = 0.0
gamma = 0.0

asm_start = time.time()
tacs.zeroVariables()
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
assembly_time = time.time() - asm_start

# ------------------------------------------------------------
# Factorization timing
# ------------------------------------------------------------
factor_start = time.time()
pc.factor()
factor_time = time.time() - factor_start

# ------------------------------------------------------------
# Triangular solve timing
# ------------------------------------------------------------
tri_start = time.time()
pc.applyFactor(forces, ans)
tri_solve_time = time.time() - tri_start

# ------------------------------------------------------------
# Matrix-vector product timing
# ------------------------------------------------------------
matvec_start = time.time()
mat.mult(ans, res)
mat_vec_time = time.time() - matvec_start

# ------------------------------------------------------------
# Optional output for visualization
# ------------------------------------------------------------
flag = (
    TACS.OUTPUT_CONNECTIVITY
    | TACS.OUTPUT_NODES
    | TACS.OUTPUT_DISPLACEMENTS
    | TACS.OUTPUT_STRAINS
    | TACS.OUTPUT_STRESSES
    | TACS.OUTPUT_EXTRAS
    | TACS.OUTPUT_LOADS
)
f5 = TACS.ToFH5(tacs, TACS.BEAM_OR_SHELL_ELEMENT, flag)
f5.writeToFile("plate.f5")

# ------------------------------------------------------------
# Report / CSV
# ------------------------------------------------------------
dof = (args.nxe + 1) ** 2 * 6

# crude label
fill_name = f"ILU({args.fill})" if args.fill < 1000 else "LU"

if rank == 0:
    print("\nTiming results")
    print("--------------")
    print(f"nproc           = {tacs_comm.size}")
    print(f"nxe             = {args.nxe}")
    print(f"dof             = {dof}")
    # print(f"ordering        = {args.ordering}")
    print(f"fill            = {fill_name}")
    # print(f"create mat      = {mat_create_time:.6e} s")
    # print(f"create pc       = {pc_create_time:.6e} s")
    # print(f"assembly        = {assembly_time:.6e} s")
    print(f"factor          = {factor_time:.6e} s")
    print(f"tri solve       = {tri_solve_time:.6e} s")
    print(f"mat vec         = {mat_vec_time:.6e} s")
    print()

    file_exists = os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(
                [
                    "nproc",
                    "nxe",
                    "dof",
                    "ordering",
                    "fill",
                    "mat_create_s",
                    "pc_create_s",
                    "assembly_s",
                    "factor_s",
                    "tri_solve_s",
                    "mat_vec_s",
                ]
            )
        writer.writerow(
            [
                tacs_comm.size,
                args.nxe,
                dof,
                args.ordering,
                fill_name,
                f"{mat_create_time:.6e}",
                f"{pc_create_time:.6e}",
                f"{assembly_time:.6e}",
                f"{factor_time:.6e}",
                f"{tri_solve_time:.6e}",
                f"{mat_vec_time:.6e}",
            ]
        )
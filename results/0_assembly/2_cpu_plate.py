# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up using the
# TACS assembler directly as opposed to using the pyTACS user interface.
#
# This version measures only:
#   1) factorization time
#   2) triangular solve time
#   3) matrix-vector product time
#
# MPI timing notes:
#   - place barriers before/after each timed region
#   - measure with MPI.Wtime()
#   - report the maximum time across all ranks

import numpy as np
import os
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


def timed_max(comm, func):
    """
    Time a callable with MPI barriers before and after.
    Return the maximum elapsed wall time across all ranks.
    """
    comm.Barrier()
    t0 = MPI.Wtime()
    func()
    comm.Barrier()
    dt_local = MPI.Wtime() - t0
    dt_max = comm.allreduce(dt_local, op=MPI.MAX)
    return dt_max


# ------------------------------------------------------------
# MPI setup
# ------------------------------------------------------------
tacs_comm = MPI.COMM_WORLD
rank = tacs_comm.rank

# ------------------------------------------------------------
# Mesh generation
# ------------------------------------------------------------
if rank == 0:
    ret = os.system(f"python _gen_plate_mesh.py --nxe {args.nxe}")
    if ret != 0:
        raise RuntimeError("Mesh generation failed: _gen_plate_mesh.py")
tacs_comm.Barrier()

# ------------------------------------------------------------
# Load structural mesh from BDF file
# ------------------------------------------------------------
bdfFile = "plate.bdf"

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

def create_mat():
    global mat
    mat = tacs.createSchurMat(ordering)

mat = None
mat_create_time = timed_max(tacs_comm, create_mat)

def create_pc():
    global pc
    pc = TACS.Pc(mat, lev_fill=args.fill)

pc = None
pc_create_time = timed_max(tacs_comm, create_pc)

# ------------------------------------------------------------
# Assemble Jacobian
# ------------------------------------------------------------
alpha = 1.0
beta = 0.0
gamma = 0.0

def assemble_jacobian():
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)

assembly_time = timed_max(tacs_comm, assemble_jacobian)

# ------------------------------------------------------------
# Factorization timing
# ------------------------------------------------------------
factor_time = timed_max(tacs_comm, lambda: pc.factor())

# ------------------------------------------------------------
# Triangular solve timing
# ------------------------------------------------------------
tri_solve_time = timed_max(tacs_comm, lambda: pc.applyFactor(forces, ans))

# ------------------------------------------------------------
# Matrix-vector product timing
# ------------------------------------------------------------
mat_vec_time = timed_max(tacs_comm, lambda: mat.mult(ans, res))

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

# avoid every rank writing the same file unless ToFH5 requires collective behavior
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
    print(f"fill            = {fill_name}")
    print(f"create mat      = {mat_create_time:.6e} s")
    print(f"create pc       = {pc_create_time:.6e} s")
    print(f"assembly        = {assembly_time:.6e} s")
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
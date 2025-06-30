import argparse
import os
import numpy as np
import os, time
from mpi4py import MPI
from tacs import TACS, elements, constitutive, pyTACS

# -----------------------------
# Argparse Options:
# --iluk       : ILU(k) fill level (int, default=0)
# --case       : 'plate' or 'ucrm' (required)
# --nonlinear  : Flag to enable nonlinear solve (default: linear)
# --ordering   : Matrix ordering method: RCM, ND, AMD, Qorder (default=RCM)
# --nxe        : Number of elements in x-direction for plate (int, default=16)
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FEM Solver Options")

    # ILU(k) fill level
    parser.add_argument('--iluk', type=int, default=0,
                        help='ILU(k) fill level (default: 0)')

    # Case: plate or uCRM
    parser.add_argument('--case', choices=['plate', 'ucrm'], default='plate',
                        help='Select the test case (plate or ucrm)')

    # Nonlinear flag
    parser.add_argument('--nonlinear', action='store_true',
                        help='Enable nonlinear solver (default: linear)')

    # Matrix reordering method
    parser.add_argument('--ordering', choices=['RCM', 'ND', 'AMD', 'Qorder'], default='AMD',
                        help='Matrix reordering method (default: RCM)')

    # Number of elements in x-direction (only relevant for plate case)
    parser.add_argument('--nxe', type=int, default=80,
                        help='Number of elements in x-direction (only for plate case, default: 16)')

    return parser.parse_args()

def make_plate_bdf(args) -> str:
    os.system(f"python _gen_plate_mesh.py --nxe {args.nxe}")
    return

def run_linear_static(args):
    if args.case == "plate":
        bdfFile = "out/plate.bdf"
    else: # uCRM
        bdfFile = "../uCRM/CRM_box_2nd.bdf"

    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdfFile)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    E = 70e9  # elastic modulus, Pa
    nu = 0.3  # poisson's ratio
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.002
    max_thickness = 0.20
    thickness = 0.02

    # Loop over components, creating stiffness and element object for each
    num_components = struct_mesh.getNumComponents()
    for i in range(num_components):
        descriptor = struct_mesh.getElementDescript(i)
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        # Set one thickness dv for every component
        con = constitutive.IsoShellConstitutive(
            prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i
        )

        element = None
        transform = None
        if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
            if args.nonlinear:
                element = elements.Quad4NonlinearShell(transform, con)
            else:
                element = elements.Quad4Shell(transform, con)

        struct_mesh.setElement(i, element)

    # Create tacs assembler object from mesh loader
    tacs = struct_mesh.createTACS(6)

    start_time = time.time()

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray()
    force_array[2::6] += 100.0  # uniform load in z direction
    tacs.applyBCs(forces)

    # Set up and solve the analysis problem
    res = tacs.createVec()
    ans = tacs.createVec()
    u = tacs.createVec()
    # default is AMD ordering
    nz_start = time.time()

    # ['RCM', 'ND', 'AMD', 'Qorder']
    if args.ordering == 'AMD':
        ordering = TACS.TACS_AMD_ORDER
    elif args.ordering == 'RCM':
        ordering = TACS.RCM_ORDER
    elif args.ordering == 'ND':
        ordering = TACS.ND_ORDER
    elif args.ordering == 'Qorder':
        ordering = TACS.Q_ORDER

    mat = tacs.createSchurMat(ordering)
    # ILUk = 3
    # ILUk = 8
    ILUk = 1000
    pc = TACS.Pc(mat, lev_fill=ILUk) # for ILU(3), default is full LU or ILU(1000) if not set
    # subspace = 100
    subspace = 200
    restarts = 0
    rtol = 1e-10
    atol = 1e-6
    gmres = TACS.KSM(mat, pc, subspace, restarts)
    tacs_comm = MPI.COMM_WORLD
    gmres.setMonitor(tacs_comm)
    gmres.setTolerances(rtol, atol)
    nz_time = time.time() - nz_start

    # Assemble the Jacobian and factor
    start_jac = time.time()
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)
    jac_time = time.time() - start_jac

    # assemble residual time
    start_res = time.time()
    tacs.assembleRes(res)
    res_time = time.time() - start_res

    solve_start = time.time()
    pc.factor()

    # Solve the linear system
    gmres.solve(forces, ans)
    tacs.setVariables(ans)
    solve_time = time.time() - solve_start

    dof = force_array.shape[0]

    tot_time = time.time() - start_time

    if tacs_comm.rank == 0:
        print("nprocs, nxe,  dof, nz_time, jac_time,  res_time,  solve_time,  tot_time")
        print(f"{tacs_comm.size}, {args.nxe}, {dof}, {nz_time:.4e}, {jac_time:.4e}, {res_time:.4e} {solve_time:.4e}, {tot_time:.4e}\n")

    outFile = "out/plate.f5" if args.case == "plate" else "out/ucrm.f5"

    # Output for visualization
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
    f5.writeToFile(outFile)
    return

def run_nonlinear_static(args):
    # Load structural mesh from BDF file
    COMM = MPI.COMM_WORLD
    PWD = os.path.join(os.path.dirname(__file__), "out")

    if args.case == "plate":
        BDF_FILE = "plate.bdf"
    else: # uCRM
        BDF_FILE = "../uCRM/CRM_box_2nd.bdf"
    # BDF_FILE = "../uCRM/CRM_box_2nd.bdf"
    # BDF_FILE = "uCRM-135_wingbox_fine.bdf"


    structOptions = {
        "printtiming": True,
    }
    FEAAssembler = pyTACS(BDF_FILE, options=structOptions, comm=COMM)

    # material properties, shell thickness, etc.
    # ------------------------------------------
    RHO = 2500 # density kg/m^3
    E = 70e9 # elastic modulus, Pa
    NU = 0.3 # poisson's ratio
    YIELD_STRESS = 350e6 # yield stress, Pa
    THICKNESS = 0.02 # meters

    def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
        matProps = constitutive.MaterialProperties(rho=RHO, E=E, nu=NU, ys=YIELD_STRESS)
        con = constitutive.IsoShellConstitutive(
            matProps, t=THICKNESS, tNum=dvNum, tlb=1e-2 * THICKNESS, tub=1e2 * THICKNESS
        )
        transform = None
        element = elements.Quad4NonlinearShell(transform, con)
        tScale = [10.0]
        return element, tScale

    FEAAssembler.initialize(elemCallBack)

    probOptions = {
        "printTiming": True,
        "printLevel": 1,
    }
    newtonOptions = {"useEW": True, "MaxLinIters": 10}
    continuationOptions = {
        "CoarseRelTol": 1e-3,
        "InitialStep": 0.1,
        "UsePredictor": False, # True orig
        "NumPredictorStates": 7,
    }
    forceProblem = FEAAssembler.createStaticProblem("NLVertLoads", options=probOptions)
    try:
        forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)
        forceProblem.nonlinearSolver.setOptions(continuationOptions)
    except AttributeError:
        pass

    # ==============================================================================
    # Add linear vertical loads
    # ==============================================================================

    bdfInfo = FEAAssembler.getBDFInfo()
    nodeIDs = list(bdfInfo.node_ids)

    load_mag = 3.0
    max_load_factor = 23.0

    forceProblem.addLoadToNodes(
        nodeIDs, [0, 0, load_mag * max_load_factor, 0, 0, 0], nastranOrdering=True
    )

    # ==============================================================================
    # Run analysis with load scales in 10% increments from 10% to 100%
    # ==============================================================================

    forceFactor = np.arange(0.10, 1.01, 0.10)
    ForceVec = np.copy(forceProblem.F_array)

    print(f"{forceFactor=}")

    for scale in forceFactor:
        Fext = (scale - 1.0) * ForceVec
        forceProblem.solve(Fext=Fext)
        forceProblem.writeSolution(outputDir=PWD) 

    return

if __name__ == "__main__":
    args = parse_args()

    if args.case == "plate":
        make_plate_bdf(args)
        
    if args.nonlinear:
        run_nonlinear_static(args)
    else:
        run_linear_static(args)
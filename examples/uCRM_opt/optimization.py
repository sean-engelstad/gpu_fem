import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os

# setup GPU solver and other init dvs, etc.
rhoKS = 10.0 # 100.0
SF = 1.5 # safety factor
load_mag = 30.0
solver = gpusolver.TACSGPUSolver(rhoKS, SF, load_mag)
ndvs = solver.get_num_dvs()
# x0 = np.array([5e-2]*ndvs)
# need to start infeasible otherwise SNOPT will exit early once tries to go infeasible
x0 = np.array([6e-3]*ndvs)
solver.set_design_variables(x0)

def get_functions(xdict):
    # update design
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])
    solver.set_design_variables(xarr)

    # solve and get functions
    solver.solve()
    mass = solver.evalFunction("mass")
    ksfail = solver.evalFunction("ksfailure")
    funcs = {
        'mass' : mass,
        'ksfailure' : ksfail,
    }

    comp_names = [f"comp{icomp}" for icomp in range(ndvs)]
    with open("out/design_variables.txt", "w") as f:
        f.write(f"{mass=:.4e} {ksfail=:.4e}\n")
        for name, value in zip(comp_names, xarr):
            f.write(f"{name}\t{value:.16e}\n")

    return funcs, False

def get_function_grad(xdict, funcs):
    # update design, this shouldn't really come with a new forward solve, just adj solve
    # but if optimizer calls grads twice in a row or something, this won't work right (needs companion get_functions call every time)
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])
    solver.set_design_variables(xarr)

    mass_grad = solver.evalFunctionSens("mass")
    ksfail_grad = solver.evalFunctionSens("ksfailure")
    sens = {
        'mass' : {'vars' : mass_grad},
        'ksfailure' : {'vars' : ksfail_grad},
    }
    return sens, False

opt_problem = Optimization("tuning-fork", get_functions)
opt_problem.addVarGroup(
    "vars",
    ndvs,
    lower=np.array([1e-4]*ndvs), # TODO : change min of non-struct-masses?
    upper=np.array([1e-1]*ndvs),
    value=x0,
    scale=np.array([1e2]*ndvs),
)
opt_problem.addObj('mass', scale=1e-2)
opt_problem.addCon("ksfailure", scale=1.0, upper=1.0)

verify_level = -1
# verify_level = 0
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-4,
        "Verify level": verify_level, #-1,
        "Major iterations limit": 1000, #1000, # 1000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        # "Major step limit": 5e-2,
        "Nonderivative linesearch": None,
        "Linesearch tolerance": 0.9,
        #"Difference interval": 1e-6,
        # "Function precision": 1e-6, #results in btw 1e-4, 1e-6 step sizes
        "New superbasics limit": 2000,
        "Penalty parameter": 1,
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("out/SNOPT_print.out"),
        "Summary file": os.path.join("out/SNOPT_summary.out"),
    }
)

hot_start = False
sol = snoptimizer(
    opt_problem,
    sens=get_function_grad,
    storeHistory="out/myhist.hst",
    hotStart="out/myhist.hst" if hot_start else None,
)

solver.writeSolution("out/uCRM_opt.vtk")
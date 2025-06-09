import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os

# setup GPU solver and other init dvs, etc.
rhoKS = 100.0 # 100.0
SF = 1.5 # safety factor
load_mag = 100.0
# load_mag = 30.0
solver = gpusolver.TACSGPUSolver(rhoKS, SF, load_mag)
ndvs = solver.get_num_dvs()
# x0 = np.array([5e-2]*ndvs)
# need to start infeasible otherwise SNOPT will exit early once tries to go infeasible
# x0 = np.array([2e-2]*ndvs)
# x0 = np.array([6e-2] * ndvs)
# solver.set_design_variables(x0) # don't call here messes up

# test starting from an intermediate design point, maybe derivs messed up here
x0 = np.array([0.00291918, 0.00341138, 0.00434817, 0.0055349 , 0.00492069,
       0.00524887, 0.0057326 , 0.00619608, 0.00690362, 0.0075743 ,
       0.00845037, 0.00903539, 0.00906191, 0.00888147, 0.00972974,
       0.01074326, 0.01180957, 0.01281484, 0.01340411, 0.01387793,
       0.01477443, 0.01565993, 0.01657483, 0.0176262 , 0.01853827,
       0.01865628, 0.01756092, 0.01645125, 0.01622856, 0.01671115,
       0.01763206, 0.01811106, 0.01860782, 0.01876995, 0.0187903 ,
       0.01904377, 0.01986556, 0.01908644, 0.02051867, 0.02007202,
       0.02022946, 0.02021478, 0.02037288, 0.02057162, 0.02092626,
       0.02138538, 0.02161812, 0.02200876, 0.0137045 , 0.01569071,
       0.01817788, 0.02067788, 0.02317788, 0.02067788, 0.01822504,
       0.01762468, 0.01682185, 0.01597609, 0.01502051, 0.01428095,
       0.01401729, 0.01382572, 0.01374723, 0.01408699, 0.0144181 ,
       0.01568006, 0.01527939, 0.0156977 , 0.01675991, 0.01780379,
       0.01763288, 0.0167282 , 0.01626657, 0.01617011, 0.01627341,
       0.01713874, 0.01748946, 0.01770196, 0.01782598, 0.01794027,
       0.01830366, 0.01880088, 0.01919395, 0.01957554, 0.020135  ,
       0.0204054 , 0.02076282, 0.02108389, 0.02181225, 0.0229615 ,
       0.02367031, 0.0226634 , 0.02070315, 0.00032471, 0.00051868,
       0.00057966, 0.00153293, 0.00175666, 0.00316503, 0.00146785,
       0.00072808, 0.00067823, 0.00135375, 0.00140043, 0.00184251,
       0.00280242, 0.0026487 , 0.00178642, 0.00149233, 0.00152627,
       0.00197745, 0.0051057 , 0.00146151, 0.00136627, 0.00357671,
       0.00072477, 0.00075372, 0.00085431, 0.00125799, 0.00119976,
       0.00142896, 0.00151822, 0.00129704, 0.00158863, 0.0016868 ,
       0.00337935, 0.00421057, 0.00523396, 0.00604538, 0.00678049,
       0.00767953, 0.00893991, 0.01028716, 0.01168713, 0.01308526,
       0.01491075, 0.016906  , 0.01956973, 0.0232524 , 0.02451479,
       0.00687711, 0.00686471, 0.00729114, 0.00713843, 0.00650916,
       0.00620359, 0.0055589 , 0.0052441 , 0.00466456, 0.00439243,
       0.00386696, 0.00362278, 0.00290563, 0.00271699, 0.00096207,
       0.0016624 , 0.00211453, 0.00103848, 0.00143028, 0.0012106 ,
       0.00178296, 0.00251963, 0.00179836, 0.00282372, 0.00347658,
       0.00372416, 0.00386724, 0.00487153, 0.00289132, 0.0028514 ,
       0.00436666, 0.0048895 , 0.00381568, 0.00349008, 0.00403788,
       0.00499414, 0.00559381, 0.0074987 , 0.00508825, 0.00562792,
       0.00610133, 0.00637851, 0.00646332, 0.00694253, 0.00531247,
       0.00750651, 0.00531745, 0.00654703, 0.00712649, 0.00780417,
       0.0062371 , 0.00727316, 0.00751103, 0.00824793, 0.00812868,
       0.01043339, 0.00942506, 0.01036893, 0.00850215, 0.01092857,
       0.00850739, 0.0085753 , 0.00787042, 0.00857758, 0.00928518,
       0.00832409, 0.00706293, 0.00878781, 0.00880098, 0.01120954,
       0.00875447, 0.00727383, 0.00888248, 0.01001023, 0.00806064,
       0.00773502, 0.00958666, 0.01185875, 0.01027184, 0.00999014,
       0.01200467, 0.01450467, 0.01353807, 0.0119748 , 0.01064267,
       0.00909994, 0.00874204, 0.00996832, 0.01146742, 0.01392989,
       0.0115262 , 0.01391036, 0.01144944, 0.01362802, 0.01116728,
       0.01341727, 0.00696539, 0.0001    , 0.0116951 , 0.0131951 ,
       0.0116951 , 0.01020848])

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

    # before not including this would result in wrong state vars
    # now it only does the solve again if design changed with this call
    solver.solve() # temp debug just solve again here

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
    upper=np.array([1e0]*ndvs),
    value=x0,
    scale=np.array([1e2]*ndvs),
)
opt_problem.addObj('mass', scale=1e-3)
opt_problem.addCon("ksfailure", scale=1.0, upper=1.0)

# adjacency constraints (linear so reduces size of opt problem)
# ---------------------
def adj_vec(i, j):
    vec = np.zeros((1,ndvs), dtype=np.double)
    vec[0,i] = 1.0
    vec[0,j] = -1.0
    return vec

# LE spars (48 dvs, 0-47 entries inclusive)
for icomp in range(0, 47): # but only 47 constraints (one less)
    opt_problem.addCon(
        f"LEspar-{icomp}",
        lower=-2.5e-3,
        upper=2.5e-3,
        scale=1e2,
        linear=True,
        wrt=['vars'],
        jac={'vars': adj_vec(icomp, icomp+1)}
    )

# TE spars (45 DVs, 48-92, but 44 constr)
for icomp in range(48, 92):
    opt_problem.addCon(
        f"TEspar-{icomp-48}",
        lower=-2.5e-3,
        upper=2.5e-3,
        scale=1e2,
        linear=True,
        wrt=['vars'],
        jac={'vars': adj_vec(icomp, icomp+1)}
    )

# Uskin (48 DVs in odd positions of 140-235 inclusive, but one less bc constr)
for icomp in range(141,235,2):
    iconstr = (icomp-141) // 2
    opt_problem.addCon(
        f"Uskin-{iconstr}",
        lower=-2.5e-3,
        upper=2.5e-3,
        scale=1e2,
        linear=True,
        wrt=['vars'],
        jac={'vars': adj_vec(icomp, icomp+1)}
    )

# Lskin (48 DVs in even positions of 140-235 inclusive, but one less bc constr)
for icomp in range(140,235,2):
    iconstr = (icomp-140) // 2
    opt_problem.addCon(
        f"Lskin-{iconstr}",
        lower=-2.5e-3,
        upper=2.5e-3,
        scale=1e2,
        linear=True,
        wrt=['vars'],
        jac={'vars': adj_vec(icomp, icomp+1)}
    )

# side of body (4 DVs, but 3 constr here)
for icomp in range(238, 241):
    iconstr = (icomp - 238)
    opt_problem.addCon(
        f"SOB-{iconstr}",
        lower=-1.5e-3,
        upper=1.5e-3,
        scale=1e2,
        linear=True,
        wrt=['vars'],
        jac={'vars': adj_vec(icomp, icomp+1)}
    )

# verify_level = -1
verify_level = 0
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-5,
        "Major optimality tolerance": 1e-3,
        "Verify level": verify_level, #-1,
        "Major iterations limit": 1000, #1000, # 1000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        "Major step limit": 5e-2, # these causes l to turn on a lot
        "Nonderivative linesearch": True,
        "Linesearch tolerance": 0.9,
        # "Difference interval": 1e-6,
        # "Function precision": 1e-10, #results in btw 1e-4, 1e-6 step sizes
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0, #1.0E2
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

print(f"{sol.xStar=}")

solver.solve()
solver.writeSolution("out/uCRM_opt.vtk")
solver.free()
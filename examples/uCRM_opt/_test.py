import gpusolver
import numpy as np

rhoKS = 100.0
# SF = 1.5
SF = 1.0
load_mag = 100.0
solver = gpusolver.TACSGPUSolver(rhoKS, SF, load_mag)
ndvs = solver.get_num_dvs()

x = np.array([2e-2]*ndvs)
solver.set_design_variables(x)
solver.solve()
mass = solver.evalFunction("mass")
ksfail = solver.evalFunction("ksfailure")
print(f"{mass=:.4e} {ksfail=:.4e}")

mass_grad = solver.evalFunctionSens("mass")
print(f"{mass_grad[:5]=}")
ksfail_grad = solver.evalFunctionSens("ksfailure")
print(f"{ksfail_grad[:5]=}")

solver.writeSolution("out/uCRM.vtk")
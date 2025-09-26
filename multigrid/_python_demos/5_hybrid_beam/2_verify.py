# now let's test this out and visualize it
import numpy as np
from src import EBAssembler
from src import TSAssembler
from src import HybridAssembler

# verify the hybrid assembler solution against exact Timoshenko beam theory solution for constant distributed load

# SR = 1.0
# SR = 10.0
# SR = 100.0
# SR = 1000.0
SR = 10000.0

# setup
# -------------------------------

E = 2e7; b = 1.0; L = 1.0; rho = 1
# qmag = 2e-2
qmag = 1e4
ys = 4e5

# scale by slenderness
thick = 1.0 / SR
qmag *= thick**3

# scaling inputs
# if rho_KS was too high like 500, then the constraint was too nonlinear or close to max and optimization failed
rho_KS = 50.0 # rho = 50 for 100 elements, used 500 later
# nxe = num_elements = int(3e2) #100, 300, 1e3
nxe = num_elements = int(2e3)
# nxe = 3

# num DVs
nxh = 100 if nxe > 5 else nxe 
hvec = np.array([thick] * nxh)

# 3) hybrid beam analysis
# ------------------------

hyb_beam = HybridAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
hyb_beam.solve_forward(hvec)
hyb_beam.plot_disp()

# numerical solution for center deflection
w_vec = hyb_beam.u[0::3]
nnodes = w_vec.shape[0]
center = (0 + nnodes - 1) // 2
pred_disp = w_vec[center]

# exact solution for center deflection
I = b * thick**3 / 12.0
A = b * thick
Ks = 5.0 / 6.0
nu = 0.3
G = E / 2.0 / (1 + nu)
x = L / 2.0 # center of beam
exact_disp = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4) + qmag * L**2 / 24.0 / G / Ks * (x / L - x**2 / L**2)
term1 = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4)
term2 = qmag * L**2 / 24.0 / G / Ks * (x / L - x**2 / L**2)
print(f"{term1=:.2e} {term2=:.2e}")

# EB beam..
eb_beam = EBAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
eb_beam.solve_forward(hvec)
# eb_beam.plot_disp()

print(f"{qmag=}")
for inode in range(10):
    nodal_loads = eb_beam.force[2*inode:(2*inode+2)]
    print(f"{inode=} : {nodal_loads=}")

# compute the total force on the beam compared to qmag
qmag_tot = np.sum(eb_beam.force)
qmag_exact = qmag * L
print(f"{qmag_tot=:.2e} {qmag_exact=:.2e}")

# numerical solution for center deflection
w_vec2 = eb_beam.u[0::2]
pred_disp_eb = w_vec2[center]
print(f"{pred_disp_eb=:.2e}")

# for very thin plate, equivalent is:
eb_exact = 5.0 * qmag * L**4 / 384 / E / I
print(f"{eb_exact=:.2e}") # it's not matching this somehow, need to fix this..

rel_err = abs((pred_disp - exact_disp) / exact_disp)
print(f"{SR=:.2e} : {pred_disp=:.2e}, {exact_disp=:.2e} with {rel_err=:.2e}")
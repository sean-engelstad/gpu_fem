# now let's test this out and visualize it
import numpy as np
from src import EBAssembler #, plot_hermite_cubic
from src import TimoshenkoAssembler
from src import HybridAssembler
from src import ChebyshevTSAssembler
from src import HybridChebyshevAssembler

# SR = 0.1
# SR = 1.0
# SR = 10.0
SR = 100.0
# SR = 1000.0
# SR = 1e4

# 1) Euler-Bernoulli Beam
# -------------------------

E = 2e7; b = 4e-3; L = 1.0; rho = 1
qmag = 2e-2
ys = 4e5

# scale by slenderness
thick = 1.0 / SR
qmag *= thick**3

# scaling inputs
# if rho_KS was too high like 500, then the constraint was too nonlinear or close to max and optimization failed
rho_KS = 50.0 # rho = 50 for 100 elements, used 500 later
# nxe = num_elements = int(3e2) #100, 300, 1e3
# nxe = num_elements = int(1e3)
# nxe = num_elements = int(3e2)
nxe = num_elements = int(100)
# nxe = num_elements = int(30)

# num DVs
# nxh = 100 
nxh = nxe
hvec = np.array([thick] * nxh)

eb_beam = EBAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
eb_beam.solve_forward(hvec)
# eb_beam.plot_disp()


# 2) TACS timoshenko beam
# ------------------------

ts_beam = TimoshenkoAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, use_nondim=True)
ts_beam.solve_forward(hvec)
# ts_beam.plot_disp()


# 3) hybrid beam
# ------------------------

hyb_beam = HybridAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
hyb_beam.solve_forward(hvec)
# hyb_beam.plot_disp()

# 4) chebyshev beam
# ------------------
# order = 1
order = 2
cfe_beam = ChebyshevTSAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, order=order)
cfe_beam.solve_forward(hvec)
# cfe_beam.plot_disp()

# 4) hybrid chebyshev beam (not hermite formulation, only 2 dof per node not 3)
# ------------------
# order = 1
order = 2
# order = 3
# order = 4

# this element doesn't seem to work as well, LOL
# may need to also include dual hermite-chevyshev basis..

_nxe = nxe // order
hyb_cfe_beam = HybridChebyshevAssembler(_nxe, _nxe, E, b, L, rho, qmag, ys, rho_KS, dense=True, order=order)
hyb_cfe_beam.solve_forward(hvec[:_nxe])
# hyb_cfe_beam.plot_disp()

# plot all of them..
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(10, 6))

names = ["EB-beam", "TS-Beam", "HYB-beam", "CFE-beam"]
for i,beam in enumerate([eb_beam, ts_beam, hyb_beam, cfe_beam]):
    stride = 2 if i != 2 else 3
    xvec = beam.xvec
    disp = beam.u[0::stride]
    ax[i].plot(xvec, disp, label=names[i])
    ax[i].legend()
plt.show()
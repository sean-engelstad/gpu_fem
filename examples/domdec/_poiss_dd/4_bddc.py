import numpy as np

from src import PoissonPlateElement, BDDC_PlateAssembler
from src import right_pcg_matfree

nxe = 64
nxs = 16

ELEMENT = PoissonPlateElement()

assembler = BDDC_PlateAssembler(
    ELEMENT=ELEMENT,
    nxe=nxe,
    nxs=nxs,
    nys=nxs,
    clamped=False,
    coarse_mode="assembled",
    scale_R_GE=True,
    load_fcn=lambda x, y: np.sin(5 * np.pi * x) * np.sin(np.pi * (y**2 + x**2)),
)

assembler.assemble_all()

edge_rhs = assembler.get_interface_rhs()
print(f"{np.linalg.norm(edge_rhs)=:.4e}")

norm_hist = []
edge_soln, nsteps = right_pcg_matfree(
    assembler,
    b=edge_rhs,
    rtol=1e-6,
    atol=1e-20,
    max_iter=200,
    print_freq=3,
    norm_hist=norm_hist,
)

global_soln = assembler.get_global_solution(edge_soln)
direct_soln = assembler.direct_solve()

err_nrm = np.linalg.norm(global_soln - direct_soln)
print(f"{err_nrm=:.4e}")

assembler.u = global_soln.copy()
assembler.plot_disp()
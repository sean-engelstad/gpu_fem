# direct solve of poisson problem (global direct solve)

import numpy as np, scipy.sparse as sp
import sys

from src import PoissonPlateElement, SubdomainPlateAssembler

ELEMENT = PoissonPlateElement()
assembler = SubdomainPlateAssembler(
    ELEMENT=ELEMENT,
    # nxe=8,
    nxe=50,
    nxs=1, nys=1, # just one subdomain (so really just global)
    clamped=False,
    load_fcn=lambda x, y : np.sin(5 * np.pi * x) * np.sin(np.pi * (y**2 + x**2))
)

assembler.direct_solve()
assembler.plot_disp()
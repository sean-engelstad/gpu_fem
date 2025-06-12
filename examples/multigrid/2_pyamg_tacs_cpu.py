# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os
import pyamg
from _utils import get_tacs_matrix

if __name__=="__main__":
    import scipy.sparse.linalg as spla
    import time

    # get TACS matrix and convert to CSR since pyAMG only supports CSR
    # (this may mean it is slower or less accurate than a BSR version)
    # -----------------------------------
    tacs_bsr_mat, rhs = get_tacs_matrix()
    tacs_csr_mat = tacs_bsr_mat.tocsr()

    # solve with pyAMG
    # ----------------

    ml = pyamg.smoothed_aggregation_solver(tacs_csr_mat)
    M = ml.aspreconditioner()

    def print_residual(rk):
        print(f"residual: {rk:.2e}")

    # it converges very slowly if you print..
    callback = print_residual
    # callback = None

        
    # pyAMG only accepts CSR matrix not BSR (so that's why it's likely so slow, missing important intra-block structure)
    start_time = time.time()
    x, info = spla.gmres(tacs_csr_mat, rhs, M=M, rtol=1e-4, atol=1e0, maxiter=100, callback=callback)
    amg_solve_time = time.time() - start_time
    print(f"{amg_solve_time=:.3e}")
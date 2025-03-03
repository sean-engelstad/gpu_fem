import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import seaborn as sns

from _gmres_util import *

if __name__=="__main__":
    n = 100
    d = 5
    A = np.random.rand(n,n) + d * np.eye(n)
    A /= d
    b = np.random.rand(n)

    x = gmres(A, b, restart=n)
    xtruth = np.linalg.solve(A, b)
    err = np.linalg.norm(x - xtruth)
    print(f"{err=}")

    # compare to scipy's gmres solver
    x_scipy, info = spla.gmres(A, b, rtol=1e-8, restart=50)
    print(f"SciPy GMRES Residual: {np.linalg.norm(b - A @ x_scipy)}")

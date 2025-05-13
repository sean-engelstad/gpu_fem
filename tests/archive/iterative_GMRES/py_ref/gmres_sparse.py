import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import spilu
from scipy.sparse import csr_matrix

from _gmres_util import *

if __name__=="__main__":
    # 10x10 linear system
    n = 10
    nnz = 60 # num nonzeros
    np.random.seed(1234)

    # CSR matrix format A matrix

    # data = np.random.rand(nnz)  # 20 nonzero values
    # rows = np.random.randint(0, n, size=nnz)
    # cols = np.random.randint(0, n, size=nnz)
    # # print(f"{rows=} {cols=}")
    # eta = 2.0 # diagonal regularization
    # for i in range(nnz):
    #     if rows[i] == cols[i]:
    #         print(f"add diag to {rows[i]=} {cols[i]=}")
    #         data[i] += eta
    # # print(f"{data=}")
    # A = csr_matrix((data, (rows, cols)), shape=(n, n))
    # # A += eta * np.eye(n,n)
    # # print(f'{A=}')
    # b = np.random.rand(n)

    A = csr_matrix(np.random.rand(10, 10) + 0.1 * np.eye(10))  # Ensuring nonzero diagonal
    b = np.random.rand(n)

    # make the ILU(k) preconditioner here
    M = spilu(A)
    # M = None

    x = gmres(A, b, restart=n, M=M)
    xtruth = spla.spsolve(A, b)
    err = np.linalg.norm(x - xtruth)
    print(f"{err=}")

    # compare to scipy's gmres solver
    # x_scipy, info = spla.gmres(A, b, rtol=1e-8, restart=50)
    # print(f"SciPy GMRES Residual: {np.linalg.norm(b - A @ x_scipy)}")

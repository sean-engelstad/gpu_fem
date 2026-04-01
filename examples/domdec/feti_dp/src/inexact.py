# from ._assembler import Subdomain2DAssembler
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class IdentityPreconditioner:
    def __init__(self, A, **kwargs):
        self.shape = A.shape

    def apply(self, r: np.ndarray) -> np.ndarray:
        return np.array(r, dtype=np.double, copy=True)

class ASWPreconditioner:
    pass # TBD code this one up

class ILU0Preconditioner:
    """
    SciPy's spilu is an ILU-type factorization. With
      fill_factor=1.0, drop_tol=0.0, permc_spec='NATURAL'
    this is the closest practical ILU(0)-style setup in SciPy.
    """
    def __init__(
        self,
        A,
        drop_tol: float = 0.0,
        fill_factor: float = 1.0,
        permc_spec: str = "NATURAL",
        diag_pivot_thresh: float = 0.0,
        **kwargs,
    ):
        A = A.tocsc()
        self.ilu = spla.spilu(
            A,
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            permc_spec=permc_spec,
            diag_pivot_thresh=diag_pivot_thresh,
        )

    def apply(self, r: np.ndarray) -> np.ndarray:
        return self.ilu.solve(r)

class ILUTPreconditioner:
    """
    Incomplete LU with threshold dropping / fill control via SciPy's spilu.

    This is the general ILUT-style wrapper:
      - smaller drop_tol  -> keep more entries
      - larger fill_factor -> allow more fill

    Notes
    -----
    * SciPy's `spilu` is a SuperLU-based ILU factorization, so "ILUT" here
      means the practical threshold/fill-controlled ILU exposed by `spilu`.
    * Unlike ILU(0), this generally allows extra fill beyond the original
      sparsity pattern.
    """
    def __init__(
        self,
        A,
        drop_tol: float = 1.0e-4,
        fill_factor: float = 10.0,
        permc_spec: str = "COLAMD",
        diag_pivot_thresh: float = 0.0,
        drop_rule: str = "basic,area",
        panel_size: int = 10,
        relax: int = 1,
        **kwargs,
    ):
        A = A.tocsc()
        self.ilu = spla.spilu(
            A,
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            permc_spec=permc_spec,
            diag_pivot_thresh=diag_pivot_thresh,
            drop_rule=drop_rule,
            panel_size=panel_size,
            relax=relax,
        )

    def apply(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.double)
        return self.ilu.solve(r)

class ExactSparseSolver:
    """
    Exact sparse direct solve wrapper with a uniform .solve(rhs) API.
    """
    def __init__(self, A, **kwargs):
        self.A = A.tocsc()
        self._solve = spla.factorized(self.A)

    def solve(self, rhs: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.double)
        return np.asarray(self._solve(rhs), dtype=np.double)
    
    def apply(self, r):
        return self.solve(r, None)


class RichardsonSolver:
    """
    Generic stationary iterative solver:
        x_{k+1} = x_k + omega * M^{-1}(b - A x_k)

    where M^{-1} is supplied by any preconditioner class with:
        pc = PrecondClass(A, **kwargs)
        z  = pc.apply(r)
    """
    def __init__(
        self,
        A,
        precond_cls=IdentityPreconditioner,
        precond_kwargs=None,
        nsteps: int = 3,
        omega: float = 1.0,
        **kwargs,
    ):
        self.A = A.tocsc()
        self.nsteps = int(nsteps)
        self.omega = float(omega)
        self.precond = precond_cls(self.A, **(precond_kwargs or {}))

    def solve(self, rhs: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.double)
        x = np.zeros_like(rhs) if x0 is None else np.array(x0, dtype=np.double, copy=True)

        for _ in range(self.nsteps):
            r = rhs - self.A.dot(x)
            z = self.precond.apply(r)
            x += self.omega * z

        return x
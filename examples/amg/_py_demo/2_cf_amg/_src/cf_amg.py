import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../1_sa_amg/_src")
from _smoothers import gauss_seidel_csr, gauss_seidel_csr_transpose

def build_initial_FC_matrix(A_csr:sp.csr_matrix, C_nodes, F_nodes):
    # builds initial fine-coarse prolongation matrix
    # uses nofill sparsity and all connections in matrix to form the stencil
    return A_csr[F_nodes, :][:, C_nodes]

def direct_csr_interpolation(A_csr:sp.csr_matrix, C_nodes, F_nodes):
    """from multigrid book, direct interpolation using positive and negative connections
    better explained in matrix form from this book, https://arxiv.org/pdf/1611.01917
    """
    
    A_FC = build_initial_FC_matrix(A_csr, C_nodes, F_nodes)

    # computes diag(A_{FC} * 1)^{-1}
    diag_FF = A_FC.dot(np.ones(A_FC.shape[1]))
    Dinv_FF = np.diag(1.0 / diag_FF) # matrix form

    # computes rescaled matrix
    W = Dinv_FF @ A_FC

    P = A_csr[:, C_nodes]
    # FC part is from direct interp
    P[F_nodes, :] = W[:, :]
    # CC part is injection
    P[C_nodes, :] = np.eye(A_FC.shape[-1])
    return P

def standard_csr_interpolation(A_csr:sp.csr_matrix, C_nodes, F_nodes):
    """from multigrid book,
    better explained in matrix form from this book, https://arxiv.org/pdf/1611.01917"""
    
    
    A_FC = build_initial_FC_matrix(A_csr, C_nodes, F_nodes)
    A_FF = A_csr[F_nodes, :][:, F_nodes]

    # first get direct interpolation
    diag_FF = A_FC.dot(np.ones(A_FC.shape[1]))
    Dinv_FF = np.diag(1.0 / diag_FF) # matrix form
    W1 = Dinv_FF @ A_FC

    # then get standard interpolation
    W = W1 - Dinv_FF @ A_FF @ W1
    # then rescale for constant RBMs of CSR case
    diag2_FF = W.dot(np.ones(W.shape[-1]))
    Dinv2_FF = np.diag(1.0 / diag2_FF)
    W2 = Dinv2_FF @ W

    P = A_csr[:, C_nodes]
    # FC part is from direct interp
    P[F_nodes, :] = W2[:, :]
    # CC part is injection
    P[C_nodes, :] = np.eye(A_FC.shape[-1])
    return P


class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x



class ClassicalAMGSolver:
    """general multilevel AMG solver..."""
    def __init__(self, A_free:sp.csr_matrix, A:sp.csr_matrix, threshold:float=0.25, 
                 omega:float=0.7, pre_smooth=1, post_smooth=1, level:int=0, 
                 coarsening_fcn=None, interpolation_fcn=None):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        threshold: float for coarsening threshold
        """
        assert sp.isspmatrix_csr(A_free)
        assert sp.isspmatrix_csr(A)

        self.A_free = A_free
        self.A = A
        self.coarsening_fcn = coarsening_fcn
        self.interpolation_fcn = interpolation_fcn

        # do coarsening and get interpolation
        C_mask, F_mask = coarsening_fcn(A_free, threshold)
        print(f"{np.sum(C_mask)=} {np.sum(F_mask)=}")
        self.P = interpolation_fcn(A_free, C_mask, F_mask)
        self.R = self.P.T

        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.level = level

        print(f"{self.P.shape=}")

        # Galerkin coarse operator
        self.Ac = self.R @ (A @ self.P)
        # on GPU would not do extra allocation for this
        self.Ac_free = self.R @ (A_free @ self.P)

        self.Ac = self.Ac.tocsr()
        self.Ac_free = self.Ac_free.tocsr()
        # print(f"{type(self.Ac)=}")

        # plt.spy(self.Ac_free)
        # plt.show()
        # plt.imshow(self.Ac_free.toarray())
        # plt.show()

        # compute two-grid operator complexity
        self.fine_nnz = self.A.nnz
        self.coarse_nnz = self.Ac.nnz
        self.coarse_solver = None 
        if level == 0:
            print("level 0 is AMG solver..")

        # check fillin of coarse grid solver..
        coarse_nnodes = self.Ac.shape[0]
        max_coarse_nnz = coarse_nnodes**2
        if self.coarse_nnz >= 0.4 * max_coarse_nnz or coarse_nnodes <= 100:
            # then do direct solver
            print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            print(f"level {level+1} building AMG solver")
            self.coarse_solver = ClassicalAMGSolver(self.Ac_free, self.Ac, threshold, omega, pre_smooth, post_smooth, level+1, 
                                                    coarsening_fcn=coarsening_fcn, interpolation_fcn=interpolation_fcn)

        if level == 0:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=}")
            print(f"\tnum nodes per level = [{self.num_nodes_list}]")

    @property
    def total_nnz(self) -> int:
        # get total nnz across all levels
        if isinstance(self.coarse_solver, ClassicalAMGSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else: # direct solver
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz     

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, ClassicalAMGSolver):
            return self.coarse_solver.num_levels + 1
        else: # direct solver
            return 2
        
    @property
    def num_nodes_list(self) -> str:
        if isinstance(self.coarse_solver, ClassicalAMGSolver):
            return str(self.A.shape[0]) + "," + self.coarse_solver.num_nodes_list
        else: # direct solver
            return f"{self.A.shape[0]},{self.Ac.shape[0]}"

    def solve(self, rhs):
        """
        One AMG V-cycle
        """
        # initial guess
        x = np.zeros_like(rhs)

        # -------- pre-smoothing --------
        if self.pre_smooth > 0:
            x = gauss_seidel_csr(
                self.A, rhs,
                x0=x,
                num_iter=self.pre_smooth
            )

        # fine residual
        r = rhs - self.A @ x

        # restrict
        rc = self.R @ r

        # coarse solve
        ec = self.coarse_solver.solve(rc)

        # prolong correction
        x += self.P @ ec

        # -------- post-smoothing --------
        if self.post_smooth > 0:
            r = rhs - self.A @ x
            dx = gauss_seidel_csr_transpose(
                self.A, r,
                x0=np.zeros_like(rhs),
                num_iter=self.post_smooth
            )
            x += dx

        return x
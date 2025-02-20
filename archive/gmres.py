import numpy as np
import scipy.sparse.linalg as spla

def givens_rotation(a, b):
    """Compute the Givens rotation matrix parameters (c, s) such that:
        [ c  s ] [ a ]  =  [ r ]
        [-s  c ] [ b ]     [ 0 ]
    """
    if b == 0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1 / np.sqrt(1 + tau**2)
        c = s * tau
    else:
        tau = -b / a
        c = 1 / np.sqrt(1 + tau**2)
        s = c * tau
    return c, s

def apply_givens_rotation(H, cs, ss, k):
    """Apply the previously computed Givens rotations to H column k."""
    for i in range(k):
        temp = cs[i] * H[i, k] + ss[i] * H[i+1, k]
        H[i+1, k] = -ss[i] * H[i, k] + cs[i] * H[i+1, k]
        H[i, k] = temp

def gmres(A, b, x0=None, tol=1e-8, max_iter=1000, restart=50, M=None):
    """
    Implements the restarted GMRES algorithm with numerical stability.
    
    Parameters:
    A : callable or sparse matrix
        Function A(x) or sparse matrix representing the linear system.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess (default: zero vector).
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    restart : int, optional
        Restart parameter (default: 50).
    M : callable or sparse matrix, optional
        Preconditioner (default: None).
    
    Returns:
    x : ndarray
        Approximate solution.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda x: x
    else:
        M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))

    x = x0.copy()
    r = b - (A @ x)
    beta = np.linalg.norm(r)
    
    if beta < tol:
        return x

    for _ in range(max_iter // restart):
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        cs = np.zeros(restart)
        ss = np.zeros(restart)

        # Start Arnoldi process
        V[:, 0] = r / beta
        g[0] = beta

        for j in range(restart):
            w = A @ V[:, j]
            w = M_inv(w)  # Apply preconditioner if given

            # Modified Gram-Schmidt Orthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                break
            V[:, j+1] = w / H[j+1, j]

            # Apply previous Givens rotations
            apply_givens_rotation(H, cs, ss, j)

            # Compute new Givens rotation
            cs[j], ss[j] = givens_rotation(H[j, j], H[j+1, j])
            H[j, j] = cs[j] * H[j, j] + ss[j] * H[j+1, j]
            H[j+1, j] = 0.0

            # Apply rotation to g
            g[j+1] = -ss[j] * g[j]
            g[j] = cs[j] * g[j]

            # Check convergence
            if abs(g[j+1]) < tol:
                break

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])

        # Update solution
        x += V[:, :j+1] @ y

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break

    return x

# custom krylov methods with mat-vec operations, needed for the subdomain iterative methods

import numpy as np


def right_pcg_matfree(
        assembler, # with precond_solve and mat_vec methods
        b:np.ndarray,
        x0=None,
        rtol=1e-8, atol=1e-7, max_iter=1000, 
        print_freq:int=10, norm_hist:list=[]):
    """
    Right-preconditioned Conjugate Gradient method (version from my scitech paper)

    Solves: A M^{-1} y = b, with x = M^{-1} y

    Parameters
    ----------
    assembler with precond_solve and mat_vec methods (mat-free needed for subdomain methods)
    b : ndarray
        Right-hand side
    x0 : ndarray, optional
        Initial guess for x
    tol : float
        Convergence tolerance on ||r||
    max_iter : int
        Maximum iterations

    Returns
    -------
    x : ndarray
        Approximate solution
    """

    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    A = lambda x : assembler.mat_vec(x)
    Minv = lambda x : assembler.precond_solve(x)

    # Initial residual
    r = b - A(x)
    norm_r0 = np.linalg.norm(r)

    if norm_r0 < atol:
        return x
    
    rz_old = None
    rz = None

    norm_hist += [1.0]

    print(f"PCG iter 0: ||r|| = {norm_r0:.3e}")

    for k in range(1, max_iter + 1):

        z = Minv(r)
        rz = np.dot(r, z)

        if k == 1:
            p = z.copy()
        else:
            beta = rz / rz_old
            p = z + beta * p_old

        p_old = p.copy()
        rz_old = rz * 1.0

        w = A(p)
        alpha = rz / np.dot(w, p)
        x += alpha * p
        r -= alpha * w
        norm_r = np.linalg.norm(r)

        rel_nrm = norm_r / norm_r0
        norm_hist += [float(rel_nrm)]

        if (k % print_freq == 0) or norm_r < (atol + rtol * norm_r0):
            print(f"PCG iter {k}: ||r|| = {norm_r:.3e}")

        if norm_r < (atol + rtol * norm_r0):
            break

    print(f"PCG finished at iter {k}, ||r|| = {norm_r:.3e}")
    return x, k+1
import numpy as np
import scipy.sparse.linalg as spla

def givens_rotation(a, b):
    """Compute the Givens rotation matrix parameters (c, s) such that:
        [ c  s ] [ a ]  =  [ r ]
        [-s  c ] [ b ]     [ 0 ]
    """
    if b == 0:
        return 1.0, 0.0
    # this part doesn't make sense to me..
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
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r = M_inv(b - (A @ x))
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

        print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            w = A @ V[:, j]
            w = M_inv(w)  # Apply preconditioner if given

            # check norms:
            if j == 0:
                A_norm = np.linalg.norm(A @ V[:, j])
                MA_norm = np.linalg.norm(M_inv(A @ V[:, j]))
                print(f"{A_norm=} {MA_norm=}")

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            # Reorthogonalization step (fixes numerical loss of orthogonality)
            # optional => can improve numerical stability
            # for i in range(j + 1):
            #     correction = np.dot(V[:, i], w)
            #     w -= correction * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            givens_option = 2

            if givens_option == 1:
                # there is a mistake in the givens rotations from this one
                # I found the correction in the way we compute givens_rotation(H1, H2)
                # correction on p. 162 of Saad for Sparse Iterative Methods

                # Apply previous Givens rotations
                apply_givens_rotation(H, cs, ss, j)

                # Compute new Givens rotation
                cs[j], ss[j] = givens_rotation(H[j, j], H[j+1, j])
                H[j, j] = cs[j] * H[j, j] + ss[j] * H[j+1, j]
                H[j+1, j] = 0.0

                # Apply rotation to g
                g[j+1] = -ss[j] * g[j]
                g[j] = cs[j] * g[j]

            else:

                # so far all checks out against my old HW3 iterative methods code
                # alternative code from above block:

                for i in range(j):
                    temp = H[i,j]
                    H[i,j] = cs[i] * H[i,j] + ss[i] * H[i+1,j]
                    H[i+1,j] = -ss[i] * temp + cs[i] * H[i+1,j]

                cs[j] = H[j,j] / np.sqrt(H[j,j]**2 + H[j+1,j]**2)
                ss[j] = cs[j] * H[j+1,j] / H[j,j]

                g_temp = g[j]
                g[j] *= cs[j]
                g[j+1] = -ss[j] * g_temp
                # print(f"{cs[j]=}")
                # print(f"{g[j]=}")

                H[j,j] = cs[j] * H[j,j] + ss[j] * H[j+1,j]
                H[j+1,j] = 0.0

            print(f"iteration {j} : {g[j+1]=}")

            # Check convergence
            if abs(g[j+1]) < tol:
                print(f"g break at iteration {j}")
                break

        # debugging check if hessenberg matrix is zero in correct spots
        # plt.imshow(H[:n+1,:n])
        # plt.show()

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])
        # y = np.linalg.solve(H[:j, :j], g[:j])
        # y = np.linalg.solve(H[:restart, :restart], g[:restart])
        # print(f"{H[-1,:j+1]=}")
        import matplotlib.pyplot as plt
        # print(f"{H[j,:j+1]=}")
        # plt.imshow(H[:j+1, :j+1])
        # print(f"{j+1=}")
        plt.imshow(H)
        plt.show()


        # Update solution
        x += V[:, :j+1] @ y

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
    
    print(f"final resid {beta=}")

    return x
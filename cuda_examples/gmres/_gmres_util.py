import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as spp

def get_laplace_system(N):

    n = np.int32(np.sqrt(N))
    nz = 5 * N - 4 * n

    rowp = np.zeros((N+1,), dtype=np.int32)
    cols = np.zeros((nz,), dtype=np.int32)
    vals = np.zeros((nz,))
    b = np.zeros((N,))

    # now define rowp, cols, vals

    idx = 0
    for i in range(N):
        ix = i % n
        iy = i // n

        rowp[i] = idx
        if iy > 0: # up
            vals[idx] = 1.0
            cols[idx] = i - n
            idx += 1
        else:
            b[i] -= 1.0

        if ix > 0: # left
            vals[idx] = 1.0
            cols[idx] = i - 1
            idx += 1
        else:
            pass

        # center
        vals[idx] = -4.0
        cols[idx] = i
        idx += 1

        if (ix < (n-1)):
            vals[idx] = 1.0
            cols[idx] = i + 1
            idx += 1
        else:
            pass

        if (iy < (n-1)):
            vals[idx] = 1.0
            cols[idx] = i + n
            idx += 1

    rowp[N] = idx

    # print(f"{rowp=}\n{cols=}\n{vals=}")

    rows = np.zeros((cols.shape[0],))
    for i in range(cols.shape[0]):
        if i==0:
            rows[i] = 0
        else:
            inc = cols[i] > cols[i-1]
            if inc:
                rows[i] = rows[i-1]
            else:
                rows[i] = rows[i-1] + 1
    # print(F"{rows=}")


    A = spp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, b

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
    print(f"{beta=}")
    
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
        # print(f"{V[:,0]=}")
        print(f"{g[0]=}")

        print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            w = A @ V[:, j]
            w = M_inv(w)  # Apply preconditioner if given

            # print(f"{j=} {w=}")

            # check norms:
            if j == 0:
                A_norm = np.linalg.norm(A @ V[:, j])
                MA_norm = np.linalg.norm(M_inv(A @ V[:, j]))
                # print(f"{A_norm=} {MA_norm=}")

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
                # print(f"({i=},{j=}) {H[i,j]=}")

            # print(f"post GS {w=}")

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

            # print(f"{V[:,j+1]=}")

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
                # print(f"{j=} {g[j]=} {g[j+1]=}")

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
        Hred = H[:j+1, :j+1]; gred = g[:j+1]
        # print(f"{Hred=} {gred=} {y=}")
        # import matplotlib.pyplot as plt
        # # print(f"{H[j,:j+1]=}")
        # # plt.imshow(H[:j+1, :j+1])
        # # print(f"{j+1=}")
        # plt.imshow(H)
        # plt.show()


        # Update solution
        x += V[:, :j+1] @ y

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
    
    print(f"final resid {beta=}")

    return x
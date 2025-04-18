import numpy as np

# https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
# use the first part of this to solve the eigenvalue problem
# on A = H^T H for H svd
# then I'll just do eig problem of B = H H^T since I only need rotation matrix UV^T 
# and don't care about the eigenvalues per say (I may not need perfect compatibility of UV^T?)
# or do I need to check that the eigenvalues are the same order? maybe.. we'll see how it works..

def givens_rot_mat(A, pair):
    i0 = pair[0]; i1 = pair[1]
    a11 = A[i0,i0]; a12 = A[i0, i1]; a22 = A[i1, i1]
    b = a12**2 < (a11 - a22)**2
    omega = 1.0 / np.sqrt(a12**2 + (a11 - a22)**2)
    s = omega * a12 if b else 0.5**0.5
    c = omega * (a11 - a22) if b else 0.5**0.5
    Qmat = np.zeros((3,3))
    Qmat[i0,i0] = c
    Qmat[i1,i1] = c
    i2 = (i1 + 1)%3
    Qmat[i2,i2] = 1.0
    Qmat[i0, i1] = -s
    Qmat[i1,i0] = s
    return Qmat

def eig3x3_givens(A_orig):
    A = A_orig.copy()
    V = np.eye(3)

    for i in range(15):
        for direc in range(3): # cyclic rotations of each pair (i,j)
            # pair = [(direc+1)%3, (direc+2)%3]
            pair = [direc, (direc+1)%3]
            Q = givens_rot_mat(A, pair)
            V = V @ Q
            A = Q.T @ A @ Q
            # print(f"{Q=}\n{A=}\n{V=}")
            # if direc == 2: exit()
            # print(f'{A=}')
            # exit()

    print(f"final {A=}")
    eig = np.diag(A)

    # now sort:: using conditional swap, I also 
    return eig, V


if __name__ == "__main__":

    # svd_QR(np.random.rand(3,3))
    # exit()

    # A = np.random.rand(3,3)
    # A = A + A.T # so symmetric, which ours is too

    H = np.array([1.0813e-03, 1.9075e-04, 7.4367e-06, 1.9075e-04, 1.0813e-03, -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07])
    H = H.reshape((3,3))
    print(f"{H=}")

    A = H.T @ H

    print(f"orig {A=}")

    eig, V = eig3x3_givens(A)
    print(f'{eig=}\n{V=}')
    # also double check A = V * eig * V^T
    Acheck = V @ np.diag(eig) @ V.T
    print(f"{Acheck=}")

    # compare to analytic
    eig2, V2 = np.linalg.eig(A)
    print(f"{eig2=}\n{V2=}")
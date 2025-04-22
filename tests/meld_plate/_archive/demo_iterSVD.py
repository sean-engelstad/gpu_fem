# demo the 3x3 SVD code of Aleka M.C. Adams et al. [fast efficient 3x3 SVD](https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf)

import numpy as np


def compute_givens_rotations(A_2x2, pair):
    a11 = A_2x2[0,0]; a12 = A_2x2[0,1]; a22 = A_2x2[1,1]
    b = a12**2 < (a11 - a22)**2
    omega = 1.0 / np.sqrt(a12**2 + (a11 - a22)**2)
    s = omega * a12 if b else 0.5**0.5
    c = omega * (a11 - a22) if b else 0.5**0.5
    Q = np.zeros((3,3))
    i0 = pair[0]; i1 = pair[1]
    irem = (i1+1)%3
    Q[i0,i0] = c
    Q[i0,i1] = -s
    Q[i1,i0] = s
    Q[i1,i1] = c
    Q[irem,irem] = 1
    Q_2x2 = Q[pair,:][:,pair]
    # print(f"{A_2x2=}\n{Q_2x2=}")
    # exit()
    return Q

def cyclic_jacobian_iteration(S_orig):
    # A is 3x3 matrix
    S = S_orig.copy()
    print(f"{S_orig=}")

    Qtot = np.eye(3)
    print(f"{Qtot=}")
    
    for iter in range(15): #15
        for cycle in range(3):
            pair = np.array([(cycle+1)%3, (cycle+2)%3])
            A_2x2 = A[pair, :][:, pair]

            # now compute givens rotation matrix
            Q = compute_givens_rotations(A_2x2, pair)
            # Q = compute_givens_rotations_quaternion(A_2x2, pair)
            print(f"{Q=}")

            A = Q.T @ A @ Q
            Qtot = Qtot @ Q
            # Qtot = Q @ Qtot
            print(f"{A=}")
            print(f"{Qtot=}")
        # exit()

    Afinal = A.copy()
    print(f"{Afinal=}")
    Lam = np.diag(A)
    return Lam, Qtot

if __name__=="__main__":
    A = np.random.rand(3,3)
    A = A + A.T # ensure symmetric
    Lam, Q = cyclic_jacobian_iteration(A)
    print(f"{Lam=}\n{Q=}")

    A2 = Q @ np.diag(Lam) @ Q.T
    print(f"{A2=} vs \n{A=}")
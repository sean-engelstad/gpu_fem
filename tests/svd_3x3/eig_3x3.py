import numpy as np

H = np.array([1.0813e-03, 1.9075e-04, 7.4367e-06, 1.9075e-04, 1.0813e-03, -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07])
H = H.reshape((3,3))
print(f"{H=}")

A = H.T @ H

print(f"{A=}")

sigsq, V = np.linalg.eig(A)

# swap order to match C++ right now?
# sigsq[1:3] = sigsq[np.array([2,1])]
# V[:, np.array([1,2])] = V[:, np.array([2, 1])]
VT = V.T

print(f"{sigsq=}\n{VT=}")
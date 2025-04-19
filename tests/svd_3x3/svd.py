import numpy as np

H = np.array([1.0813e-03, 1.9075e-04, 7.4367e-06, 1.9075e-04, 1.0813e-03, -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07])
H = H.reshape((3,3))
print(f"{H=}")

U, sig, VT = np.linalg.svd(H)

print(f"{U=}\n{sig=}\n{VT=}")

# also compute rotation matrix R
R = U @ VT
print(f"{R=}")
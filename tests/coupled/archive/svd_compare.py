import numpy as np

# aero_ind = 521 (small disp)
H0 = np.array([8.8035e-04, 1.9263e-04, 8.0189e-06, 1.5531e-04, 1.0920e-03, -7.0859e-06, 9.5657e-06, -8.7726e-06, 1.8221e-07])
VT0 = np.array([-4.8170e-01, -8.7634e-01, 1.8062e-03, -1.0866e-02, 8.0335e-03, 9.9991e-01, -8.7627e-01, 4.8163e-01, -1.3392e-02])
Sig0 = np.array([1.1900e-03, 7.7849e-09, 7.8291e-04])
U0 = np.array([-5.9286e-04, -9.8680e-11, -6.7876e-04, -1.0318e-03, 7.9937e-11, 3.8996e-04, 3.0803e-06, 7.7838e-09, -1.2610e-05])
R0 = np.array([9.9982e-01, 1.8923e-02, -1.9642e-03, -1.8919e-02, 9.9982e-01, 2.0308e-03, 2.0023e-03, -1.9933e-03, 1.0000e+00])

H0 = H0.reshape((3,3))
VT0 = VT0.reshape((3,3))
U0 = U0.reshape((3,3))
R0 = R0.reshape((3,3))

print(f"{U0=}\n{VT0=}\n{R0=}")
print(f"{np.linalg.det(U0)=:.4e} {np.linalg.det(VT0)=:.4e} {np.linalg.det(R0)=:.4e}")
print("\n------------------\n")

# aero_ind = 522 (large disp)
H1 = np.array([1.0813e-03, 1.9075e-04, 7.4367e-06, 1.9075e-04, 1.0813e-03, -7.4367e-06, 9.0287e-06, -9.0287e-06, 1.5622e-07])
VT1 = np.array([9.0756e-01, 3.8895e-01, 1.5830e-01, -1.4986e-01, -5.2155e-02, 9.8733e-01, -3.9228e-01, 9.1978e-01, -1.0956e-02])
Sig1 = np.diag(np.array([1.2721e-03, 5.4370e-09, 8.9074e-04]))
U1 = np.array([1.0567e-03, -1.6466e-04, -2.4882e-04, 5.9252e-04, -9.2325e-05, 9.1985e-04, 4.7071e-06, -7.2794e-07, -1.1848e-05])
R1 = np.array([4.5394e+03, 1.5796e+03, -2.9901e+04, 2.5448e+03, 8.8677e+02, -1.6766e+04, 2.0073e+01, 6.9720e+00, -1.3219e+02])

H1 = H1.reshape((3,3))
VT1 = VT1.reshape((3,3))
U1 = U1.reshape((3,3))
R1 = R1.reshape((3,3))

print(f"{U1=}\n{VT1=}\n{R1=}")
print(f"{np.linalg.det(U1)=:.4e} {np.linalg.det(VT1)=:.4e} {np.linalg.det(R1)=:.4e}")

# check norm of each matrix and its columns
for i in range(3):
    print(f'U0 col{i} norm : {np.linalg.norm(U0[:,i])}')
    print(f'U1 col{i} norm : {np.linalg.norm(U1[:,i])}')
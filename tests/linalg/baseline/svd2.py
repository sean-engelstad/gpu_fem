# H:3.16327e-02,1.26592e-02,-1.07461e-02,1.26592e-02,3.52551e-02,1.24888e-02,-1.07461e-02,1.24888e-02,3.57527e-02,
# H for matrix that should give exactly identity rotation, but is slightly off now

import numpy as np
H = np.array([3.16327e-02,1.26592e-02,-1.07461e-02,1.26592e-02,3.52551e-02,1.24888e-02,-1.07461e-02,1.24888e-02,3.57527e-02])
H = H.reshape((3,3))
print(f"{H=}")

U, sig, VT = np.linalg.svd(H)

print(f"{U=}\n{sig=}\n{VT=}")

# also compute rotation matrix R
R = U @ VT
print(f"{R=}")


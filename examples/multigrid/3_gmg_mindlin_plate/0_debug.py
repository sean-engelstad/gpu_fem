import numpy as np
import matplotlib.pyplot as plt
from _src import *


""" check lagrange interpolation basis """
# 9 point lagrange basis points
w_vals = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [7, 8, 9],
]).reshape((9,1))
_xi, _eta = np.linspace(-1, 1, 10), np.linspace(-1, 1, 9)
XI, ETA = np.meshgrid(_xi, _eta)
W = np.zeros_like(XI)

for i in range(9): # eta first
    for j in range(10):
        W[i,j] = lagrange_basis_2d(XI[i,j], ETA[i,j], w_vals)

fig, ax = plt.subplots(1, 1)
cax = fig.add_subplot(1,1,1, projection='3d')
cax.plot_surface(XI, ETA, W)
plt.show()

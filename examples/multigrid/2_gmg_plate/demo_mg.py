from src._dkt_plate_elem import *
import numpy as np
import matplotlib.pyplot as plt

# right triangle
x = np.array([0.0, 1.0, 0.0])
y = np.array([0.0, 0.0, 1.0])

# equilateral triangle
# x = np.array([0.0, 1.0, 0.5])
# y = np.array([0.0, 0.0, np.sqrt(3)/2.0])

eta = np.linspace(0.0, 1.0, 10)
xi = np.linspace(0.0, 1.0, 10) # technically xi goes from 0 to 1-eta on triangular boundary

XI, ETA = np.meshgrid(xi, eta)

# nodal DOF
U = np.zeros(9)
U[1::3] = np.array([0.1, 0.3, 0.5])
U[2::3] = np.array([-0.1, 0.2, 0.4])

# check Hx, Hy at some intermed disps
Hx, Hy = dkt_H_shape_funcs(x, y, 0.0, 1.0)
# print(f"{Hx=} {Hy=}")

BETAX = np.array([[get_rotations(U, x, y, xi[i], xi[j])[0] for i in range(10)] for j in range(10)])
BETAX[XI > (1.0 - ETA)] = np.nan # zero out past the triangular boundary (for plot)

# fig, ax = plt.subplots(1, 2)
# ax0 = fig.add_subplot(1, 2, 1, projection='3d')
# ax0.plot_surface(XI, ETA, BETAX)
# plt.show()

"""test the rot interp.. on small mesh"""
_nx, _ny = 4, 5
# _nx, _ny = 8, 9

x, y = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _ny)
X, Y = np.meshgrid(x, y)
print(f'{X.shape=}')
W = np.sin(np.pi * X) * np.sin(np.pi * Y)
W_x = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
W_y = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
TH_x = -W_y
TH_y = W_x
# exit()

# now interp the TH_x, TH_y
_hx = 1.0 / (_nx - 1)
_hy = 1.0 / (_ny - 1)
x_elem = _hx * np.array([0.0, 1.0, 0.0])
y_elem = _hy * np.array([0.0, 0.0, 1.0])
# xi, eta = 0.5, 0.0
# indexed by [iy, ix]
elem_vals = []

# ix_offset, iy_offset = 1, 1
# ix_offset, iy_offset = 0, 1
ix_offset, iy_offset = 1, 0

for ix_iy in [0, 1, 2]:
    _ix = ix_iy % 2
    _iy = ix_iy // 2
    
    ix, iy = _ix + ix_offset, _iy + iy_offset
    elem_vals += [W[iy, ix], TH_x[iy, ix], TH_y[iy, ix]]

xi_vec = np.linspace(0.0, 1.0, 10)
TH_X_hat = np.array([[ -get_rotations(elem_vals, x_elem, y_elem, xi_vec[i], xi_vec[j])[1] for i in range(10)] for j in range(10)])
TH_X_hat[XI > (1.0 - ETA)] = np.nan # zero out past the triangular boundary (for plot)

# fig, ax = plt.subplots(1, 1)
# ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot_surface(X, Y, TH_x, alpha=0.8, zorder=0)
# X_elem = _hx * (ix_offset + XI)
# Y_elem = _hy * (iy_offset + ETA)
# ax0.plot_surface(X_elem, Y_elem, TH_X_hat, zorder=1)
# plt.show()

# exit()

"""test derivatives of Hx, Hy w.r.t. xi and eta using complex step"""
xi, eta = 0.461, 0.321
# xi, eta = 0.321, 0.461
Hx_xi, Hy_xi, Hx_eta, Hy_eta = dkt_H_shape_func_grads(x_elem, y_elem, xi, eta)

_Hx_xi2, _Hy_xi2 = dkt_H_shape_funcs(x_elem, y_elem, xi + 1e-30 * 1j, eta)
Hx_xi2, Hy_xi2 = np.imag(_Hx_xi2) / 1e-30, np.imag(_Hy_xi2) / 1e-30
_Hx_eta2, _Hy_eta2 = dkt_H_shape_funcs(x_elem, y_elem, xi, eta + 1e-30 * 1j)
Hx_eta2, Hy_eta2 = np.imag(_Hx_eta2) / 1e-30, np.imag(_Hy_eta2) / 1e-30

err_mat = np.zeros((2, 18))
err_mat[0,:9] = Hx_xi - Hx_xi2
err_mat[0,9:] = Hy_xi - Hy_xi2
err_mat[1,:9] = Hx_eta - Hx_eta2
err_mat[1,9:] = Hy_eta - Hy_eta2
err2 = np.linalg.norm(err_mat)
# print(F"{err2=:.3e}")
# plt.imshow(err_mat)
# plt.show()
# matches now..

"""compute and solve the fine problem (ref soln)"""

# compute the element stiffness matrix for an aluminum material
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick
# quad_wt = 1.0
# Kelem = get_kelem(E, thick, nu, x, y)

# plt.imshow(Kelem)
# plt.show()

# let's now try and assemble the global stiffness matrix (dense form)
nxe = 32
# nxe = 8 # fine grid
# nxe = 4
# nxe = 4

K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = K.shape[0]

# now apply some loads to the structure
F = np.zeros(ndof)
F[0::3] = 100.0 # vert load for bending..

K, F = apply_bcs(nxe, K, F)
nx = nxe + 1

u = np.linalg.solve(K, F)

# now plot the solution 
x_plt, y_plt = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x_plt, y_plt)
w = u[0::3]
# w =  u[2::3]
W = np.reshape(w, (nx, nx))

# fig, ax = plt.subplots(1, 1)
# ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot_surface(X, Y, W)
# plt.show()

"""solve on coarser mesh now"""
nxe_c = nxe // 2
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
ndof_c = K_c.shape[0]
F_c = np.zeros(ndof_c)
F_c[0::3] = 100.0
K_c, F_c = apply_bcs(nxe_c, K_c, F_c)
u_c = np.linalg.solve(K_c, F_c)

# try prolongation now
P = prolongation_operator(nxe_c = nxe_c)
plt.imshow(P)
plt.show()

u_f_hat = P @ u_c
nx_c = nxe_c + 1

fine_bcs = get_bcs(nxe)
# u_f_hat[fine_bcs] = 0.0
# print(f'{u_f_hat[fine_bcs]=}')

# now measure also the nodal forces after interpolation
F_f_hat = np.dot(K, u_f_hat)
F_f_hat[fine_bcs] = 0.0 # can just zero this out (we don't smooth this..)

for idof in range(3):
    Disp = np.reshape(u[idof::3], (nx, nx))
    Disp_hat = np.reshape(u_f_hat[idof::3], (nx, nx))
    Load_hat = np.reshape(F_f_hat[idof::3], (nx, nx))

    fig, ax = plt.subplots(1, 3)
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax0.plot_surface(X, Y, Disp)
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.plot_surface(X, Y, Disp_hat)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.plot_surface(X, Y, Load_hat)
    dof_str = ['w', 'thx', 'thy'][idof]
    ax0.set_title(f"True {dof_str}")
    ax1.set_title(f"Pred {dof_str}")
    ax2.set_title(f"Pred Load")
    plt.show()

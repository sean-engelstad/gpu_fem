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
print(f"{Hx=} {Hy=}")

BETAX = np.array([[get_rotations(U, x, y, xi[i], xi[j])[0] for i in range(10)] for j in range(10)])
BETAX[XI > (1.0 - ETA)] = np.nan # zero out past the triangular boundary (for plot)

# fig, ax = plt.subplots(1, 2)
# ax0 = fig.add_subplot(1, 2, 1, projection='3d')
# ax0.plot_surface(XI, ETA, BETAX)
# plt.show()

# compute the element stiffness matrix for an aluminum material
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick
quad_wt = 1.0
Kelem = get_kelem(E, thick, nu, x, y)

# plt.imshow(Kelem)
# plt.show()

# let's now try and assemble the global stiffness matrix (dense form)
nxe = 32
# nxe = 8 # fine grid
# nxe = 4

nx = nxe + 1
N = nx**2
ndof = 3 * N # 3 dof per node

nelems = nxe**2 * 2 # x2 because we have two triangle elems in each quad element slot

K = np.zeros((ndof, ndof))

# unit square grid
h = 1.0 / nxe

for ielem in range(nelems):
    iquad = ielem // 2
    itri = ielem % 2
    ixe = iquad % nxe
    iye = iquad // nxe

    x1 = h * ixe
    x2 = x1 + h
    y1 = h * iye
    y2 = y1 + h
    n1 = nx * iye + ixe
    n3 = n1 + nx
    n2, n4 = n1 + 1, n3 + 1

    if itri == 0: # first tri element in quad slot
        x_elem = np.array([x1, x2, x1])
        y_elem = np.array([y1, y1, y2])
        local_nodes = [n1, n2, n3]

    else: # second tri element in quad slot
        x_elem = np.array([x2, x2, x1])
        y_elem = np.array([y1, y2, y2])
        local_nodes = [n2, n4, n3]

    Kelem = get_kelem(E, thick, nu, x_elem, y_elem)
    local_dof = [3*inode+idof for inode in local_nodes for idof in range(3)]
    # print(f"{iquad=} {itri=} {local_dof=}")

    arr_ind = np.ix_(local_dof, local_dof)
    # plt.imshow(Kelem)
    # plt.show()

    K[arr_ind] += Kelem

# plt.imshow(K)
# plt.show()

# now apply some loads to the structure
F = np.zeros(ndof)
F[0::3] = 100.0 # vert load for bending..

# now apply bcs to the stiffness matrix and forces
bcs = []
for iy in range(nx):
    for ix in range(nx):
        inode = nx * iy + ix

        if ix in [0, nx-1] or iy in [0, nx-1]:
            bcs += [3 * inode]
        elif ix in [0, nx-1]:
            bcs += [3 * inode + 2] # theta y = 0 on y=const edge
        elif iy in [0, nx-1]:
            bcs += [3 * inode + 1] # theta x = 0 on x=const edge
K[bcs,:] = 0.0
K[:,bcs] = 0.0
for bc in bcs:
    K[bc,bc] = 1.0
F[bcs] = 0.0

u = np.linalg.solve(K, F)

# now plot the solution 
x_plt, y_plt = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x_plt, y_plt)
w = u[0::3]
W = np.reshape(w, (nx, nx))

fig, ax = plt.subplots(1, 1)
ax0 = fig.add_subplot(1, 1, 1, projection='3d')
ax0.plot_surface(X, Y, W)
plt.show()
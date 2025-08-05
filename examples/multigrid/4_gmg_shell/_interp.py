"""
develop a coarse-fine or prolongation operator for TACS CQUAD4 FSDT MITC shells that works well near thin-plate limit
* previous basic avg interp of u,v,w and rot dof produces unrealistic transverse shear strains (explain more below)
* need a scheme that coarse-fine produces realistic transverse shear strains

* reason why before averaging interp doesn't produce right mag transverse shear strains:
    * actual transverse shear strains in an MITC element are only small or right magnitudes on the tying points (which helps prevent shear locking)
    * tranverse tying strains scale by 1/slenderness^2 (making them very small) => and their only small on tying points line

* I will now demonstrate on the plate geometry (after linear solve => will only cause the expected tying interp behavior),
    what the tying strain behavior in the element is and why my original coarse-fine interp breaks down => and how to change it
"""

import numpy as np
import matplotlib.pyplot as plt
from __src import get_tacs_matrix, plot_plate_vec
import scipy as sp
from scipy.sparse.linalg import spsolve

""" first let's solve the linear system on the coarse mesh """

SR = 1000.0 # fairly slender plate
thickness = 1.0 / SR
nxe = 32 # num elements in coarse mesh
_tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"in/plate{nxe}.bdf", thickness=thickness)
_tacs_csr_mat = _tacs_bsr_mat.tocsr()

disp = spsolve(_tacs_csr_mat, _rhs)

# choose element 10,10 with nodes:
ix, iy = 10, 10
nx = nxe + 1
inode = iy * nx + ix
elem_nodes = [inode, inode+1, inode+nx+1, inode+nx]
elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
# print(F'{inode=} {elem_nodes=} {elem_dof=}')
elem_disp = disp[elem_dof]
# print(f"{elem_disp=}")

# get also the xpt coordinates for this element
h = 1.0 / nxe
xpts_elem = h * np.array([[_inode % nx, _inode // nx, 0.0] for _inode in elem_nodes])
# print(f"{xpts_elem=}")

# plot to check right disp
# fig, ax = plt.subplots(1, 1)
# ax = fig.add_subplot(1,1,1, projection='3d')
# plot_plate_vec(nxe=16, vec=disp, ax=ax, sort_fw=None, nodal_dof=2, cmap='jet')
# plt.show()

""" now let's compute the disp, rot, dispgrad, strain fields on the coarse element"""

n = 11
xi, eta = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
XI, ETA = np.meshgrid(xi, eta)

N = [
    0.25 * (1 - XI) * (1 - ETA),
    0.25 * (1 + XI) * (1 - ETA),
    0.25 * (1 + XI) * (1 + ETA),
    0.25 * (1 - XI) * (1 + ETA),
]

def interp(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N[i] * nodal_vals[i]
    return VAL

""" plot dof fields w, thx, thy """

W = interp(elem_disp[0::3])
THX = interp(elem_disp[1::3])
THY = interp(elem_disp[2::3])

def plot_multi(DOFS, dof_strs):
    n_plot = len(DOFS)
    assert(n_plot == len(dof_strs))
    
    fig, ax = plt.subplots(1, n_plot)
    for i in range(n_plot):
        _ax = fig.add_subplot(1, n_plot, i+1, projection='3d')
        _ax.plot_surface(XI, ETA, DOFS[i])
        _ax.set_title(dof_strs[i])
    plt.show()

plot_multi(
    DOFS=[W, THX, THY],
    dof_strs=['w', 'thx', 'thy']
)

""" define disp grad interp methods """

N_XI = [
    -0.25 * (1 - ETA),
    0.25 * (1 - ETA),
    0.25 * (1 + ETA),
    -0.25 * (1 + ETA),
]

N_ETA = [
    -0.25 * (1 - XI),
    -0.25 * (1 + XI),
    0.25 * (1 + XI),
    0.25 * (1 - XI),
]

def interp_dxi(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N_XI[i] * nodal_vals[i]
    return VAL

def interp_deta(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N_ETA[i] * nodal_vals[i]
    return VAL

def interp_dx(nodal_vals):
    return interp_dxi(nodal_vals) / (0.5 * h) # scale by dx elem size

def interp_dy(nodal_vals):
    return interp_deta(nodal_vals) / (0.5 * h) # scale by dx elem size

""" now let's compute disp grads """

W_X = interp_dx(elem_disp[0::3])
W_Y = interp_dy(elem_disp[0::3])

# W_X is linear in y or eta (constant in x and xi)
# W_Y is linear in x or xi (constant in y and eta)
# plot_multi(
#     DOFS=[W_X, W_Y],
#     dof_strs=['w_x', 'w_y'],
# )

""" transverse shear bending strains """
GAM_13 = h/4.0 * (THY + W_X)
GAM_23 = h/4.0 * (W_Y - THX)

# get gam_13, gam_23 at tying points.. compared to gam_13 and gam_23 overall mag.. (approx locations..)
# also meshgrid flips so [eta,xi] order of arsg
GAM_13_TY = np.array([GAM_13[2, 5], GAM_13[8, 5]])
GAM_23_TY = np.array([GAM_23[5, 2], GAM_23[5, 8]])

# now normalize by mags
GAM_13_MAG = np.max(np.abs(GAM_13))
GAM_23_MAG = np.max(np.abs(GAM_23))
GAM_13_TY_NRM = GAM_13_TY / GAM_13_MAG
GAM_23_TY_NRM = GAM_23_TY / GAM_23_MAG

print(f"{GAM_13_TY=} {GAM_13_MAG=} {GAM_13_TY_NRM=}")
print(f"{GAM_23_TY=} {GAM_23_MAG=} {GAM_23_TY_NRM=}")

# basically gam_13 is near zero on xi=0, eta line (of gam_13 tying points)
# and gam_23 near zero on eta=0, xi changing line (of gam_23 tying pts)
#    recall xi in [-1,1] and same for eta

plot_multi(
    DOFS=[GAM_13, GAM_23],
    dof_strs=['gam13', 'gam23'],
)

# note the normalized gam_13 and gam_23 scale by approx 1/SR (so get very small near thin-walled limit)

# NOTE : this means if we do averaging, the shear strains on tying strains of the fine points
#   for example xi=-0.5 line doesn't have small shear strains of near thin-plate limit
#   we need to explicitly enforce this by making fine interped elems have similar mag shear strains (with MITC)

""" develop coarse-fine prolongation scheme that respect thin-plate limit for MITC elements"""

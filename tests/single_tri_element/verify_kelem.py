import numpy as np

# verify C++ element residual and matrix on a single triangle element:

# consider a triangular element with 3 mesh nodes
# and 6 basis nodes

geo_conn = [0, 1, 2]
vars_conn = [0, 1, 2, 3, 4, 5]

xpts = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 2.0])
xpts = np.reshape(xpts, newshape=(3,2))
E = 70e9
nu = 0.3
t = 0.005

q = np.array([0, 0, 3, 1, 3, -1, 3, 0, 1.5, -0.5, 1.5, 0.5])
q = np.reshape(q, newshape=(6,2))

print(f"{xpts=}\n {q=}")

nvars = 12
res = np.zeros((nvars))
mat = np.zeros((nvars, nvars))

quad_pts = np.array([
    [0.5, 0.5],
    [0.0, 0.5],
    [0.5, 0.0],
])

# the geometry interpolation gradients
dNdxi_geom = np.array([
    [-1, -1],
    [1, 0],
    [0, 1]
])

# compute the element residual
for iquad in range(3):
    pt = quad_pts[iquad]
    weight = 1.0/3.0

    # use geom to compute J = dX/dxi jacobian
    J = xpts.T @ dNdxi_geom
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    print(f"{J=}\n {detJ=}\n {Jinv=}")

    # use basis to compute disp gradients in computation space
    xi = pt[0]
    eta = pt[1]
    dNdxi_basis = np.array([
        [4 * xi + 4 * eta - 3] * 2,
        [4 * xi  - 1, 0.0],
        [0.0, 4 * eta - 1],
        [4 * eta, 4 * xi],
        [-4 * eta, -4 * xi - 8 * eta + 4],
        [-8 * xi - 4 * eta + 4, -4 * xi],
    ])
    dUdxi = q.T @ dNdxi_basis

    # convert disp grad to physical space
    dUdx = dUdxi @ Jinv
    scale = detJ * weight

    # compute strain energy for linear elasticity
    strain = 0.5 * (dUdx + dUdx.T)
    # hooke's law for the stress-strain relationship
    C = E / (1.0 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, 1-nu]
    ])
    strain_flat = np.array([strain[0,0], strain[1,1], strain[0,1]]).reshape((3,1))
    stress_flat = C @ strain_flat
    energy = 0.5 * scale * np.dot(stress_flat[:,0], strain_flat[:,0])
    stress_vec = stress_flat[:,0]
    stress = np.array([
        [stress_vec[0], stress_vec[2]], 
        [stress_vec[2], stress_vec[1]]
    ])
    # need derivatives denergy/d(dUdx)
    dUdx_bar = 0.5 * scale * (stress + stress.T)
    
    # now backprop to outer residual
    dUdxi_bar = dUdx_bar.T @ Jinv
    q_bar = dNdxi_basis @ dUdxi_bar.T

    # add into residual
    res += q_bar.reshape((12,))

print(f"{res=}")
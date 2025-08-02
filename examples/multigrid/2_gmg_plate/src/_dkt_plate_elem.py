"""DKT kirchoff plate triangular element by https://web.mit.edu/kjb/www/Publications_Prior_to_1998/A_Study_of_Three-Node_Triangular_Plate_Bending_Elements.pdf"""

import numpy as np

def triang_basis(xi, eta):
    return np.array([
        2.0 * (1 - xi - eta) * (0.5 - xi - eta),
        xi * (2 * xi - 1.0),
        eta * (2.0 * eta - 1.0),
        4.0 * xi * eta,
        4.0 * eta * (1.0 - xi - eta),
        4.0 * xi * (1.0 - xi - eta),
    ])

def dkt_Hw_shape_funcs(x, y, xi, eta):
    """shape functions for w interp"""
    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    # shift by one so one-based (easier to make y12 equiv to y[3] more readable)
    xij = np.concatenate([np.zeros(1), xij])
    yij = np.concatenate([np.zeros(1), yij])

    Hw = np.array([
        1 - xi - eta,
        0.5 * (1 - xi - eta) * (-yij[3] * xi + yij[2] * eta),
        0.5 * (1 - xi - eta) * (xij[3] * xi - xij[2] * eta),
        xi,
        0.5 * xi * (-yij[1] * eta + yij[3] * (1 - xi - eta)),
        0.5 * xi * (xij[1] * eta - xij[3] * (1 - xi - eta)),
        eta,
        0.5 * eta * (-yij[2] * (1 - xi - eta) + yij[1] * xi),
        0.5 * eta * (xij[2] * (1 - xi - eta) - xij[1] * xi),
    ])
    return Hw

def dkt_H_shape_funcs(x, y, xi, eta):
    N = triang_basis(xi, eta)

    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    lij = np.sqrt(xij**2 + yij**2)
    a = -xij / lij**2
    b = 0.75 * xij * yij / lij**2
    c = (0.25 * xij**2 - 0.5 * yij**2) / lij**2
    d = -yij / lij**2
    e = (0.25 * yij**2 - 0.5 * xij**2) / lij**2

    # now offset stuff so we can type it the same as 4,5,6 here like page 36 of the pdf
    # less error prone
    a = np.concatenate([np.zeros(4), a])
    b = np.concatenate([np.zeros(4), b])
    c = np.concatenate([np.zeros(4), c])
    d = np.concatenate([np.zeros(4), d])
    e = np.concatenate([np.zeros(4), e])
    N = np.concatenate([np.zeros(1), N])

    # Hx and Hy
    Hx, Hy = [], []
    for loop in range(3):
        if loop == 0:
            i, j, m = 5, 6, 1
        elif loop == 1:
            i, j, m = 6, 4, 2
        else:
            i, j, m = 4, 5, 3

        Hx += [
            1.5 * (a[j] * N[j] - a[i] * N[i]),
            b[i] * N[i] + b[j] * N[j],
            N[m] - c[i] * N[i] - c[j] * N[j]
        ]

        Hy += [
            1.5 * (d[j] * N[j] - d[i] * N[i]),
            -N[m] + e[i] * N[i] + e[j] * N[j],
            -(b[i] * N[i] + b[j] * N[j])
        ]

    Hx = np.array(Hx)
    Hy = np.array(Hy)

    return Hx, Hy

def dkt_H_shape_func_grads(x, y, xi, eta):
    """get d[Hx, Hy]/d[xi, eta] derivs"""
    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    lij = np.sqrt(xij**2 + yij**2)

    a = -xij / lij**2
    b = 0.75 * xij * yij / lij**2
    c = (0.25 * xij**2 - 0.5 * yij**2) / lij**2
    d = -yij / lij**2
    e = (0.25 * yij**2 - 0.5 * xij**2) / lij**2

    P = 6 * a
    q = 4 * b
    t = 6 * d
    r = 3 * yij**2 / lij**2   

    # now offset stuff so we can type it the same as 4,5,6 here like page 36 of the pdf
    P = np.concatenate([np.zeros(4), P])
    q = np.concatenate([np.zeros(4), q])
    t = np.concatenate([np.zeros(4), t])
    r = np.concatenate([np.zeros(4), r])

    Hx_xi = np.array([
        P[6] * (1 - 2 * xi) + (P[5] - P[6]) * eta,
        q[6] * (1 - 2*xi) - (q[5] + q[6]) * eta,
        -4 + 6 * (xi + eta) + r[6] * (1.0 - 2 * xi) - eta * (r[5] + r[6]),
        -P[6] * (1 - 2 * xi) + eta * (P[4] + P[6]),
        q[6] * (1 - 2 * xi) - eta * (q[6] - q[4]),
        -2 + 6 * xi + r[6] * (1 - 2 * xi) + eta * (r[4] - r[6]),
        -eta * (P[5] + P[4]),
        eta * (q[4] - q[5]),
        -eta * (r[5] - r[4]),
    ])

    Hy_xi = np.array([
        t[6] * (1 - 2 * xi) + (t[5] - t[6]) * eta,
        1.0 + r[6] * (1 - 2*xi) - (r[5] + r[6]) * eta,
        -q[6] * (1 - 2 * xi) + eta * (q[5] + q[6]),
        -t[6] * (1 - 2 * xi) + eta * (t[4] + t[6]),
        -1 + r[6] * (1-2*xi) + eta * (r[4] - r[6]),
        -q[6] * (1 - 2 * xi) - eta * (q[4] - q[6]),
        -eta * (t[4] + t[5]),
        eta * (r[4] - r[5]),
        -eta * (q[4] - q[5]),
    ])

    Hx_eta = np.array([
        -P[5] * (1 - 2 * eta) - xi * (P[6] - P[5]),
        q[5] * (1 - 2 * eta) - xi * (q[5] + q[6]),
        -4 + 6 * (xi + eta) + r[5] * (1 - 2 * eta) - xi * (r[5] + r[6]),
        xi * (P[4] + P[6]),
        xi * (q[4] - q[6]),
        -xi * (r[6] - r[4]),
        P[5] * (1 - 2 * eta) - xi * (P[4] + P[5]),
        q[5] * (1 - 2 * eta) + xi * (q[4] - q[5]),
        -2 + 6 * eta + r[5] * (1 - 2 * eta) + xi * (r[4] - r[5]),
    ])

    Hy_eta = np.array([
        -t[5] * (1 - 2 * eta) - xi * (t[6] - t[5]),
        1 + r[5] * (1 - 2 * eta) - xi * (r[5] + r[6]),
        -q[5] * (1 - 2 * eta) + xi * (q[5] + q[6]),
        xi * (t[4] + t[6]),
        xi * (r[4] - r[6]),
        -xi * (q[4] - q[6]),
        t[5] * (1 - 2 * eta) - xi * (t[4] + t[5]),
        -1 + r[5] * (1 - 2 * eta) + xi * (r[4] - r[5]),
        -q[5] * (1 - 2 * eta) - xi * (q[4] - q[5]),
    ])

    return Hx_xi, Hy_xi, Hx_eta, Hy_eta

def add_element_quadpt_jacobian(E, thick, nu, x, y, xi, eta):
    Hx_xi, Hy_xi, Hx_eta, Hy_eta = dkt_H_shape_func_grads(x, y, xi, eta)

    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])

    Hx_xi = np.expand_dims(Hx_xi, axis=0)
    Hy_xi = np.expand_dims(Hy_xi, axis=0)
    Hx_eta = np.expand_dims(Hx_eta, axis=0)
    Hy_eta = np.expand_dims(Hy_eta, axis=0)

    area = 0.5 * (xij[1] * yij[-1] - xij[-1] * yij[1])
    B = 0.5 / area * np.concatenate([
        yij[1] * Hx_xi + yij[-1] * Hx_eta,
        -xij[1] * Hy_xi + xij[-1] * Hy_eta,
        -xij[1] * Hx_xi - xij[-1] * Hx_eta + yij[1] * Hy_xi + yij[-1] * Hy_eta,
    ])
    # print(F"{B.shape=}")

    Db = E * thick**3 / 12.0 / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0],
        [0, 0, 0.5 * (1.0 - nu)]
    ])

    # add Kelem at quadrature point
    K = 2.0 * area * B.T @ Db @ B
    return K

def get_kelem(E, thick, nu, x, y):
    """use three point quadrature (since integrand is purely quadratic only need 3 points)"""
    # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    xi_vec = np.array([4.0, 1.0, 1.0]) / 6.0
    eta_vec = np.array([1.0, 4.0, 1.0]) / 6.0
    weights = np.array([1.0]*3) / 3.0

    Kelem = np.zeros((9, 9))

    for i in range(3):
        xi, eta, wt = xi_vec[i], eta_vec[i], weights[i]

        Kelem += wt * add_element_quadpt_jacobian(E, thick, nu, x, y, xi, eta)
    return Kelem

def get_rotations(elem_disp, x, y, xi, eta):
    Hx, Hy = dkt_H_shape_funcs(x, y, xi, eta)
    betax = np.dot(Hx, elem_disp)
    betay = np.dot(Hy, elem_disp)
    return betax, betay

def assemble_stiffness_matrix(nxe, E, thick, nu):
    """assemble the global stiffness matrix for a unit square regular grid with triangle elements"""

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

    return K

def get_bcs(nxe):
    # now apply bcs to the stiffness matrix and forces
    nx = nxe + 1
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
    return bcs

def apply_bcs(nxe, K, F):
    """apply bcs to the global stiffness matrix and forces"""
    bcs = get_bcs(nxe)
    K[bcs,:] = 0.0
    K[:,bcs] = 0.0
    for bc in bcs:
        K[bc,bc] = 1.0
    F[bcs] = 0.0

    return K, F

def prolongation_operator(nxe_c):
    """go coarse to fine with 2x grid multiplier on coarse disps"""
    nxe_f = nxe_c * 2
    nx_f = nxe_f + 1
    nx_c = nxe_c + 1
    N_c = nx_c**2
    N_f = nx_f**2
    ndof_c, ndof_f = 3 * N_c, 3 * N_f

    H = 1.0 / nxe_c # coarse mesh step size

    P = np.zeros((ndof_f, ndof_c))

    # interpolate
    for i_f in range(ndof_f):
        inode_f = i_f // 3
        idof = i_f % 3
        iy_f = inode_f // nx_f
        ix_f = inode_f % nx_f

        # print(f"{ix_f=} {iy_f=}")

        ix_c, iy_c = ix_f // 2, iy_f // 2
        ix_cr, iy_cr = ix_f % 2, iy_f % 2 
        inode_c = nx_c * iy_c + ix_c

        if ix_cr == 0 and iy_cr == 0:
            # fine node on top of coarse node
            i_c = 3 * inode_c + idof
            P[i_f, i_c] = 1.0

        elif ix_c == nx_c - 1 or iy_c == nx_c - 1:
            # far right or far left elems
            if ix_c == nx_c -1:
                xi, eta = 0.5, 0.0
                _inode_c = inode_c - 1
            elif iy_c == nx_c - 1:
                xi, eta = 0.0, 0.5
                _inode_c = inode_c - nx_c

            elem_nodes = [_inode_c+1, _inode_c+nx_c+1, _inode_c + nx_c]
            elem_dof = np.array([3 * _node + _dof for _node in elem_nodes for _dof in range(3)])
            x_elem = H * np.array([1.0, 1.0, 0.0])
            y_elem = H * np.array([0.0, 1.0, 1.0])
            Hw = dkt_Hw_shape_funcs(x_elem, y_elem, xi, eta)
            Hx, Hy = dkt_H_shape_funcs(x_elem, y_elem, xi, eta)

            if idof == 0: # w dof
                P[i_f, elem_dof] = Hw
            elif idof == 1: # th_x dof
                P[i_f, elem_dof] = -Hy
            elif idof == 2: # th_y dof
                P[i_f, elem_dof] = Hx

        else:
            # fine node on edges of coarse quad
            # since right bndry nodes treated separately each set of bndry nodes is treated first on its left side node
            # NOTE : no need to symmetrize it, I think you can show the interp on an edge only depends upon endpt quantities on that edge?
            xi = 0.5 if ix_cr == 1 else 0.0
            eta = 0.5 if iy_cr == 1 else 0.0

            # print(f"{inode_c=} {ix_c=} {iy_c=} {nx_c=}")

            elem_nodes = [inode_c, inode_c + 1, inode_c + nx_c]
            elem_dof = np.array([3 * _node + _dof for _node in elem_nodes for _dof in range(3)])
            x_elem = H * np.array([0.0, 1.0, 0.0])
            y_elem = H * np.array([0.0, 0.0, 1.0])
            Hw = dkt_Hw_shape_funcs(x_elem, y_elem, xi, eta)
            Hx, Hy = dkt_H_shape_funcs(x_elem, y_elem, xi, eta)
            
            if idof == 0: # w dof
                P[i_f, elem_dof] = Hw
            elif idof == 1: # th_x dof
                P[i_f, elem_dof] = -Hy
            elif idof == 2: # th_y dof
                P[i_f, elem_dof] = Hx

    # get bcs also on each side..
    fine_bcs = get_bcs(nxe_f)
    coarse_bcs = get_bcs(nxe_c)
    P[fine_bcs,:] = 0.0
    P[:,coarse_bcs] = 0.0

    return P


def restriction(nxe_f, u_f):
    assert(nxe_f % 2 == 0)
    nxe_c = nxe_f // 2


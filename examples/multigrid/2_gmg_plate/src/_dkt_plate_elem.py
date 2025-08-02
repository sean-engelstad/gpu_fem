"""DKT kirchoff plate triangular element by https://web.mit.edu/kjb/www/Publications_Prior_to_1998/A_Study_of_Three-Node_Triangular_Plate_Bending_Elements.pdf"""

import numpy as np

def triang_basis(xi, eta):
    return np.array([
        2. * (1 - xi - eta) * (0.5 - xi -eta),
        xi * (2 * xi - 1.0),
        eta * (2.0 * eta - 1.0),
        4.0 * xi * eta,
        4.0 * eta * (1 - xi - eta),
        4.0 * xi * (1.0 - xi - eta),
    ])

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

    # Hx and Hy 1-3 each
    Hx = [
        1.5 * (a[2] * N[5] - a[1] * N[4]),
        b[1] * N[4] + b[2] * N[5],
        N[0] - c[1] * N[4] - c[2] * N[5],
    ]
    Hy = [
        1.5 * (d[2] * N[5] - d[1] * N[4]),
        -N[0] + e[1] * N[4] + e[2] * N[5],
        -Hx[1],
    ]

    # now repeat for Hx and Hy 4-6
    Hx += [
        1.5 * (a[0] * N[3] - a[2] * N[5]),
        b[2] * N[5] + b[0] * N[3],
        N[1] - c[2] * N[5] - c[0] * N[3],
    ]
    Hy += [
        1.5 * (d[0] * N[3] - d[2] * N[5]),
        -N[1] + e[2] * N[5] + e[0] * N[3],
        -Hx[1],
    ]

    # now repeat for Hx and Hy 7-9
    Hx += [
        1.5 * (a[1] * N[4] - a[0] * N[3]),
        b[0] * N[3] + b[1] * N[4],
        N[2] - c[0] * N[3] - c[1] * N[4],
    ]
    Hy += [
        1.5 * (d[1] * N[4] - d[0] * N[3]),
        -N[2] + e[0] * N[3] + e[1] * N[4],
        -Hx[1],
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
        -q[6] * (1 - 2 * xi) + eta * (q[4] + q[6]),
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

    print(F"{B.shape=}")

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
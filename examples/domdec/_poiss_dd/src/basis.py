import numpy as np

# =============================
# QUADRATURE
# =============================

def zero_order_quadrature():
    return [0.0], [2.0] # why 2.0 here? because 1/2 in integration?

def first_order_quadrature():
    irt3 = np.sqrt(1.0/3.0)
    return [-irt3, irt3], [1.0, 1.0]

def second_order_quadrature():
    rt35 = np.sqrt(3.0 / 5.0)
    return [-rt35, 0.0, rt35], [5.0/9.0, 8.0/9.0, 5.0/9.0]

def third_order_quadrature():
    a =  0.8611363115940526
    b = 0.3399810435848563
    wa = 0.3478548451374539
    wb = 0.6521451548625461
    return [-a, -b, b, a], [wa, wb, wb, wa]

# =============================
# BASIS
# =============================


def get_lagrange_basis(xi):
    N = 0.5 * np.array([1.0 - xi, 1.0 + xi])
    dN = 0.5 * np.array([-1.0, 1.0])
    return N, dN


def get_lagrange_basis_2d_all(xi, eta):
    N1, dN1 = get_lagrange_basis(xi)
    N2, dN2 = get_lagrange_basis(eta)
    N = np.zeros(4);  Nxi = np.zeros(4);  Neta = np.zeros(4)
    
    # for n in range(4):
    #     i = n % 2; j = n // 2
    mapping = [(0,0), (1,0), (1,1), (0,1)]
    for n,(i,j) in enumerate(mapping):
        N[n] = N1[i] * N2[j]
        Nxi[n] = dN1[i] * N2[j]
        Neta[n] = N1[i] * dN2[j]
    return N, Nxi, Neta
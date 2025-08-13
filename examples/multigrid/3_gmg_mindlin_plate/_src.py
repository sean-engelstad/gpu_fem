import numpy as np
import matplotlib.pyplot as plt

def lagrange_basis_1d(xi, vals):
    """1d order 2, basis interp (3 points)"""
    result = -0.5 * xi * (1.0 - xi) * vals[0]
    result += (1.0 - xi**2) * vals[1]
    result += 0.5 * xi * (1.0 + xi) * vals[2]
    return result

def lagrange_basis_1d_grad(xi, vals):
    """get d/dxi derivs of the 1D lagrange basis"""
    result = (xi - 0.5) * vals[0]
    result += (-2.0 * xi) * vals[1]
    result += (xi + 0.5) * vals[2]
    return result

def modified_lbasis_1d(xi, vals):
    """modified lagrange basis => becomes linear from quadratic for consistent interp"""
    result = (1.0 / 6.0 - xi/2.0) * vals[0]
    result += 2.0 / 3.0 * vals[1]
    result += (1.0 / 6.0 + xi/2.0) * vals[2]
    return result

def lagrange_basis_2d(xi, eta, vals):
    """2d order 2, basis interp (9 points)"""
    # sum-factor method, first interp over each eta and add in
    eta_vals = [lagrange_basis_1d(xi,vals[3*i:(3*i+3)]) for i in range(3)]
    return lagrange_basis_1d(eta, eta_vals)

def lagrange_basis_2d_grad(xi, eta, vals, deriv:int=0):
    """2d order 2, basis interp (9 points)"""
    if deriv == 0: # d/dxi deriv
        eta_vals = [lagrange_basis_1d_grad(xi,vals[3*i:(3*i+3)]) for i in range(3)]
        return lagrange_basis_1d(eta, eta_vals)

    else: # d/deta deriv
        eta_vals = [lagrange_basis_1d(xi,vals[3*i:(3*i+3)]) for i in range(3)]
        return lagrange_basis_1d_grad(eta, eta_vals)

def get_xpts_basis(xi, eta, xpts):
    """get xpts a_xi, a_eta, a_zeta natural coords basis"""

    # get nodal vals here
    xyz = [xpts[i::3] for i in range(3)]

    # get param grads
    r_xi = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=0) for _vec in xyz])
    r_eta = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=1) for _vec in xyz])

    # magnitudes
    A_xi, A_eta = np.linalg.norm(r_xi), np.linalg.norm(r_eta)

    # unit vecs for basis
    a_xi, a_eta = r_xi / A_xi, r_eta / A_eta
    a_gam = np.cross(a_xi, a_eta)
    a_gam /= np.linalg.norm(a_gam)

    return a_xi, a_eta, a_gam, A_xi, a_eta

def get_param_strains(xi, eta, xpts, vars):
    """get all 8 strains at (xi,eta) in dxi, deta param coords""" 
    strains = np.zeros(8)

    # first get the relevant xpts basis data
    _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    # get nodal disp grads in dxi, deta (with 5 DOF per node [u, v, w, alpha, beta], no drill it's elim from eqns)
    vars_list = [vars[i::5] for i in range(5)]
    U_xi = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=0) for _vec in vars_list])
    U_eta = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=1) for _vec in vars_list])

    # get also the nodal disps & rotations
    U = np.array([lagrange_basis_2d(xi, eta, _vec) for _vec in vars_list])

    # TODO : apply corrected bases for these later for consistent interp

    # first the midplane strains
    strains[0] = U_xi[0] / A_xi # eps_11
    strains[1] = U_eta[1] / A_eta # eps_22
    strains[2] = 0.5 * (U_xi[1] / A_xi + U_eta[0] / A_eta)

    # then the bending strains
    strains[3] = U_xi[3] / A_xi # k_11
    strains[4] = U_eta[4] / A_eta # k_22
    strains[5] = 0.5 * (U_xi[4] / A_xi + U_eta[3] / A_eta)

    # then the transverse shear strains
    strains[6] = 0.5 * (U[3] + U_xi[2] / A_xi)
    strains[7] = 0.5 * (U[4] + U_eta[2] / A_eta)

    return strains
import numpy as np
import matplotlib.pyplot as plt

def lagrange_basis_1d(xi, vals):
    """1d order 2, basis interp (3 points)"""
    result = -0.5 * xi * (1.0 - xi) * vals[0]
    result += (1.0 - xi**2) * vals[1]
    result += 0.5 * xi * (1.0 + xi) * vals[2]
    return result

def lagrange_basis_1d_transpose(xi, out_bar):
    """1d order 2, basis interp (3 points)"""
    vals_bar = np.zeros(3)
    vals_bar[0] = -0.5 * xi * (1.0 - xi) * out_bar
    vals_bar[1] += (1.0 - xi**2) * out_bar
    vals_bar[2] += 0.5 * xi * (1.0 + xi) * out_bar
    return vals_bar

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

def modified_lbasis_1d_transpose(xi, out_bar):
    """modified lagrange basis => becomes linear from quadratic for consistent interp"""
    vals_bar = np.zeros(3)
    vals_bar = (1.0 / 6.0 - xi/2.0) * out_bar
    vals_bar += 2.0 / 3.0 * out_bar
    vals_bar += (1.0 / 6.0 + xi/2.0) * out_bar
    return vals_bar

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

def get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts):
    """get Tinv matrix.."""

    a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

    # get shell normals here in local cartesian basis first
    e_x = shell_xi_axis.copy()
    e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
    e_y = np.cross(a_gam, e_y)
    e_z = a_gam.copy() # since e_z, a_gam assumed same direction

    # make transform from global disps to local cartesian basis
    T_LG = np.zeros((3,3))
    T_LG[0] = e_x
    T_LG[1] = e_y
    T_LG[2] = e_z
    # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

    # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
    cross_nrm = np.linalg.norm(np.cross(a_xi, a_eta))
    T = np.array([
        [np.dot(e_y, a_eta), -np.dot(e_y, a_xi), 0.0],
        [-np.dot(e_x, a_eta), np.dot(e_x, a_xi), 0.0],
        [0.0, 0.0, cross_nrm]
    ]) / cross_nrm
    return T

def get_cov_nodal_disps(shell_xi_axis, xpts, vars):
    """compute shell normals and covariant basis, then rotate disps and rotations to each local nodal cov basis (uvw-tildes, etc.)"""
    # you do need the displacements rotated in their natural coord basis
    # to account for curvature effects

    vars_cov = np.zeros_like(vars)

    for node in range(9):
        inode, jnode = node % 3, node // 3
        xi, eta = 1.0*(inode-1), 1.0*(jnode-1)

        # get cov basis at the current node
        a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

        # get shell normals here in local cartesian basis first
        e_x = shell_xi_axis.copy()
        e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
        e_y = np.cross(a_gam, e_y)
        e_z = a_gam.copy() # since e_z, a_gam assumed same direction

        # make transform from global disps to local cartesian basis
        T_LG = np.zeros((3,3))
        T_LG[0] = e_x
        T_LG[1] = e_y
        T_LG[2] = e_z
        # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

        # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
        Tinv = np.array([
            [np.dot(e_x, a_xi), np.dot(e_y, a_xi), 0],
            [np.dot(e_x, a_eta), np.dot(e_y, a_eta), 0],
            [0, 0, 1.0],
        ])
        Tinv_planar = Tinv[:2, :][:, :2]

        # rotate the current shell disps and rotations.. (note u_z doesn't get modified, so you can copy that in if you want)
        uvw_loc_xyz = np.dot(T_LG, vars[5*node:(5*node+3)])
        thx, thy = vars[5*node+3], vars[5*node+4] # copy thx, thy (but flip order and signs based on director based on J. fish paper here..)
        rots = np.array([thy, -thx]) # comes from directors and cross product..

        vars_cov[5*node:(5*node+3)] = np.dot(Tinv, uvw_loc_xyz) # uv rotate
        vars_cov[(5*node+3):(5*node+5)] = np.dot(Tinv_planar, rots) # alpha,beta rotate

    return vars_cov

def get_cov_nodal_disps_transpose(shell_xi_axis, xpts, vars_cov_bar):
    """compute shell normals and covariant basis, then rotate disps and rotations to each local nodal cov basis (uvw-tildes, etc.)"""
    # you do need the displacements rotated in their natural coord basis
    # to account for curvature effects

    vars_bar = np.zeros_like(vars_cov_bar)

    for node in range(9):
        inode, jnode = node % 3, node // 3
        xi, eta = 1.0*(inode-1), 1.0*(jnode-1)

        # get cov basis at the current node
        a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

        # get shell normals here in local cartesian basis first
        e_x = shell_xi_axis.copy()
        e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
        e_y = np.cross(a_gam, e_y)
        e_z = a_gam.copy() # since e_z, a_gam assumed same direction

        # make transform from global disps to local cartesian basis
        T_LG = np.zeros((3,3))
        T_LG[0] = e_x
        T_LG[1] = e_y
        T_LG[2] = e_z
        # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

        # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
        Tinv = np.array([
            [np.dot(e_x, a_xi), np.dot(e_y, a_xi), 0],
            [np.dot(e_x, a_eta), np.dot(e_y, a_eta), 0],
            [0, 0, 1.0],
        ])
        Tinv_planar = Tinv[:2, :][:, :2]

        # now begin transpose steps
        uvw_loc_xyz_bar = np.dot(Tinv_planar.T, vars_cov_bar[5*node:(5*node+3)])
        vars_bar[5*node:(5*node+3)] += np.dot(T_LG.T, uvw_loc_xyz_bar)

        rots_bar = np.dot(Tinv_planar, vars_cov_bar[(5*node+3):(5*node+5)])
        thx_bar, thy_bar = -rots_bar[1], rots_bar[0]
        vars_bar[5*node+3] += thx_bar
        vars_bar[5*node+4] += thy_bar

    return vars_bar

def get_barlow_transform(node, xpts, direc:int=0, barlow_ind:int=0):
    """get barlow transform matrices from local node to a barlow pt on either xi or eta const lines"""
    inode, jnode = node % 3, node // 3
    xi, eta = inode - 1, jnode - 1

    irt3 = 1.0 / np.sqrt(3)
    barlow_pt = irt3 if barlow_ind == 1 else -irt3

    if direc == 0: # xi then change xi for barlow pts
        xi_b, eta_b = barlow_pt, eta
    else:
        xi_b, eta_b = xi, barlow_pt

    # NOTE: this is why it's not a geometrically exact shell theory.. TBD on that I guess vs TACS shells..
    
    # get cov basis at barlow and local nodal
    a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)
    a_xi_b, a_eta_b, a_gam_b, _, _ = get_xpts_basis(xi_b, eta_b, xpts)

    # get transform matrix for barlow point now
    T = np.array([
        [np.dot(a_xi, a_xi_b), np.dot(a_eta, a_xi_b), np.dot(a_gam, a_xi_b)],
        [np.dot(a_xi, a_eta_b), np.dot(a_eta, a_eta_b), np.dot(a_gam, a_eta_b)],
        [np.dot(a_xi, a_gam_b), np.dot(a_eta, a_gam_b), np.dot(a_gam, a_gam_b)],
    ])

    return T

def get_barlow_1d_interp_vecs(xi, primed:bool=False):
    """get the c_l, c_r barlow interps for strains, and primed if need be for curvature correction"""
    
    irt3 = 1.0 / np.sqrt(3)
    rt3 = np.sqrt(3)
    
    c_left = np.array([
        -(0.5 + irt3) + (1.0 + 0.5 * rt3) * xi,
        -2.0 * (xi - irt3),
        (0.5 - irt3) + (1.0 - rt3 / 2.0) * xi
    ])

    c_right = np.array([
        -(0.5 - irt3) + (1.0 - 0.5 * rt3) * xi,
        -2.0 * (xi + irt3),
        (0.5 + irt3) + (1.0 + rt3 / 2.0) * xi
    ])

    if not primed:
        return c_left, c_right
    else:
        c_left_p = np.array([
            4 * c_left[0] + c_left[1] - 2 * c_left[2],
            4.0 * (c_left[0] + c_left[1] + c_left[2]),
            -2 * c_left[0] + c_left[1] + 4 * c_left[2]
        ]) / 6.0

        c_right_p = np.array([
            4 * c_right[0] + c_right[1] - 2 * c_right[2],
            4.0 * (c_right[0] + c_right[1] + c_right[2]),
            -2 * c_right[0] + c_right[1] + 4 * c_right[2]
        ]) / 6.0

        return c_left_p, c_right_p


def get_curved_beam_length(perp_ind, xpts, direc:int=0):
    """TODO : supposed to do better approx than just ds^2 between each pair of nodes? (use quadratic interp?)"""
    # I'm just doing the ds^2 for xpts_line here..
    # perp_ind is xi or eta perp to the current line
    node = 3*perp_ind if direc == 0 else perp_ind
    if direc == 0:
        xyz_list = [xpts[3*_node:(3*_node+3)] for _node in [node, node+1, node+2]]
    else: # direc == 1
        xyz_list = [xpts[3*_node:(3*_node+3)] for _node in [node, node+3, node+6]]
    
    # now with these three points add up ds for each set
    dr_1 = xyz_list[1] - xyz_list[0]
    dr_2 = xyz_list[2] - xyz_list[1]
    ds_1, ds_2 = np.linalg.norm(dr_1), np.linalg.norm(dr_2)
    return ds_1 + ds_2

def collect_xi_vals(eta, xpts, vars_cov, prime_arr:np.ndarray, out_direc:int, rots:bool=False):
    """collect strain xi vals so you can then do N(xi) * [bj_eta * vars_j] (where [cdot] part is collected)"""
    xi_vals = [0.0]*3
    for ixi in range(3):
        ds_eta = get_curved_beam_length(ixi, xpts, direc=1)
        c_left, c_right = get_barlow_1d_interp_vecs(eta, primed=False)
        c_left_p, c_right_p = get_barlow_1d_interp_vecs(eta, primed=True)

        for ieta in range(3):
            node = 3 * ieta + ixi

            # NOTE : could make this more efficient later.. (less re-computing)
            T_left = get_barlow_transform(node, xpts, direc=1, barlow_ind=0)
            T_right = get_barlow_transform(node, xpts, direc=1, barlow_ind=1)

            loc_vars = np.array([vars_cov[5*node+3], vars_cov[5*node+4], 0.0]) if rots else vars_cov[5*node:(5*node+3)]
            cl = np.array([c_left_p[ieta] if prime_arr[_i] else c_left[ieta] for _i in range(3)])
            cr = np.array([c_right_p[ieta] if prime_arr[_i] else c_right[ieta] for _i in range(3)])

            b_vec = (cl * T_left[out_direc,:] + cr * T_right[out_direc,:]) / ds_eta
            xi_vals[ieta] += np.dot(b_vec, loc_vars)
    return xi_vals

def collect_xi_vals_transpose(eta, xpts, xi_vals_bar, prime_arr:np.ndarray, out_direc:int, rots:bool=False):
    """collect strain xi vals so you can then do N(xi) * [bj_eta * vars_j] (where [cdot] part is collected)"""
    vars_cov_bar = np.zeros(45)
    for ixi in range(3):
        ds_eta = get_curved_beam_length(ixi, xpts, direc=1)
        c_left, c_right = get_barlow_1d_interp_vecs(eta, primed=False)
        c_left_p, c_right_p = get_barlow_1d_interp_vecs(eta, primed=True)

        for ieta in range(3):
            node = 3 * ieta + ixi

            # NOTE : could make this more efficient later.. (less re-computing)
            T_left = get_barlow_transform(node, xpts, direc=1, barlow_ind=0)
            T_right = get_barlow_transform(node, xpts, direc=1, barlow_ind=1)
            cl = np.array([c_left_p[ieta] if prime_arr[_i] else c_left[ieta] for _i in range(3)])
            cr = np.array([c_right_p[ieta] if prime_arr[_i] else c_right[ieta] for _i in range(3)])

            b_vec = (cl * T_left[out_direc,:] + cr * T_right[out_direc,:]) / ds_eta
            if rots:
                vars_cov_bar[5*node+3], vars_cov_bar[5*node+4] += b_vec[0] * xi_vals_bar[ieta], b_vec[1] * xi_vals_bar[ieta]
            else:
                vars_cov_bar[5*node:(5*node+3)] += xi_vals_bar[ieta] * b_vec
    return vars_cov_bar

def collect_eta_vals(xi, xpts, vars_cov, prime_arr:np.ndarray, out_direc:int, rots:bool=False):
    """collect eta vals so you can then do N(eta) * [bi_xi * vars_i] (where [cdot] part is collected)"""

    e_xi_etavals = [0.0]*3
    for ieta in range(3):
        ds_xi = get_curved_beam_length(ieta, xpts, direc=0)
        c_left, c_right = get_barlow_1d_interp_vecs(xi, primed=False)
        c_left_p, c_right_p = get_barlow_1d_interp_vecs(xi, primed=True)

        for ixi in range(3):
            node = 3 * ieta + ixi

            # NOTE : could make this more efficient later.. (less re-computing)
            T_left = get_barlow_transform(node, xpts, direc=0, barlow_ind=0)
            T_right = get_barlow_transform(node, xpts, direc=0, barlow_ind=1)

            loc_vars = np.array([vars_cov[5*node+3], vars_cov[5*node+4], 0.0]) if rots else vars_cov[5*node:(5*node+3)]
            cl = np.array([c_left_p[ixi] if prime_arr[_i] else c_left[ixi] for _i in range(3)])
            cr = np.array([c_right_p[ixi] if prime_arr[_i] else c_right[ixi] for _i in range(3)])

            b_vec = (cl * T_left[out_direc,:] + cr * T_right[out_direc,:]) / ds_xi
            e_xi_etavals[ieta] += np.dot(b_vec, loc_vars)
    return e_xi_etavals

def collect_eta_vals_transpose(xi, xpts, eta_vals_bar, prime_arr:np.ndarray, out_direc:int, rots:bool=False):
    """collect eta vals so you can then do N(eta) * [bi_xi * vars_i] (where [cdot] part is collected)"""

    vars_cov_bar = np.zeros(45)
    for ieta in range(3):
        ds_xi = get_curved_beam_length(ieta, xpts, direc=0)
        c_left, c_right = get_barlow_1d_interp_vecs(xi, primed=False)
        c_left_p, c_right_p = get_barlow_1d_interp_vecs(xi, primed=True)

        for ixi in range(3):
            node = 3 * ieta + ixi

            # NOTE : could make this more efficient later.. (less re-computing)
            T_left = get_barlow_transform(node, xpts, direc=0, barlow_ind=0)
            T_right = get_barlow_transform(node, xpts, direc=0, barlow_ind=1)

            cl = np.array([c_left_p[ixi] if prime_arr[_i] else c_left[ixi] for _i in range(3)])
            cr = np.array([c_right_p[ixi] if prime_arr[_i] else c_right[ixi] for _i in range(3)])

            b_vec = (cl * T_left[out_direc,:] + cr * T_right[out_direc,:]) / ds_xi
            if rots:
                vars_cov_bar[5*node+3], vars_cov_bar[5*node+4] += b_vec[0] * eta_vals_bar[ieta], b_vec[1] * eta_vals_bar[ieta]
            else:
                vars_cov_bar[5*node:(5*node+3)] += eta_vals_bar[ieta] * b_vec
    return vars_cov_bar

def get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts):
    """for rotating xi,eta strains => xyz strains (membrane and bending)"""

    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    T2 = np.array([
        [T[0,0]**2, T[0,1]**2, T[0,0] * T[0,1]],
        [T[1,0]**2, T[1,1]**2, T[1,0] * T[1,1]],
        [2 * T[0,1] * T[1,0], 2 * T[0,1] * T[1,1], T[0,0] * T[1,1] + T[0,1] * T[1,0]]
    ])
    return T2

def get_membrane_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    mem_strains = np.zeros(3)

    # vars_cov is already the local nodal cov rotated bases
    e11_etavals = collect_eta_vals(xi, xpts, vars_cov, prime_arr=[0,0,1], out_direc=0, rots=False)
    mem_strains[0] = lagrange_basis_1d(eta, e11_etavals)

    # now get the e_eta,eta strains
    e22_xivals = collect_xi_vals(eta, xpts, vars_cov, prime_arr=[0,0,1], out_direc=1, rots=False)
    mem_strains[1] = lagrange_basis_1d(xi, e22_xivals)

    # now do the e_{xi,eta} in-plane 
    e12_etavals = collect_eta_vals(xi, xpts, vars_cov, prime_arr=[0,0,1], out_direc=1, rots=False)
    mem_strains[2] = modified_lbasis_1d(eta, e12_etavals)

    e12_xivals = collect_xi_vals(eta, xpts, vars_cov, prime_arr=[0,0,1], out_direc=0, rots=False)
    mem_strains[2] += modified_lbasis_1d(xi, e12_xivals)   

    # rotate membrane strains to local cartesian from nat cov basis
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    mem_strains_xyz = np.dot(T2, mem_strains)

    return mem_strains_xyz

def get_membrane_strains_transpose(shell_xi_axis, xi, eta, xpts, mem_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    mem_strains_bar = np.doT(T2.T, mem_strains_xyz_bar)

    # vars_cov is already the local nodal cov rotated bases
    e11_eta_bar = lagrange_basis_1d_transpose(eta, mem_strains_bar[0])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, e11_eta_bar, prime_arr=[0,0,1], out_direc=0, rots=False)

    # now get the e_eta,eta strains
    e22_xi_bar = lagrange_basis_1d_transpose(xi, mem_strains_bar[1])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, e22_xi_bar, prime_arr=[0,0,1], out_direc=1, rots=False)

    # now do the e_{xi,eta} in-plane 
    e12_eta_bar = modified_lbasis_1d_transpose(eta, mem_strains_bar[2])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, e12_eta_bar, prime_arr=[0,0,1], out_direc=1, rots=False)
    e12_xi_bar = modified_lbasis_1d_transpose(xi, mem_strains_bar[2])
    vars_cov_bar += collect_eta_vals_transpose(eta, xpts, e12_xi_bar, prime_arr=[0,0,1], out_direc=0, rots=False)
    
    return vars_cov_bar

def get_bending_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    bend_strains = np.zeros(3)

    # vars_cov is already the local nodal cov rotated bases
    k11_etavals = collect_eta_vals(xi, xpts, vars_cov, prime_arr=[0,0,1], out_direc=0, rots=True)
    bend_strains[0] = lagrange_basis_1d(eta, k11_etavals)

    # now get the e_eta,eta strains
    k22_xivals = collect_xi_vals(eta, xpts, vars_cov, prime_arr=[0,0,1], out_direc=1, rots=True)
    bend_strains[1] = lagrange_basis_1d(xi, k22_xivals)

    # now do the e_{xi,eta} in-plane 
    k12_etavals = collect_eta_vals(xi, xpts, vars_cov, prime_arr=[0,0,1], out_direc=1, rots=True)
    bend_strains[2] = modified_lbasis_1d(eta, k12_etavals)

    k12_xivals = collect_xi_vals(eta, xpts, vars_cov, prime_arr=[0,0,1], out_direc=0, rots=True)
    bend_strains[2] += modified_lbasis_1d(xi, k12_xivals)   

    # rotate bending strains to local cartesian from nat cov basis
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    bend_strains_xyz = np.dot(T2, bend_strains)
    return bend_strains_xyz

def get_bending_strains_transpose(shell_xi_axis, xi, eta, xpts, bend_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    bend_strains_bar = np.doT(T2.T, bend_strains_xyz_bar)

    # vars_cov is already the local nodal cov rotated bases
    k11_eta_bar = lagrange_basis_1d_transpose(eta, bend_strains_bar[0])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, k11_eta_bar, prime_arr=[0,0,1], out_direc=0, rots=True)

    # now get the e_eta,eta strains
    k22_xi_bar = lagrange_basis_1d_transpose(xi, bend_strains_bar[1])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, k22_xi_bar, prime_arr=[0,0,1], out_direc=1, rots=True)

    # now do the e_{xi,eta} in-plane 
    k12_eta_bar = modified_lbasis_1d_transpose(eta, bend_strains_bar[2])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, k12_eta_bar, prime_arr=[0,0,1], out_direc=1, rots=True)
    k12_xi_bar = modified_lbasis_1d_transpose(xi, bend_strains_bar[2])
    vars_cov_bar += collect_eta_vals_transpose(eta, xpts, k12_xi_bar, prime_arr=[0,0,1], out_direc=0, rots=True)
    return vars_cov_bar

def get_trv_shear_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    trv_shear_strains = np.zeros(3)

    # TODO : is it supposed to be out_direc = 2 for the first term in each trv_shear_strain?

    # vars_cov is already the local nodal cov rotated bases
    gam13_etavals = collect_eta_vals(xi, xpts, vars_cov, prime_arr=[1,1,0], out_direc=2, rots=False)
    trv_shear_strains[0] = lagrange_basis_1d(eta, gam13_etavals)
    # then do rots 
    alpha_vals = vars_cov[3::5]
    alpha_etas_red = [modified_lbasis_1d(xi,alpha_vals[3*i:(3*i+3)]) for i in range(3)]
    trv_shear_strains[0] += lagrange_basis_1d(eta, alpha_etas_red)

    # now the gam23 strain
    gam23_xivals = collect_xi_vals(eta, xpts, vars_cov, prime_arr=[1,1,0], out_direc=2, rots=False)
    trv_shear_strains[1] = lagrange_basis_1d(xi, gam23_xivals)
    # then do rots 
    beta_vals = vars_cov[4::5]
    beta_etas_red = [lagrange_basis_1d(xi,beta_vals[3*i:(3*i+3)]) for i in range(3)]
    trv_shear_strains[1] += modified_lbasis_1d(eta, beta_etas_red)

    # rotate membrane strains to local cartesian from nat cov basis
    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    trv_shear_strains_xyz = np.dot(T[:2,:][:,:2], trv_shear_strains)

    return trv_shear_strains_xyz

def get_trv_shear_strains_transpose(shell_xi_axis, xi, eta, xpts, trv_shear_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    T2 = T[:2,:][:,:2]
    trv_shear_strains_bar = np.doT(T2.T, trv_shear_strains_xyz_bar)

    # vars_cov is already the local nodal cov rotated bases
    gam13_eta_bar = lagrange_basis_1d_transpose(eta, trv_shear_strains_bar[0])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, gam13_eta_bar, prime_arr=[1,1,0], out_direc=2, rots=False)
    for i in range(3):
        nodes = np.array([3*i+j for j in range(3)])
        alpha_dof = 5 * nodes + 3
        vars_cov_bar[alpha_dof] += modified_lbasis_1d_transpose(xi, gam13_eta_bar)

    # now get the e_eta,eta strains
    gam23_xi_bar = lagrange_basis_1d_transpose(xi, trv_shear_strains_bar[1])
    vars_cov_bar += collect_eta_vals_transpose(xi, xpts, gam23_xi_bar, prime_arr=[1,1,0], out_direc=2, rots=False)
    gam23_xi_mod_bar = modified_lbasis_1d_transpose(xi, trv_shear_strains_bar[1])
    for i in range(3):
        nodes = np.array([3*i+j for j in range(3)])
        beta_dof = 5 * nodes + 4
        vars_cov_bar[beta_dof] += lagrange_basis_1d_transpose(xi, gam23_xi_mod_bar)
    return vars_cov_bar

def get_quadpt_stresses(shell_xi_axis, xi, eta, xpts, vars, E:float, nu:float, thick:float):
    """get quadpt stresses (important step in the kelem formulation)"""

    vars_cov = get_cov_nodal_disps(shell_xi_axis, xpts, vars)

    quadpt_strains = np.zeros(8)
    quadpt_strains[:3] = get_membrane_strains(shell_xi_axis, xi, eta, xpts, vars_cov)
    quadpt_strains[3:6] = get_bending_strains(shell_xi_axis, xi, eta, xpts, vars_cov)
    quadpt_strains[6:] = get_trv_shear_strains(shell_xi_axis, xi, eta, xpts, vars_cov)

    # now get A,D,As matrix (for metal)
    C = E / (1.0 - nu**2) * np.array([
        [1, nu, 0], [nu, 1, 0], [0, 0, 0.5 * (1 - nu)],
    ])
    A = C * thick
    D = C * thick**3 / 12.0
    ks = 5.0 / 6.0 # shear correction factor
    As = ks * C[-1,-1] * np.diag([1.0, 1.0])

    # now get the shell stress resultants
    N = np.dot(A, quadpt_strains[:3])
    M = np.dot(D, quadpt_strains[3:6])
    Q = np.dot(As, quadpt_strains[6:])

    loads = np.concatenate([N, M, Q], axis=0)

    return loads

def get_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain):
    """transpose operation of strain sens back to disps"""

    vars_cov_bar = get_membrane_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[:3])
    vars_cov_bar += get_bending_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[3:6])
    vars_cov_bar += get_trv_shear_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[6:])

    vars_bar = get_cov_nodal_disps_transpose(shell_xi_axis, xpts, vars_cov_bar)
    return vars_bar


def get_quadpt_kelem(shell_xi_axis, xi, eta, xpts, vars, E:float, nu:float, thick:float):
    """get linear kelem stiffness matrix"""

    N = vars.shape[0]
    Kelem = np.zeros((N,N))

    # TODO : need to loop over each of 45 input strains to do hess-vec prod method..
    for i in range(N):
        p_vars = np.zeros_like(vars)
        p_vars[i] = 1.0
        quadpt_stresses = get_quadpt_stresses(shell_xi_axis, xi, eta, xpts, vars, E, nu, thick)

        vars_bar = get_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain=quadpt_stresses)
        Kelem[:,i] = vars_bar

    return Kelem

def get_kelem(shell_xi_axis, xpts, vars, E:float, nu:float, thick:float):
    """get full kelem by looping over quadpts and weights"""

    rt_35 = np.sqrt(3.0 / 5.0)
    pts = np.array([-rt_35, 0.0, rt_35])
    weights = np.array([5/9, 8/9, 5/9])

    full_Kelem = np.zeros((45,45))

    for ixi in range(3):
        xi, wt_xi = pts[ixi], weights[ixi]

        for ieta in range(3):
            eta, wt_eta = pts[ieta], weights[ieta]
            weight = wt_xi * wt_eta

            a_xi, a_eta, _, _, _ = get_xpts_basis(xi, eta, xpts)
            dxy_jac = np.linalg.norm(np.cross(a_xi, a_eta)) # dS/dxi/deta

            _kelem = get_quadpt_kelem(shell_xi_axis, xi, eta, xpts, vars, E, nu, thick)
            full_Kelem += _kelem * weight * dxy_jac

    return full_Kelem
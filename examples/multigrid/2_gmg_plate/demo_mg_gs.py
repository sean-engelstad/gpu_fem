from src._dkt_plate_elem import *
from src.linalg import *
import numpy as np
import matplotlib.pyplot as plt

"""demo multigrid with block gauss-seidel"""

"""compute and solve the fine problem (ref soln)"""

# compute the element stiffness matrix for an aluminum material
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick

# let's now try and assemble the global stiffness matrix (dense form)
# nxe = 64
nxe = 32

K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = K.shape[0]

# now apply some loads to the structure
F = np.zeros(ndof)
F[0::3] = 100.0 # vert load for bending..

K, F = apply_bcs(nxe, K, F)
nx = nxe + 1

# true soln
u = np.linalg.solve(K, F)

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

"""define a convenient plot method"""
# plot results of random defect schwarz smoothing
def plot_defect_change(_nx, _defect, __du, _defect_2, idof:int=0, all:bool=False):
    """plot the defect change"""

    x_plt, y_plt = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _nx)
    X, Y = np.meshgrid(x_plt, y_plt)
    Load = np.reshape(_defect[idof::3], (_nx, _nx))
    Disp_hat = np.reshape(__du[idof::3], (_nx, _nx))
    Load_hat = np.reshape(_defect_2[idof::3], (_nx, _nx))

    if all:
        idof_list = [0,1,2]
    else:
        idof_list = [idof]

    for idof in idof_list:
        fig, ax = plt.subplots(1, 3)
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(X, Y, Load)
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(X, Y, Disp_hat)
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(X, Y, Load_hat)

        dof_str = ['w', 'thx', 'thy'][idof]
        ax0.set_title(f"Defect {dof_str}")
        ax1.set_title(f"Update {dof_str}")
        ax2.set_title(f"Defect v2")
        plt.show()


"""TODO : now try solving the multigrid with two-levels.. and different smoothers"""

K_f = assemble_stiffness_matrix(nxe, E, thick, nu)
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
K_f, _ = apply_bcs(nxe, K_f, F)
K_c, _ = apply_bcs(nxe_c, K_c, np.zeros(K_c.shape[0]))

K_f_csr = sp.csr_matrix(K_f)

P = prolongation_operator(nxe_c)
R = P.T # restriction operator

defect_0 = F.copy()
u_f_0 = np.zeros(ndof)
defect, u_f = defect_0.copy(), u_f_0.copy()

print(f"{defect_0=}")

# number of pre and post-smoothing steps to do
# n_pres, n_cfs, n_posts = 1, 1, 1
n_pres, n_cfs, n_posts = 3, 3, 3

init_norm = np.linalg.norm(defect)
print(f"{init_norm=:.4e}")

# can_plot = True
can_plot = False

for v_cycle in range(10):
    """do v-cycle loops here"""

    # step 1 - pre-smoothing at fine level
    _du = block_gauss_seidel(K_f_csr, defect, x0=np.zeros(ndof), num_iter=n_pres)
    defect_2 = defect - K_f_csr.dot(_du)
    u_f_2 = u_f + _du # TODOP plus or minus update here?
    if can_plot:
        print(f"{v_cycle=} : 1 - pre-smooth")
        plot_defect_change(nx, defect, u_f_2, defect_2)

    # step 2 - restrict defect to coarse
    defect_c_1 = np.dot(R, defect_2)
    # plot_defect_change(nx_c, defect_c_1, defect_c_1, defect_c_1)
    coarse_bcs = get_bcs(nxe_c)

    # step 3 - coarse solve
    du_c_1 = np.linalg.solve(K_c, defect_c_1)
    # plot_defect_change(nx_c, defect_c_1, du_c_1, defect_c_1)

    # step 4 - coarse-fine or prolongation
    du_f_1 = np.dot(P, du_c_1)
    _dF_f_1 = K_f_csr.dot(du_f_1)
    # plot_defect_change(nx, defect_2, du_f_1, _dF_f_1)

    # gauss-seidel smoothing on coarse-fine update itself
    _du_cf = block_gauss_seidel(K_f_csr, _dF_f_1, x0=np.zeros(ndof), num_iter=n_cfs)
    du_f_2 = du_f_1 - _du_cf
    _dF_f_2 = K_f_csr.dot(du_f_2)
    if can_plot:
        print(f"{v_cycle=} : 2 - prolong smooth")
        plot_defect_change(nx, _dF_f_1, du_f_2, _dF_f_2)

    # exit()

    # compare disps of the smoothing.. (NOTE : they do have about the same magnitude)
    # plot_defect_change(nx, _dF_f_2, du_f_1, du_f_2)

    # step 5 - apply disp update using line search
    # TODO : do two-parameter updates here (two-dof line search, see paper on unstructured multigrid methods for shells)
    # s = du_f_1
    s = du_f_2
    omega = np.dot(s, defect_2) / np.dot(s, K_f_csr.dot(s))
    # defect_3 = defect_2 - K_f_csr.dot(omega * s)
    defect_3 = defect_2 - omega * _dF_f_2
    u_f_3 = u_f_2 + omega * s
    if can_plot:
        print(f"{v_cycle=} : 3 - prolong update, {omega=}")
        plot_defect_change(nx, defect_2, u_f_3, defect_3)

    # step 6 - post-smoothing with Gauss-seidel
    _du_2 = block_gauss_seidel(K_f_csr, defect_3, x0=np.zeros(ndof), num_iter=n_posts)
    defect_4 = defect_3 - K_f_csr.dot(_du_2)
    u_f_4 = u_f_3 + _du # TODOP plus or minus update here?
    if can_plot:
        print(f"{v_cycle=} : 4 - postsmooth")
        plot_defect_change(nx, defect_3, u_f_4, defect_4)

    # step 7 - reset for next step
    defect, u_f = defect_4, u_f_4

    c_norm = np.linalg.norm(defect)
    print(f"v-cycle step {v_cycle}, {c_norm=:.4e}")

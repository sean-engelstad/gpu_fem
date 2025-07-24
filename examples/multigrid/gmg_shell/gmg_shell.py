"""
geometric multigrid for shells
"""
import numpy as np
import matplotlib.pyplot as plt
from _utils import get_tacs_matrix, delete_rows_and_columns, reduced_indices, plot_vec_compare, plot_plate_vec
from _utils import gauss_seidel_csr, block_gauss_seidel_6dof, mg_coarse_fine_operators, sort_vis_maps, zero_non_nodal_dof
import scipy as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

# get TACS matrix and convert to CSR since pyAMG only supports CSR
# (this may mean it is slower or less accurate than a BSR version)
# -----------------------------------
tacs_csr_mat_list = []
rhs_list = []
sort_fw_map_list = []
sort_bk_map_list = []

# normally you remove dirichlet bcs in multigrid
# but you don't have to if you're careful (and it will be tricky to remove bcs for block Gauss-seidel smoothing, TBD)
remove_bcs = True

nxe_list = [16, 8, 4]

for nxe in nxe_list:
    _tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"plate{nxe}.bdf")
    _tacs_csr_mat = _tacs_bsr_mat.tocsr()

    _nnodes = _xpts.shape[0] // 3
    _N = 6 * _nnodes

    # remove bcs..
    if remove_bcs:
        _free_dof = reduced_indices(_tacs_csr_mat)
        bcs = [_ for _ in range(_tacs_csr_mat.shape[0]) if not(_ in _free_dof)]
        _tacs_csr_mat = delete_rows_and_columns(_tacs_csr_mat, dof=_free_dof)
        _rhs = _rhs[_free_dof]
    
    else: # not removing bcs
        _free_dof = [_ for _ in range(_N)]
    
    # get the sort maps (to undo any reordering)
    _sort_fw, _sort_bk = sort_vis_maps(nxe, _xpts, _free_dof)
    sort_fw_map_list += [_sort_fw]
    sort_bk_map_list += [_sort_bk]
    
    # put in hierarchy list
    tacs_csr_mat_list += [_tacs_csr_mat.copy()]
    rhs_list += [_rhs.copy()]

"""plot and solve the original res error"""
mat, rhs, sort_fw, sort_bk = tacs_csr_mat_list[0], rhs_list[0], sort_fw_map_list[0], sort_bk_map_list[0]
nfree = sort_bk.shape[0]
nnodes = nfree // 6
nxe, N = int(nnodes**0.5) - 1, rhs.shape[0] # here N is the # DOF after bc removal
soln = spsolve(mat, rhs)

# this shows we get the correct solution in the SS plate case
# plot_vec_compare(nxe, rhs, soln, sort_fw, filename=None)

"""now we try error smoothing on fine grid first"""
# let's work first on the fine grid, see if smoothing even works..
x0, res = np.zeros(N), rhs.copy()
x = x0.copy()
x = gauss_seidel_csr(mat, rhs, x, num_iter=5)
# x = block_gauss_seidel_6dof(mat, rhs, x, num_iter=5)
res2 = res - mat.dot(x)

# see whether residual smoothing happens or not?
# plot_vec_compare(nxe, res, res2, sort_fw, filename=None)

"""compute and try coarse-fine operators (check error shape stays the same)"""
I_cf, I_fc = mg_coarse_fine_operators(nxe, sort_bk_map_list[0], sort_bk_map_list[1])
# plt.imshow(I_fc)
# plt.show()
res_c = np.dot(I_fc, res)
res2_c = np.dot(I_fc, res2)
res_f = np.dot(I_cf, res_c)
res2_f = np.dot(I_cf, res2_c)

# it is working now

# see whether residuals look similar on the coarse mesh
# plot_vec_compare(nxe//2, res_c, res2_c, sort_fw_map_list[1], filename=None)

# # and then back to the fine mesh
# plot_vec_compare(nxe, res_f, res2_f, sort_fw, filename=None)

"""now we'll compute the coarse-fine operators at each level"""
Icf_list = []
Ifc_list = []
n_levels = len(rhs_list)
for i in range(n_levels-1):
    _I_cf, _I_fc = mg_coarse_fine_operators(nxe_list[i], sort_bk_map_list[i], sort_bk_map_list[i+1])
    Icf_list += [_I_cf]
    Ifc_list += [_I_fc]

"""now do multigrid V-cycle"""

# for TEMP DEBUG (this is very important input though, only when False does it operate normally..)
# zero_non_w_dof = True
zero_non_w_dof = False

# num of gauss-seidel smoothing steps (fw and backwards)
gs_fw = 10
# gs_fw = 3
gs_bk = 10
# gs_bk = 3

# now let's implement a V cycle demo (based on 1_poisson/vcycle_demo.py) 
x0 = np.zeros(rhs.shape[0])
x = x0.copy()
defect = rhs.copy() # beginning defect
last_res_norm = np.linalg.norm(defect)
init_res_norm = last_res_norm
import os

last_ovr_defect = defect.copy()

for i_vcycle in range(10):

    folder = f"out/vcyc_{i_vcycle}"
    if not os.path.exists(folder):
        os.mkdir(folder)

    x_update_list = []
    defect_list = []
    c_nxe = nxe
    ct = -1
    for i_level in range(n_levels):
        ct += 1

        # get operators for this level (and to level below)
        lhs = tacs_csr_mat_list[i_level]

        if i_level < n_levels - 1: # then smoothing
            # pre-smooth solution
            _x0 = np.zeros(lhs.shape[0])
            x_update = gauss_seidel_csr(lhs, defect, _x0, num_iter=gs_fw)

            # restrict the defect (rhs error or current resid) to the coarse grid
            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

            # TEMP DEBUG: zero out all error outside the w DOF
            if zero_non_w_dof:
                _defect = zero_non_nodal_dof(_defect, sort_fw_map_list[i_level], inodal=2)

            # coarsen here
            Ifc = Ifc_list[i_level]
            defect = np.dot(Ifc, _defect)

        else: # if last level just do direct solve (v. small or coarsest level)
            x_update = spsolve(lhs, defect)

            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

        # print(f"coarse to fine defects for {i_level=} / [0-3]")
        c_nxe = nxe_list[i_level]
        
        plot_vec_compare(c_nxe, old_defect, _defect, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_down{i_level}_smooth_.svg", nodal_dof=2)

        x_update_list += [x_update.copy()] # save the updates from pre-smoothing for later   
        # defect gets passed to the next level

    # print("----------------------\n")
    # print("coarse back to fine part of V-cycle")
    # print("----------------------\n")

    # now we iterate back up the hierarchy
    _coarse_update = x_update_list[-1]
    for i_level in range(n_levels-2, -1, -1):
        ct += 1
        # get operators for this level (and to level below)
        lhs = tacs_csr_mat_list[i_level]
        Icf = Icf_list[i_level]

        # interpolate the correction
        # print(f"{_coarse_update.shape=} {Icf.shape=}")
        cf_update = np.dot(Icf, _coarse_update) 
        # add correction from this level to coarser update
        fine_update = x_update_list[i_level] + cf_update

        defect = defect_list[i_level].copy()
        start_defect = defect - lhs @ cf_update

        # TEMP DEBUG: zero out all error outside the w DOF
        if zero_non_w_dof:
            defect = zero_non_nodal_dof(defect, sort_fw_map_list[i_level], inodal=2)
            start_defect = zero_non_nodal_dof(start_defect, sort_fw_map_list[i_level], inodal=2)

        # do a post-smoothing and update the update at this level
        _x0 = np.zeros(lhs.shape[0])
        _update = gauss_seidel_csr(lhs, start_defect, _x0, num_iter=gs_bk)
        x_update = _update + fine_update

        new_defect = start_defect - lhs @ _update
        c_nxe = nxe // 2**i_level # debug
        plot_vec_compare(c_nxe, defect, start_defect, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_up{i_level}_interp.svg")
        ct += 1
        plot_vec_compare(c_nxe, start_defect, new_defect, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_up{i_level}_smooth.svg")

        # new _coarse_update is then: to next level
        _coarse_update = x_update.copy()

    # check final error after V-cycle
    x += x_update
    defect = rhs - mat.dot(x)
    res_norm = np.linalg.norm(defect)
    print(f"vcyc{i_vcycle}: {last_res_norm=:.3e} => {res_norm=:.3e}")
    plot_vec_compare(c_nxe, last_ovr_defect.copy(), defect.copy(), sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/_ovr.svg")

    last_ovr_defect = defect.copy()
    last_res_norm = res_norm

    plt.close('all')


# now check the solution against the true soln
print("final multigrid results:\n")
final_res = rhs - mat.dot(x)
final_res_norm = np.linalg.norm(final_res)
print(f"{init_res_norm=:.3e} => {final_res_norm=:.3e}")
plot_vec_compare(c_nxe, final_res.copy(), x.copy(), sort_fw_map=sort_fw_map_list[0], 
                         filename=f"out/_ovr.svg")
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

def get_soln():
    soln = read_from_csv("csv/plate_soln.csv")
    num_nodes = soln.shape[0]//6
    nxe = int(num_nodes**0.5 - 1)
    return soln, num_nodes, nxe

def build_mesh():
    nye = nxe; Lx = 2.0; Ly = 1.0
    x = np.linspace(0.0, Lx, nxe+1)
    y = np.linspace(0.0, Ly, nye+1)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_vec(vec, filename, dof = 2):
    dof_vec = vec[dof::6] # only show w disps in plot
    VALS = np.reshape(dof_vec, newshape=X.shape)
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, VALS, cmap='viridis', edgecolor='none')

    # Add labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('SOLN (Z-axis)')
    ax.set_title('3D Surface Plot')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # ax.set_zscale('symlog')
    plt.savefig(filename, dpi=400)

def get_kelem_mat():

    # Example usage
    kelem_vec = read_from_csv("csv/kelem.csv")
    kelem_mat = np.zeros((24,24))
    # have to restructure data from block format to dense here
    for iblock in range(16):
        inode = iblock // 4
        jnode = iblock % 4
        for inz in range(36):
            _irow = inz // 6
            _icol = inz % 6
            irow = 6 * inode + _irow
            icol = 6 * jnode + _icol
            kelem_mat[irow,icol] = kelem_vec[36 * iblock + inz]

    # kelem_dense = read_from_csv("csv/kelem-dense.csv")
    # kelem_dense = np.reshape(kelem_dense, newshape=(24,24))

    return kelem_mat #, kelem_dense

def get_python_kmat(kelem_mat, num_elements, num_nodes, nx):

    assembly_csv_df_dict = {
        mystr:[] for mystr in ['ielem', 'grow', 'gcol', 'erow', 'ecol', 'value']
    }    

    bcs = []
    for iy in range(nx):
        for ix in range(nx):
            inode = nx * iy + ix
            if ix == 0 and iy == 0:
                local_bcs = [0, 1, 2, 3, 4, 5]
            elif ix == 0:
                local_bcs = [0, 2]
            elif iy == 0:
                local_bcs = [1, 2]
            elif ix == nx - 1 or iy == nx - 1:
                local_bcs = [2]
            else:
                local_bcs = []
            bcs += [6*inode + idof for idof in local_bcs]

    kmat_py = np.zeros((6*num_nodes, 6*num_nodes))
    for ielem in range(num_elements):
        ix = ielem // (nx-1)
        iy = ielem % (nx-1)
        inode = nx * ix + iy
        # local_node_conn = [inode, inode + 1, inode + nx + 1, inode + nx]
        # due to issue in elem_conn ?
        local_node_conn = [inode, inode + 1, inode + nx, inode + nx + 1]
        # print(local_node_conn)
        local_dof_conn = [[6*inode + idof for idof in range(6)] for inode in local_node_conn]
        local_dof_conn = np.array(local_dof_conn).flatten()

        for erow,grow in enumerate(local_dof_conn):
            for ecol,gcol in enumerate(local_dof_conn):
                if grow in bcs:
                    val = 1.0 if grow == gcol else 0.0
                else:
                    val = kelem_mat[erow,ecol]
                assembly_csv_df_dict['ielem'] += [ielem]
                assembly_csv_df_dict['grow'] += [grow]
                assembly_csv_df_dict['gcol'] += [gcol]
                assembly_csv_df_dict['erow'] += [erow]
                assembly_csv_df_dict['ecol'] += [ecol]
                assembly_csv_df_dict['value'] += [val]

        # now add kelem into this as dense matrix
        global_ind = np.ix_(local_dof_conn, local_dof_conn)
        kmat_py[global_ind] += kelem_mat
        
    kmat_py[kmat_py == 0] = 1e-8
    for bc in bcs:
        kmat_py[bc,:] = 1e-8
        # kmat_mat2[:,bc] = 1e-8
        kmat_py[bc,bc] = 1.0

    # write assembly csv to csv file
    assembly_csv_df = pd.DataFrame(assembly_csv_df_dict)
    assembly_csv_df.to_csv(f"elems{num_elements}/python-kmat-assembly.csv", float_format='%.4e', index=False)


    return kmat_py

def get_loads():
    loads = read_from_csv("csv/plate_loads.csv")
    loads = np.reshape(loads, newshape=(loads.shape[0], 1))
    return loads

def get_analytic_mag():
    E = 70e9; nu = 0.3; thick = 0.005
    a = 2.0; b = 1.0
    D = E * thick**3 / 12.0 / (1 - nu**2)
    print(f"{D=}")
    Q = 1.0
    pi = np.pi
    denom1 = a**(-4) + b**(-4) + 2.0 * a**(-2) * b**(-2)
    analytic_magnitude = Q / D / pi**4 / denom1
    # print(f"{analytic_magnitude=}")
    return analytic_magnitude

def get_cpp_kmat():
    kmat_vec = read_from_csv("csv/plate_kmat.csv")
    # kmat_vec = read_from_csv("csv/plate_kmat_nobcs.csv")
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)

    print(f"{kmat_vec.shape[0]}")
    _nnodes = rowPtr.shape[0] - 1
    nx = int(_nnodes**0.5)
    nxe = nx - 1
    nnodes = nx**2
    num_elements = nxe**2
    kmat_cpp = np.zeros((nnodes*6,nnodes*6))

    if not os.path.exists(f"elems{num_elements}"):
        os.mkdir(f"elems{num_elements}")

    for block_row in range(nnodes):
        istart = rowPtr[block_row]
        iend = rowPtr[block_row+1]
        for block_ind in range(istart, iend):
            block_col = colPtr[block_ind]
            val_ind = 36 * block_ind
            for inz in range(36):
                _irow = inz // 6
                _icol = inz % 6
                irow = 6 * block_row + _irow
                icol = 6 * block_col + _icol
                bsr_ind = 36 * block_ind + inz
                # if block_row < 2 and block_col < 2:
                    # print(f"{irow=},{icol=} from {bsr_ind=}")
                kmat_cpp[irow,icol] = kmat_vec[36 * block_ind + inz]

    kmat_cpp[np.abs(kmat_cpp) < 1e-8] = 1e-8

    return kmat_cpp, num_elements, nnodes, nx

def print_cpp_val(val_ind):
    kmat_vec = read_from_csv("csv/plate_kmat.csv")
    print(f"kmat_vec{val_ind=} = {kmat_vec[val_ind]}")

def plot_mat(mat, filename):
    plt.figure()
    ax = sns.heatmap(mat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
    plt.savefig(filename, dpi=400)
    return

def get_kmat_rel_err(kmat_cpp, kmat_py):
    return np.where(
        np.abs(kmat_py) > 1e-6,
        np.abs((kmat_cpp - kmat_py) / (1e-8 + kmat_py)),
        np.abs((kmat_cpp - kmat_py) / (1e-8 + kmat_py)) * 1e-8
    )

def write_matrices_to_file(kmat_cpp, kmat_py, kmat_rel_err, filename):
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)
    _nnodes = rowPtr.shape[0] - 1
    nx = int(_nnodes**0.5)
    nnodes = nx**2

    df_dict = {
        mystr:[] for mystr in ['grow', 'gcol','kmat_py', 
                               'kmat_cpp', 'kmat_rel_err']
    }

    for block_row in range(nnodes):
        istart = rowPtr[block_row]
        iend = rowPtr[block_row+1]
        for block_ind in range(istart, iend):
            block_col = colPtr[block_ind]
            val_ind = 36 * block_ind
            for inz in range(36):
                _irow = inz // 6
                _icol = inz % 6
                irow = 6 * block_row + _irow
                icol = 6 * block_col + _icol

                # kmat_cpp[irow,icol] = kmat_vec[36 * block_ind + inz]

                df_dict['grow'] += [irow]
                df_dict['gcol'] += [icol]
                df_dict['kmat_py'] += [kmat_py[irow,icol]]
                df_dict['kmat_cpp'] += [kmat_cpp[irow,icol]]
                df_dict['kmat_rel_err'] += [kmat_rel_err[irow,icol]]

    df = pd.DataFrame(df_dict)
    df.to_csv(filename, index=False, float_format='%.4e')

def write_python_bsr_mat(kmat_py, filename):
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)
    _nnodes = rowPtr.shape[0] - 1
    nx = int(_nnodes**0.5)
    nnodes = nx**2

    df_dict = {
        mystr:[] for mystr in ['grow', 'gcol','kmat_py', 
                               'kmat_cpp', 'kmat_rel_err']
    }

    for block_row in range(nnodes):
        istart = rowPtr[block_row]
        iend = rowPtr[block_row+1]
        for block_ind in range(istart, iend):
            block_col = colPtr[block_ind]
            val_ind = 36 * block_ind
            for inz in range(36):
                _irow = inz // 6
                _icol = inz % 6
                irow = 6 * block_row + _irow
                icol = 6 * block_col + _icol

                # kmat_cpp[irow,icol] = kmat_vec[36 * block_ind + inz]

                df_dict['grow'] += [irow]
                df_dict['gcol'] += [icol]
                df_dict['kmat_py'] += [kmat_py[irow,icol]]

    df = pd.DataFrame(df_dict)
    df.to_csv(filename, index=False, float_format='%.4e')

if __name__ == "__main__":
    soln, num_nodes, nxe = get_soln()
    num_elements = nxe * nxe
    if not os.path.exists(f"elems{num_elements}"):
        os.mkdir(f"elems{num_elements}")
    
    # plot solution from C++
    X, Y = build_mesh()
    folder = f"elems{num_elements}/"
    plot_vec(soln, folder + "cpp-soln.png", dof=2)

    # also plot solution from python
    kelem_mat = get_kelem_mat()
    kmat_py = get_python_kmat(kelem_mat, num_elements, num_nodes, nxe + 1)
    loads = get_loads()

    print_cpp_val(576)

    # optional nugget term
    # theta = 1e-6
    # kmat_mat_py += theta * np.eye(kmat_mat_py.shape[0]) 

    soln_py = np.linalg.solve(kmat_py, loads)
    plot_vec(soln_py, folder + "python-soln.png", dof=2)
    plot_vec(loads, folder + "cpp-loads.png", dof=2)

    # compare to analytic magnitude and solved magnitude
    # analytic_mag = get_analytic_mag()
    # print(f"{analytic_mag=}")

    # loads_check = kmat_py @ soln_py
    # resid = loads_check - loads

    # compare the matrices with heatmaps
    kmat_cpp,_,_,_ = get_cpp_kmat()
    kmat_rel_err = get_kmat_rel_err(kmat_cpp, kmat_py)

    plot_mat(kmat_py, folder + "kmat_py.png")
    plot_mat(kmat_cpp, folder + "kmat_cpp.png")
    plot_mat(kmat_rel_err, folder + "kmat_rel_err.png")

    # write out the matrices to a csv file now
    write_matrices_to_file(kmat_cpp, kmat_py, kmat_rel_err, folder + "kmat-compare.csv")
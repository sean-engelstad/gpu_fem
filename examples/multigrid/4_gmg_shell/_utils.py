import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive
import scipy
import matplotlib.pyplot as plt

def get_tacs_matrix(bdf_file, thickness:float=0.02):

    # Load structural mesh from BDF file
    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdf_file)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    E = 70e9  # elastic modulus, Pa
    nu = 0.3  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.002
    max_thickness = 10.0 # 0.2
    # thickness = 0.02

    # Loop over components, creating stiffness and element object for each
    num_components = struct_mesh.getNumComponents()
    for i in range(num_components):
        descriptor = struct_mesh.getElementDescript(i)
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        # Set one thickness dv for every component
        stiff = constitutive.IsoShellConstitutive(
            prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i
        )

        element = None
        transform = None
        if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.Quad4Shell(transform, stiff)
        struct_mesh.setElement(i, element)

    # Create tacs assembler object from mesh loader
    tacs = struct_mesh.createTACS(6)

    # Set up and solve the analysis problem
    res = tacs.createVec()
    mat = tacs.createSchurMat(TACS.NATURAL_ORDER)

    xpts_vec = tacs.createNodeVec()
    tacs.getNodes(xpts_vec)
    xpts_arr = xpts_vec.getArray()

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray()
    # force_array[2::6] += 100.0  # uniform load in z direction
    x = xpts_arr[0::3]
    y = xpts_arr[0::3]
    r = np.sqrt(x**2 + y**2)
    force_array[2::6] += 100.0 * np.sin(3 * np.pi * x)
    # force_array[2::6] += 100.0 * np.sin(3 * np.pi * r)
    tacs.applyBCs(forces)

    # Assemble the Jacobian and factor
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)
    # tacs.applyBCs(res, mat)

    data = mat.getMat()
    # print(f"{data=}")
    A_bsr = data[0]
    # print(f"{A_bsr=} {type(A_bsr)=}")

    # TODO : just get serial A part in a minute
    return A_bsr, force_array, xpts_arr

def reduced_indices(A):
    # takes in A a csr matrix
    A.eliminate_zeros()

    dof = []
    for k in range(A.shape[0]):
        if (
            A.indptr[k + 1] - A.indptr[k] == 1
            and A.indices[A.indptr[k]] == k
            and np.isclose(A.data[A.indptr[k]], 1.0)
        ):
            # This is a constrained DOF
            pass
        else:
            # Store the free DOF index
            dof.append(k)
    return dof

def delete_rows_and_columns(A, dof=None):
    if dof is None:
        dof = reduced_indices(A)

    iptr = [0]
    cols = []
    data = []

    # inverse map from full dof => red dof
    indices = -np.ones(A.shape[0])
    indices[dof] = np.arange(len(dof))

    for i in dof:
        for jp in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[jp]

            if indices[j] >= 0:
                cols.append(indices[j])
                data.append(A.data[jp])

        iptr.append(len(cols))

    return scipy.sparse.csr_matrix((data, cols, iptr), shape=(len(dof), len(dof)))

def plot_init():
    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

def plot_plate_vec(nxe, vec, ax, sort_fw, nodal_dof:int=2):
    """assume vec is one DOF per node only here (and includes bcs)"""
    nx = nxe + 1
    N = nx**2

    plot_vec = np.zeros((6*N,))
    plot_vec[sort_fw] = vec[:]
    plot_vec = plot_vec[nodal_dof::6]

    # plot_vec[_free_dof] = vec[:]
    # # reordering of solution for plotting..
    # x, y = _xpts[0::3], _xpts[1::3]
    # sort_idx = np.lexsort((y, x))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
    # # print(f"{sort_idx=}")
    # plot_vec = plot_vec[nodal_dof::6][sort_idx]

    x = np.linspace(0.0, 1.0, nx)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    VALS = np.reshape(plot_vec, (nx, nx))
    ax.plot_surface(X, Y, VALS, cmap='viridis')

def plot_vec_compare(nxe, old_defect, new_defect, sort_fw_map, filename=None, nodal_dof:int=2):
    from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
    plot_init()
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), subplot_kw={'projection': '3d'})
    plot_plate_vec(nxe=nxe, vec=old_defect, ax=ax[0], sort_fw=sort_fw_map, nodal_dof=nodal_dof)
    plot_plate_vec(nxe=nxe, vec=new_defect, ax=ax[1], sort_fw=sort_fw_map, nodal_dof=nodal_dof)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_vec_compare_all(nxe, old_defect, new_defect, sort_fw_map, filename=None):
    from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
    plot_init()
    fig, ax = plt.subplots(2, 3, figsize=(26, 8), subplot_kw={'projection': '3d'})

    # only w, thy, thz are coupled for shell + plate (plotted all six before)
    for idof in [2,3,4]:
        plot_plate_vec(nxe=nxe, vec=old_defect, ax=ax[0,idof-2], sort_fw=sort_fw_map, nodal_dof=idof)
        plot_plate_vec(nxe=nxe, vec=new_defect, ax=ax[1,idof-2], sort_fw=sort_fw_map, nodal_dof=idof)
    ax[0,0].set_title("w")
    ax[0,1].set_title("thx")
    ax[0,2].set_title("thy")

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def zero_non_nodal_dof(vec, sort_fw, inodal:int=2):
    """zero errors in the other nodal dof"""
    for ired in range(vec.shape[0]):
        ifree = sort_fw[ired]
        if ifree % 6 != inodal:
            vec[ired] = 0.0
    return vec

def gauss_seidel_csr(A, b, x0, num_iter=1):
    """
    Perform Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]
    for _ in range(num_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]
            x[i] = (b[i] - sum_) / diag
    return x

def block_gauss_seidel_6dof(A, b: np.ndarray, x0: np.ndarray, num_iter=1):
    """
    Perform Block Gauss-Seidel smoothing for 6 DOF per node.
    A: csr_matrix of size (6*nnodes, 6*nnodes)
    b: RHS vector (6*nnodes,)
    x0: initial guess (6*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    ndof = 6
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x

def sort_vis_maps(nxe, xpts, free_dof):
    """create dof maps from reordered red to unsorted full and reverse"""
    # first is unsort_red to sort_free
    nx = nxe + 1
    N = nx**2
    nred = len(free_dof)
    nfree = 6 * N
    x, y = xpts[0::3], xpts[1::3]
    node_sort_map = np.lexsort((y, x))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
    # temp if don't want to resort basically
    # node_sort_map = np.arange(node_sort_map.shape[0])
    inv_node_sort_map = np.empty_like(node_sort_map)
    inv_node_sort_map[node_sort_map] = np.arange(N)

    sort_free_fw_map = np.zeros(nred, dtype=np.int32)
    sort_free_bk_map = -np.ones(nfree, dtype=np.int32) # -1 if not free var
    for ired in range(nred):
        ifree = free_dof[ired]
        inode = ifree // 6
        idof = ifree % 6

        inode_sort = inv_node_sort_map[inode]
        ifree_sort = 6 * inode_sort + idof

        sort_free_fw_map[ired] = ifree_sort
        sort_free_bk_map[ifree_sort] = ired

    return sort_free_fw_map, sort_free_bk_map

def mg_coarse_fine_operators(nxe_fine, sort_bk_fine, sort_bk_coarse, bcs_list=None):
    """include a bit of reordering in the Ifc and Icf operators, also we first include then remove the bcs.."""

    nxe_coarse = nxe_fine // 2
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    Nc = nxc**2
    Nf = nxf**2
    # assumes it does have bcs here..
    Nc_dof = Nc * 6
    Nf_dof = Nf * 6

    # first include all nodes, then we'll remove the 
    I_fc = np.zeros((Nc_dof, Nf_dof)) # coarse to fine

    for i_c in range(Nc_dof):
        # coarse point
        idof = i_c % 6
        inode_c = i_c // 6
        iyc = inode_c // nxc
        ixc = inode_c % nxc

        # corresponding fine point
        ixf = 2 * ixc
        iyf = 2 * iyc
        inode_f = nxf * iyf + ixf
        i_f = 6 * inode_f + idof

        # print(f"{i_c=}, [n,d]={inode_c=},{idof=}] c:[{ixc},{iyc}], {inode_f=} {i_f=}/{Nf=}")

        I_fc[i_c, i_f] = 4.0 / 16.0 # middle, middle

        if iyf > 0:
            I_fc[i_c, i_f-6*nxf] = 2.0 / 16.0 # middle, bottom

        if iyf < nxf-1:
            I_fc[i_c, i_f+6*nxf] = 2.0 / 16.0 # middle, top

        if ixf > 0:
            I_fc[i_c, i_f-6] = 2.0 / 16.0 # left, middle

            if iyf > 0:
                I_fc[i_c, i_f-6-6*nxf] = 1.0 / 16.0 # left, bottom

            if iyf < nxf-1:
                I_fc[i_c, i_f-6+6*nxf] = 1.0 / 16.0 # left, top
        
        if ixf < nxf-1:
            I_fc[i_c, i_f+6] = 2.0 / 16.0 # right, middle

            if iyf > 0:
                I_fc[i_c, i_f+6-6*nxf] = 1.0 / 16.0 # right, bottom

            if iyf < nxf-1:
                I_fc[i_c, i_f+6+6*nxf] = 1.0 / 16.0 # right, top

    # remove bcs and apply sorting (since the I_fc currently is perfectly sorted)
    I_fc_0 = I_fc.copy() # fine to coarse mapping

    nc_keep = np.sum(sort_bk_coarse != -1)
    nf_keep = np.sum(sort_bk_fine != -1)
    I_fc = np.zeros((nc_keep, nf_keep))

    # print(f"{nf_keep=} {Nf_dof=}")

    for i_c in range(Nc_dof):
        i2_c = sort_bk_coarse[i_c]
        if i2_c == -1: continue

        for i_f in range(Nf_dof):
            i2_f = sort_bk_fine[i_f]
            if i2_f == -1: continue

            # now copy from old mapping operator
            I_fc[i2_c, i2_f] = I_fc_0[i_c, i_f]

    # plt.imshow(I_fc)
    # plt.show()

    # remove bcs for case where we kept them
    if bcs_list is not None:
        bcs_fine = np.array(bcs_list[0])
        bcs_coarse = np.array(bcs_list[1])
        I_fc[bcs_coarse,:] *= 0.0
        I_fc[:,bcs_fine] *= 0.0
    
    # coarse to fine is transpose operator
    I_cf = I_fc.T * 4

    return I_cf, I_fc
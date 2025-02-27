import numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

def get_mat(prefix, nxe):
    kmat_vec = read_from_csv(f"csv/{prefix}_vals.csv")
    # kmat_vec = read_from_csv("csv/plate_kmat_nobcs.csv")
    rowPtr = read_from_csv(f"csv/{prefix}_rowp.csv").astype(np.int32)
    colPtr = read_from_csv(f"csv/{prefix}_cols.csv").astype(np.int32)

    # print(f"{kmat_vec=}")

    # print(f"{kmat_vec.shape[0]}")
    # _nnodes = rowPtr.shape[0] - 1
    # nx = int(_nnodes**0.5)
    # nxe = nx - 1
    nx = nxe + 1
    nnodes = nx**2
    num_elements = nxe**2
    print(f"{nxe=}")
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

def plot_mat(mat, filename):
    plt.figure()
    ax = sns.heatmap(mat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
    plt.savefig(filename, dpi=400)
    return

if __name__=="__main__":
    kmat,_,_,_ = get_mat("kmat", nxe=10)
    plot_mat(kmat, "kmat.png")
    precond,_,_,_ = get_mat("precond", nxe=10)
    plot_mat(precond, "precond.png")
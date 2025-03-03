import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

def plot_sparse_matrix(name):
    rowPtr = read_from_csv(f"csv/{name}_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv(f"csv/{name}_colPtr.csv").astype(np.int32)

    # plot the sparsity now, maybe compute bandwidth
    nnodes = rowPtr.shape[0] - 1
    nnzb = colPtr.shape[0]
    vals = np.ones((nnzb,), dtype=np.double)

    cpp_csr = sp.sparse.csr_matrix((vals, colPtr, rowPtr), shape=(nnodes, nnodes))
    cpp_block_dense = cpp_csr.toarray()
    sns.heatmap(cpp_block_dense)
    # plt.show()
    plt.savefig(f"img/{name}.png", dpi=400)
    plt.close('all')

if __name__ == "__main__":
    plot_sparse_matrix("RCM")
    plot_sparse_matrix("qorder")
    plot_sparse_matrix("ILU")
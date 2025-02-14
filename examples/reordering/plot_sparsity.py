import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

if __name__ == "__main__":
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)

    # plot the sparsity now, maybe compute bandwidth
    nnodes = rowPtr.shape[0] - 1
    nnzb = colPtr.shape[0]
    vals = np.ones((nnzb,), dtype=np.double)

    cpp_csr = sp.sparse.csr_matrix((vals, colPtr, rowPtr), shape=(nnodes, nnodes))
    cpp_block_dense = cpp_csr.toarray()
    sns.heatmap(cpp_block_dense)
    # plt.show()
    plt.savefig("q_ordering.png", dpi=400)
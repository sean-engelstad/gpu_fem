import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

if __name__ == "__main__":
    
    chain_lengths = read_from_csv(f"csv/chain_lengths.csv").astype(np.float32)
    nnodes = chain_lengths.shape[0]
    plt.plot([_ for _ in range(nnodes)], chain_lengths, 'o', markersize=0.5) # reverse chain lengths plot order
    plt.savefig("img/chains/chains_out.png", dpi=400)
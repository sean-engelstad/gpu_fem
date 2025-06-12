import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

if __name__ == "__main__":
    
    un_precond_vec = read_from_csv(f"csv/unprecond_vec.csv").astype(np.float32)
    un_precond_vec = np.abs(un_precond_vec)

    precond_vec = read_from_csv(f"csv/precond_vec.csv").astype(np.float32)
    precond_vec = np.abs(precond_vec)
    N = precond_vec.shape[0]
    
    plt.plot([_ for _ in range(N)], un_precond_vec, 'o', markersize=0.5, label="before-precond") 
    plt.plot([_ for _ in range(N)], precond_vec, 'o', markersize=0.5, label="after-precond") 
    plt.yscale('log')
    plt.ylabel("Variable")
    plt.legend()
    plt.xlabel("Node Number")
    plt.savefig("img/vec/precond_vec.png", dpi=400)
import matplotlib.pyplot as plt
import seaborn as sns

# Input: two dictionaries for each preconditioner
ilu7 = {
    "BiCGStab": 30.0, # no conv
    "DirectLU": 1.322,
    "MGMRES": 12.2,
    "HGMRES": 12.42,
    "PCG": 13.0, # 
    "DirectChol": 4.625,
    "CholGMRES": 29.5,
}

ilu1 = {
    "BiCGStab": 4.1306,
    "DirectLU": 1.322,
    "MGMRES": 1.000,
    "HGMRES": 1.148,
    "PCG": 1.8,
    "DirectChol": 4.625,
    "CholGMRES": 2.15
}

# Prepare for plotting
methods = list(ilu1.keys())
x = range(len(methods))
width = 0.35

# Plot
plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], [ilu1[m] for m in methods], width=width, label='ILU(1)')
plt.bar([i + width/2 for i in x], [ilu7[m] for m in methods], width=width, label='ILU(7)')

plt.xticks(x, methods, rotation=30)
plt.ylabel("Value")
plt.title("Solver Comparison with ILU Preconditioners")
plt.legend()
plt.yscale('log')
plt.tight_layout()
# plt.show()
plt.savefig("out/solver-bar.png", dpi=400)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# num nodes
# L1 - 23738, L2 - , L3 - 
meshes = [23738, 54439, 88341]

# gpu_fem GPU runtimes data, on H100 GPU's
# gpu_dict = {
#     # factorization is on the host anyways
#     'factorization' : [n, 4.182, ],
#     'assembly' : [n, 0.0172, ],
#     'LU_solve' : [n, 1.983, ]
# }

# 1 process
cpu_1_dict = {
    'factorization' : [0.345, 1.0412, 1.686],
    'assembly' : [0.705, 1.4856, 2.517],
    'LU_solve' : [3.2891, 9.7261, 22.13]
}

# on 4 processes
cpu_4_dict = {
    'factorization' : [0.2021, 0.7452, 1.2479],
    'assembly' : [0.2752, 0.5211, 0.996],
    'LU_solve' : [3.0511, 9.1801, 21.29]
}

# plot script
# --------------
df_bar = pd.DataFrame({
    'Mesh Size': np.repeat(meshes, 2),
    'Factorization': cpu_1_dict['factorization'] + cpu_4_dict['factorization'],
    'Assembly': cpu_1_dict['assembly'] + cpu_4_dict['assembly'],
    'LU Solve': cpu_1_dict['LU_solve'] + cpu_4_dict['LU_solve'],
    'Process': ['1 CPU']*3 + ['4 CPU']*3
})

df_melted = df_bar.melt(id_vars=['Mesh Size', 'Process'], var_name='Operation', value_name='Runtime (s)')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Mesh Size", y="Runtime (s)", hue="Operation", ci=None, palette="Set2", dodge=True)
plt.yscale('log')
plt.title("Runtime Comparison for Different FEM Components")
plt.xlabel("Mesh Size")
plt.ylabel("Runtime (s) (log scale)")
plt.legend(title="Operation")
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.show()
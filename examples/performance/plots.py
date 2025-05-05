import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

# MELD plots
# ----------

meld_df = pd.read_csv("csv/meld.csv")
arr = meld_df.to_numpy()
print(f"{arr=}")
plt.style.use(niceplots.get_style())
plt.figure('meld')
plt.plot(arr[:,0], arr[:,2], label="transferDisps")
plt.plot(arr[:,0], arr[:,3], label="transferLoads")
plt.legend()
plt.xlabel("Struct nnodes")
plt.ylabel("MELD runtime (s)")
plt.xscale('log')
plt.yscale('log')
plt.savefig("out/meld_runtime.svg", dpi=400)
plt.close('meld')

# plate runtimes
# --------------

cmap = plt.get_cmap("RdYlBu_r")
color_values = np.linspace(0, 1, 4)
colors = [cmap(val) for val in color_values]

linear_static_df = pd.read_csv("csv/linear_static.csv")
arr = linear_static_df.to_numpy()
print(f"{arr=}")
plt.style.use(niceplots.get_style())
plt.figure('linaer_static', figsize=(8, 5))
for i, ordering in enumerate(["none", "AMD", "RCM", "qorder"]):
    for solve in ["LU", "GMRES"]:
    # for solve in ["LU"]:
    # for solve in ["GMRES"]:
        mask = np.logical_and(arr[:,2] == (" " + ordering), arr[:,4] == (" " + solve))
        if (solve == "GMRES"):
            mask = np.logical_and(mask, arr[:,3] == "  ILU(5)")
        x = np.array(arr[mask,1], dtype=np.double)
        y = np.array(arr[mask, -1], dtype=np.double)
        plt.plot(x, y,  label=f"{ordering}-{solve}", linestyle="-" if solve == "LU" else "--", color=colors[i])

plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.subplots_adjust(right=0.75)  # or 0.8 depending on your legend width


plt.xlabel("nnodes")
plt.ylabel("linear static runtime (s)")
plt.xscale('log')
plt.yscale('log')
plt.savefig("out/linear_static.svg", dpi=400)
plt.close('linaer_static')
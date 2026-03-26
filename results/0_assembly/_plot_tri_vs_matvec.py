import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

plt.style.use(niceplots.get_style())

filename = "csv/tri_matvec.csv"
savefig = "out/tri_vs_matvec_ratio.svg"

df = pd.read_csv(filename)

# numeric cleanup
df["nxe"] = pd.to_numeric(df["nxe"], errors="coerce")
df["tri_solve_ms"] = pd.to_numeric(df["tri_solve_ms"], errors="coerce")
df["mat_vec_ms"] = pd.to_numeric(df["mat_vec_ms"], errors="coerce")

# compute ratio
df["ratio"] = df["tri_solve_ms"] / df["mat_vec_ms"]

# BIG fonts
fs = 24
plt.rcParams.update({
    "font.size": fs,
    "axes.titlesize": fs + 4,
    "axes.labelsize": fs + 2,
    "xtick.labelsize": fs - 2,
    "ytick.labelsize": fs - 2,
    "legend.fontsize": fs - 2,
    "axes.linewidth": 2.5,
})

fig, ax = plt.subplots(figsize=(8.5, 6.8))

# loop over GPUs (so you can add A100 later)
gpus = df["GPU"].unique()

for gpu in gpus:
    sub = df[df["GPU"] == gpu].copy()
    sub = sub.sort_values("nxe")

    x = sub["nxe"].to_numpy()
    y = sub["ratio"].to_numpy()

    x = 6 * (x+1)**2 # nxe => DOF

    ax.plot(
        x, y, "o-",
        linewidth=3,
        markersize=9,
        label=f"{gpu}",
    )

# log scale is important here
ax.set_yscale("log")
ax.set_xscale('log')

ax.set_xlabel("Degree of Freedom")
ax.set_ylabel(r"LU Solve Time / Mat-Vec Time")

plt.margins(x=0.05, y=0.05)
ax.grid(True, which="major", alpha=0.3)
ax.legend()


ax.set_yticks([10, 50, 100, 1000])
ax.set_yticklabels([r"$10^{1}$", r"$50$", r"$10^{2}$", r"$10^{3}$"])


plt.tight_layout()
plt.savefig(savefig, dpi=400)
plt.show()
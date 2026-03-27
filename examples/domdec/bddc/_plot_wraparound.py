import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

plt.style.use(niceplots.get_style())

filename = "out/wing_wrap.csv"
# savefig = "out/wrap_sd_runtime_ratio.png"
savefig = "out/wrap_sd_runtime_ratio.svg"

plot_ratio_to_best = True

df = pd.read_csv(filename)

# numeric cleanup
df["thick"] = pd.to_numeric(df["thick"], errors="coerce")
df["wrap_sd_frac"] = pd.to_numeric(df["wrap_sd_frac"], errors="coerce")
df["frac_vert_interior"] = pd.to_numeric(df["frac_vert_interior"], errors="coerce")
df["num_iters"] = pd.to_numeric(df["num_iters"], errors="coerce")
df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")

thicknesses = [1e-1, 1e-2, 1e-3]

# colors = {
#     1e-1: "#1f77b4",
#     1e-2: "#ff7f0e",
#     1e-3: "#2ca02c",
# }
labels = {
    1e-1: r"$t = 10^{-1}$",
    1e-2: r"$t = 10^{-2}$",
    1e-3: r"$t = 10^{-3}$",
}

# BIG fonts (publication / slides)
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

for thick in thicknesses:
    # use isclose for float comparison
    sub = df[np.isclose(df["thick"], thick)].copy()
    sub = sub.sort_values("wrap_sd_frac")

    if len(sub) == 0:
        print(f"Warning: no rows found for thick = {thick}")
        continue

    x = sub["wrap_sd_frac"].to_numpy()

    if plot_ratio_to_best:
        y_raw = sub["runtime_sec"].to_numpy(dtype=float)

        valid = np.isfinite(y_raw)
        if not np.any(valid):
            print(f"Warning: no valid runtime values for thick = {thick}")
            continue

        best = np.nanmin(y_raw)
        y = y_raw / best
        ylabel = "Runtime / Best Runtime"
        title = "Wraparound Fraction vs Runtime Ratio"
    else:
        y = sub["num_iters"].to_numpy(dtype=float)
        if not np.any(np.isfinite(y)):
            print(f"Warning: no valid iteration values for thick = {thick}")
            continue

        ylabel = "PCG Iterations"
        title = "Wraparound Fraction vs PCG Iterations"

    ax.plot(
        x, y, "o-",
        linewidth=3,
        markersize=9,
        # color=colors[thick],
        label=labels[thick],
    )

if plot_ratio_to_best:
    ax.axhline(1.0, linestyle="--", linewidth=2, color="gray")
    ax.set_yscale("log")
else:
    ax.set_yscale("log")

ax.set_xlabel("Wraparound Subdomain Fraction")
ax.set_ylabel(ylabel)
# ax.set_title(title)
plt.margins(x=0.05, y=0.05)
ax.set_xlim(-0.02, 1.05)
ax.grid(True, which="major", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(savefig, dpi=400)
plt.show()
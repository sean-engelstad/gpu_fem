import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

fs = 20

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

# base_runtime = thick-shell runtime at t = 1e-1
# S_t = thick/thin runtime ratio = time(t=1e-1) / time(t=1e-3)
# so thin_runtime = thick_runtime / S_t
rows = [
    ("Direct-LU", False, 16.50, 1.000),
    ("GMG-CP",    True,  1.261, 0.067),
    ("GMG-ASW",   True,  1.031, 0.167),
    ("BDDC-LU",   True,  2.526, 0.687),
    ("BDDC-AMG",  True,  3.489, 0.578),
]

df = pd.DataFrame(rows, columns=["solver", "multilevel", "thick_runtime", "S_t"])
df["thin_runtime"] = df["thick_runtime"] / df["S_t"]

plt.style.use(niceplots.get_style())

fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_axes([0.08, 0.12, 0.62, 0.82])

# Wing image on the right
img_ax = fig.add_axes([0.76, 0.4, 0.2, 0.5])
img = plt.imread("aob-wing.png")
img_ax.imshow(img)
img_ax.axis("off")

single_level = df[df["solver"] == "Direct-LU"]
multigrid = df[df["solver"].isin(["GMG-CP", "GMG-ASW"])]
domain_decomp = df[df["solver"].isin(["BDDC-LU", "BDDC-AMG"])]

marker_size = 90

ax.scatter(
    single_level["thin_runtime"], single_level["thick_runtime"],
    label="Single-level", s=marker_size, zorder=3
)
ax.scatter(
    multigrid["thin_runtime"], multigrid["thick_runtime"],
    label="Multigrid", s=marker_size, zorder=3
)
ax.scatter(
    domain_decomp["thin_runtime"], domain_decomp["thick_runtime"],
    label="Domain-Decomp", s=marker_size, zorder=3
)

# Direct-LU reference box
lu_row = df[df["solver"] == "Direct-LU"].iloc[0]
lu_t = lu_row["thin_runtime"]
lu_T = lu_row["thick_runtime"]

# Dashed LU bounding box from (0,0) to (LU_thin, LU_thick)
ax.plot([0, lu_t], [0, 0], "--", lw=1.2, color="black", zorder=1)
ax.plot([0, 0], [0, lu_T], "--", lw=1.2, color="black", zorder=1)
ax.plot([lu_t, lu_t], [0, lu_T], "--", lw=1.2, color="black", zorder=1)
ax.plot([0, lu_t], [lu_T, lu_T], "--", lw=1.2, color="black", zorder=1)

# Diagonal dashed line: thick shell = thin shell
diag_max = max(df["thin_runtime"].max(), df["thick_runtime"].max()) * 1.05
ax.plot([0, diag_max], [0, diag_max], "--", lw=1.2, color="black", zorder=1)

# label_offsets = {
#     "Direct-LU": (10, -16),
#     "GMG-CP":    (10, 2),
#     "GMG-ASW":   (10, 10),
#     "BDDC-LU":   (10, 8),
#     "BDDC-AMG":  (10, -16),
# }

# label_offsets = {
#     "Direct-LU": (10, -16),

#     # move right cluster apart
#     "GMG-CP":    (12, 6),

#     "GMG-ASW":   (14, 14),      # push up-right
#     "BDDC-LU":   (-70, 8),      # push left
#     "BDDC-AMG":  (14, -18),     # push down-right
# }
label_offsets = {
    "Direct-LU": (10, -16),
    "GMG-CP":    (-10, 12),

    "GMG-ASW":   (-10, 14),      # right / slightly up
    "BDDC-LU":   (-40, 14),      # left
    "BDDC-AMG":  (-10, 12),     # left / slightly down
}

for _, r in df.iterrows():
    x = r["thin_runtime"]
    y = r["thick_runtime"]
    dx, dy = label_offsets[r["solver"]]

    ax.annotate(
        r["solver"],
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=16,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            pad=0.25,
            alpha=1.0,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            lw=0.5,
        ),
        zorder=4,
    )

ax.legend(
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.8)
)

ax.set_xlabel(r"Thin-shell runtime, $t = 10^{-3}$ (s)")
ax.set_ylabel(r"Thick-shell runtime, $t = 10^{-1}$ (s)")

ax.set_xlim(0.0, diag_max)
ax.set_ylim(0.0, max(df["thick_runtime"].max() * 1.05, 17.5))
ax.tick_params(axis="both", labelsize=fs)
ax.grid(True)

plt.margins(x=0.05, y=0.05)

plt.savefig("solver_runtime_thick_vs_thin_wing.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_runtime_thick_vs_thin_wing.svg", dpi=200, bbox_inches="tight")
plt.show()
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
    ("Direct-LU", False, 12.31, 1.000),
    ("DJ",        False, 2.841, 0.144),
    ("MCGS",      False, 3.976, 0.140),
    ("CP",        False, 2.620, 0.146),
    ("ILU(0)",    False, 22.61, 1.0e-3),  # upper bound from table: <1e-3
    ("ILU(1)",    False, 8.712, 1.0e-3),  # upper bound from table: <1e-3
    ("ILU(2)",    False, 5.718, 1.0e-3),  # upper bound from table: <1e-3
    ("ASW",       False, 2.942, 0.521),

    ("GMG-CP",    True,  0.400, 0.026),
    ("GMG-ASW",   True,  0.412, 0.065),
    ("SA-AMG",    True,  1.871, 0.082),
    ("RN-AMG",    True,  1.423, 0.088),
    ("BDDC-LU",   True,  1.507, 0.845),
    ("BDDC-AMG",  True,  1.293, 0.167),
]

df = pd.DataFrame(rows, columns=["solver", "multilevel", "thick_runtime", "S_t"])
# Remove ILU(k) since they effectively fail for thin shells
df = df[~df["solver"].str.contains("ILU")]
df["thin_runtime"] = df["thick_runtime"] / df["S_t"]

plt.style.use(niceplots.get_style())

fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_axes([0.08, 0.12, 0.62, 0.82])

# Cylinder image on the right
img_ax = fig.add_axes([0.76, 0.4, 0.2, 0.5])  # [left, bottom, width, height]
# img_ax = fig.add_axes([0.71, 0.18, 0.26, 0.66])
img = plt.imread("cylinder.png")
img_ax.imshow(img)
img_ax.axis("off")

single_level = df[~df["multilevel"]]
multigrid = df[df["solver"].str.contains("GMG|AMG")]
domain_decomp = df[df["solver"].str.contains("BDDC")]

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

# Direct-LU reference
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

label_offsets = {
    "Direct-LU": (10, -16),
    "ASW":       (8, 8),
    "CP":        (8, 14),
    "DJ":        (16, 12),
    "MCGS":      (14, -2),
    "ILU(0)":    (-70, -10),
    "ILU(1)":    (-60, 8),
    "ILU(2)":    (-60, -18),
    "GMG-CP":    (10, 4),
    "GMG-ASW":   (10, 10),
    "SA-AMG":    (12, -18),
    "RN-AMG":    (12, 2),
    "BDDC-LU":   (10, 8),
    "BDDC-AMG":  (10, 8),
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
    # loc="upper left",
    loc="upper right",
    bbox_to_anchor=(1.0, 0.85)
)

ax.set_xlabel(r"Thin-shell runtime, $t = 10^{-3}$ (s)")
ax.set_ylabel(r"Thick-shell runtime, $t = 10^{-1}$ (s)")

ax.set_xlim(0.0, diag_max)
ax.set_ylim(0.0, max(df["thick_runtime"].max() * 1.05, 13.0))
ax.tick_params(axis="both", labelsize=fs)
ax.grid(True)

plt.margins(x=0.05, y=0.05)

plt.savefig("solver_runtime_thick_vs_thin.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_runtime_thick_vs_thin.svg", dpi=200, bbox_inches="tight")
plt.show()
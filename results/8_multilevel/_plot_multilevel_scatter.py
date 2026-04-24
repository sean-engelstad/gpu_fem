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

rows = [
    ("Direct-LU", False, 12.31, 1.000, 0.786),
    ("DJ",        False, 2.841, 0.144, 0.726),
    ("MCGS",      False, 5.173, 0.140, 0.739),
    ("CP",        False, 2.620, 0.146, 0.722),
    ("ILU(0)",    False, 22.61, 0.005, 0.519),
    ("ILU(1)",    False, 8.712, 0.005, 0.590),
    ("ILU(2)",    False, 5.718, 0.005, 0.587),
    ("ASW",       False, 2.942, 0.521, 0.743),

    ("GMG-CP",    True,  0.400, 0.026, 1.509),
    ("GMG-ASW",   True,  0.412, 0.065, 1.641),
    ("SA-AMG",    True,  1.871, 0.082, 0.910),
    ("RN-AMG",    True,  1.423, 0.088, 0.948),
    ("BDDC-LU",   True,  1.507, 0.845, 1.027),
    ("BDDC-AMG",  True,  1.293, 0.167, 1.124),
]

df = pd.DataFrame(rows, columns=["solver", "multilevel", "base_runtime", "S_t", "S_h"])

plt.style.use(niceplots.get_style())
# fig, ax = plt.subplots(figsize=(8, 6))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.08, 0.1, 0.65, 0.85])  # main plot (left)

# Add cylinder image on the right
img_ax = fig.add_axes([0.76, 0.4, 0.2, 0.5])  # [left, bottom, width, height]
# img_ax = fig.add_axes([0.76, 0.25, 0.2, 0.5])  # [left, bottom, width, height]
# img_ax = fig.add_axes([0.72, 0.18, 0.26, 0.65])

img = plt.imread("cylinder.png")  # <-- your image file
img_ax.imshow(img)
img_ax.axis("off")

single_level = df[~df["multilevel"]]
multigrid = df[df["solver"].str.contains("GMG|AMG")]
domain_decomp = df[df["solver"].str.contains("BDDC")]

marker_size = 80

ax.scatter(
    single_level["S_h"], single_level["S_t"],
    label="Single-level", s=marker_size, zorder=2
)
ax.scatter(
    multigrid["S_h"], multigrid["S_t"],
    label="Multigrid", s=marker_size, zorder=2
)
ax.scatter(
    domain_decomp["S_h"], domain_decomp["S_t"],
    label="Domain-Decomp", s=marker_size, zorder=2
)

# Offsets are in display points, not data units.
label_offsets = {
    "Direct-LU":  (10, -16),
    "ASW":        (8, 10),
    "CP":         (-10, 18),
    "DJ":         (22, 14),
    "MCGS":       (22, -2),
    "ILU(0)":     (0, 28),
    "ILU(1)":     (42, 2),
    "ILU(2)":     (24, 16),
    "GMG-CP":     (-70, 10),
    # "GMG-ASW":    (-70, 18),
    "GMG-ASW": (-60, 15),
    "SA-AMG":     (20, -22),
    "RN-AMG":     (24, 3),
    "BDDC-LU":    (6, 8),
    "BDDC-AMG":   (6, 8),
}

for _, r in df.iterrows():
    x = r["S_h"]
    y = r["S_t"]
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
    loc="center right",
)

ax.set_xlabel(r"$S_h\; (h\mathrm{-}scalability)$")
ax.set_ylabel(r"$S_t\; (t\mathrm{-}independence)$")

ax.set_xlim(0.45, 1.75)
ax.set_ylim(-0.05, 1.05)
ax.tick_params(axis="both", labelsize=fs)
ax.grid(True)

plt.savefig("solver_scatter.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_scatter.svg", dpi=200, bbox_inches="tight")
plt.show()
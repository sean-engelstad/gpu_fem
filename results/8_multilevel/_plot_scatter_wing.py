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

# Wing-only solver data
rows = [
    ("Direct-LU", False, 16.50, 1.000, 0.776),
    ("GMG-CP",    True,  1.261, 0.067, 1.149),
    ("GMG-ASW",   True,  1.031, 0.167, 1.247),
    ("BDDC-LU",   True,  2.526, 0.687, 1.028),
    ("BDDC-AMG",  True,  3.489, 0.578, 1.013),
]

df = pd.DataFrame(rows, columns=["solver", "multilevel", "base_runtime", "S_t", "S_h"])

plt.style.use(niceplots.get_style())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.08, 0.1, 0.65, 0.85])

# Add wing image on the right
img_ax = fig.add_axes([0.76, 0.4, 0.2, 0.5])
img = plt.imread("aob-wing.png")
img_ax.imshow(img)
img_ax.axis("off")

single_level = df[~df["multilevel"]]
multigrid = df[df["solver"].str.contains("GMG|AMG")]
domain_decomp = df[df["solver"].str.contains("BDDC")]

marker_size = 90

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

label_offsets = {
    "Direct-LU":  (10, -16),
    "GMG-CP":     (10, -6),
    "GMG-ASW":    (10, 10),
    "BDDC-LU":    (10, 8),
    "BDDC-AMG":   (10, -16),
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
    loc="lower left",
)

ax.set_xlabel(r"$S_h\; (h\mathrm{-}scalability)$")
ax.set_ylabel(r"$S_t\; (t\mathrm{-}independence)$")

ax.set_xlim(0.70, 1.32)
ax.set_ylim(-0.05, 1.05)
ax.tick_params(axis="both", labelsize=fs)
ax.grid(True)

plt.savefig("solver_scatter_wing.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_scatter_wing.svg", dpi=200, bbox_inches="tight")
plt.show()
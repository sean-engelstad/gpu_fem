import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots
from adjustText import adjust_text

fs = 20  # change this to whatever you want
# fs = 24

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

# Build dataset from table
rows = [
    ("Direct-LU", False, 12.31, 1.000, 0.786),
    ("DJ",        False, 2.841, 0.144, 0.726),
    ("MCGS",      False, 5.173, 0.140, 0.739),
    ("CP",        False, 2.620, 0.146, 0.722),
    ("ILU(0)",    False, 22.61, 0.005, 0.519),  # using 0.005 for "<0.01"
    ("ILU(1)",    False, 8.712, 0.005, 0.590),
    ("ILU(2)",    False, 5.718, 0.005, 0.587),
    ("ASW",       False, 2.942, 0.521, 0.743),

    ("GMG-CP",    True,  0.400, 0.026, 1.509),
    ("GMG-ASW",   True,  0.412, 0.065, 1.641),
    ("SA-AMG",    True,  1.871, 0.082, 0.910),
    ("RN-AMG",    True,  1.423, 0.088, 0.948),
    ("BDDC",      True,  1.507, 0.845, 1.027),
]

df = pd.DataFrame(rows, columns=["solver", "multilevel", "base_runtime", "S_t", "S_h"])

print(df)

plt.style.use(niceplots.get_style())

fig, ax = plt.subplots(figsize=(8, 6))

# plot points
# ax.scatter(df["S_h"], df["S_t"])
# Define groups
single_level = df[~df["multilevel"]]
multigrid = df[df["solver"].str.contains("GMG|AMG")]
domain_decomp = df[df["solver"] == "BDDC"]

# Plot each group with different colors
ax.scatter(single_level["S_h"], single_level["S_t"],
           label="Single-level", s=80)

ax.scatter(multigrid["S_h"], multigrid["S_t"],
           label="Multigrid", s=80)

ax.scatter(domain_decomp["S_h"], domain_decomp["S_t"],
           label="Domain-Decomp", s=80)

# label points
texts = []
points = []

for _, r in df.iterrows():
    x = r["S_h"]
    y = r["S_t"]
    txt = ax.text(x + 0.01, y + 0.01, r["solver"], fontsize=16)
    texts.append(txt)
    points.append((x, y))

# adjust labels
adjust_text(
    texts,
    target_x=[p[0] for p in points],
    target_y=[p[1] for p in points],
    arrowprops=None,
    expand=(1.2, 1.3),
    force_text=(0.5, 0.7),
    force_static=(0.5, 0.7),
    force_pull=(0.05, 0.05),
    ensure_inside_axes=True,
    ax=ax,
)

# draw connector lines only when needed
for txt, (x, y) in zip(texts, points):
    xt, yt = txt.get_position()
    dist2 = (xt - x) ** 2 + (yt - y) ** 2
    if dist2 > (0.035) ** 2:
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(xt, yt),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

# ax.legend()
ax.legend(frameon=True, facecolor="white", framealpha=1.0, loc="center right")

ax.set_xlabel(r"$S_h\; (h\mathrm{-}independence)$")
ax.set_ylabel(r"$S_t\; (t\mathrm{-}independence)$")

ax.set_xlim(0.45, 1.75)
ax.set_ylim(-0.05, 1.05)

ax.tick_params(axis='both', labelsize=fs)

ax.grid(True)

plt.savefig("solver_scatter.png", dpi=200, bbox_inches="tight")
plt.show()
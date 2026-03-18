import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

# Seed for reproducibility
rng = np.random.default_rng(0)

# Build dataset
rows = [
    ("Direct-LU", False, 1.000, 0.468),
    ("DJ", False, 0.144, np.nan),
    ("MCGS", False, 0.140, np.nan),
    ("CP", False, 0.146, 0.408),
    ("ILU(0)", False, 0.005, np.nan),
    ("ILU(1)", False, 0.005, np.nan),
    ("ILU(2)", False, 0.005, np.nan),
    ("SPAI", False, 0.005, np.nan),
    ("ASW", False, 0.521, np.nan),

    ("GMG", True, 0.087, np.nan),
    ("EP-GMG", True, 0.205, np.nan),
    ("ASW-ILU", True, np.nan, np.nan),
    ("SA-AMG", True, 0.089, np.nan),
    ("CF-AMG", True, np.nan, np.nan),
    ("RN-AMG", True, np.nan, np.nan),
    ("AMGe", True, np.nan, np.nan),
    ("ML-ASW", True, 0.067, np.nan),
    ("FETI-DP", True, 0.562, np.nan),
    ("BDDC", True, np.nan, np.nan),
]

df = pd.DataFrame(rows, columns=["solver","multilevel","S_t","S_h"])

# Fill missing S_h
# single-level → random [0.3,0.5]
mask_single = (~df.multilevel) & (df.S_h.isna())
# df.loc[mask_single,"S_h"] = rng.uniform(0.3,0.5,mask_single.sum())
# spread values evenly, then add slight randomness
n = mask_single.sum()

base_vals = np.linspace(0.30, 0.50, n)          # evenly spaced
jitter = rng.uniform(-0.015, 0.015, n)          # small randomness

df.loc[mask_single, "S_h"] = base_vals + jitter

# multilevel → near 1
mask_multi = (df.multilevel) & (df.S_h.isna()) & (df.S_t.notna())
df.loc[mask_multi,"S_h"] = rng.uniform(0.95,1.0,mask_multi.sum())

print("WARNING: Using assumed h-independence values (multilevel≈1, single-level random 0.3–0.5)")

# Filter plottable
plot_df = df[df.S_t.notna() & df.S_h.notna()]

plt.style.use(niceplots.get_style())

# Plot
plt.figure(figsize=(8,6))

# plot points
plt.scatter(plot_df["S_h"], plot_df["S_t"])

# # label points
# for _, r in plot_df.iterrows():
#     plt.text(r["S_h"]+0.005, r["S_t"]+0.005, r["solver"], fontsize=9)
from adjustText import adjust_text

# texts = []
# for _, r in plot_df.iterrows():
#     texts.append(
#         plt.text(r["S_h"], r["S_t"], r["solver"], fontsize=13)
#     )

# # v = w = 1.5
# # v = w = 2.0
# v = w = 5.0

# adjust_text(
#     texts,
#     arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
#     expand_points=(v, v),
#     expand_text=(w, w),
# )

# texts = []
# xvals = []
# yvals = []

# for _, r in plot_df.iterrows():
#     x = r["S_h"]
#     y = r["S_t"]

#     dx = 0.01
#     dy = 0.01

#     texts.append(
#         plt.text(x + dx, y + dy, r["solver"], fontsize=13)
#     )
#     xvals.append(x)
#     yvals.append(y)

# adjust_text(
#     texts,
#     target_x=xvals,
#     target_y=yvals,
#     arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
#     expand_points=(2.0, 2.0),
#     expand_text=(1.5, 1.5),
#     force_points=0.6,
#     force_text=0.4,
# )

from adjustText import adjust_text

fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(niceplots.get_style())

ax.scatter(plot_df["S_h"], plot_df["S_t"])

# Create texts with a small initial offset
texts = []
points = []

for _, r in plot_df.iterrows():
    x = r["S_h"]
    y = r["S_t"]
    txt = ax.text(x + 0.008, y + 0.008, r["solver"], fontsize=13)
    texts.append(txt)
    points.append((x, y))

# First: move labels only, no arrows yet
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

# Second: draw connectors only when the label is actually far enough away
# Threshold is in DATA coordinates here; tune to taste
for txt, (x, y) in zip(texts, points):
    xt, yt = txt.get_position()
    dist2 = (xt - x)**2 + (yt - y)**2

    # only draw a line if the label moved a meaningful amount
    if dist2 > (0.035)**2:
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(xt, yt),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

plt.xlabel("S_h (h-independence)")
plt.ylabel("S_t (thickness-independence)")
plt.title("Solver Independence Comparison")

plt.xlim(0.25,1.05)
plt.ylim(0.0,1.05)

plt.grid(True)

plt.savefig("solver_scatter.png", dpi=200, bbox_inches="tight")
# plt.show()
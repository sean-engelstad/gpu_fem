import matplotlib.pyplot as plt
import niceplots
from matplotlib.patches import Patch

# --- style ---
plt.style.use(niceplots.get_style())

fs = 20
plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 1,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

# --- data ---
single_level = [
    ("Direct-LU", 0.124),
    ("DJ",        0.115),
    ("MCGS",      0.078),
    ("CP",        0.127),
    ("ILU(0)",    0.001),
    ("ILU(1)",    0.001),
    ("ILU(2)",    0.001),
    ("ASW",       0.334),
]

multigrid = [
    ("GMG-CP",    0.319),
    ("GMG-ASW",   0.816),
    ("SA-AMG",    0.104),
    ("RN-AMG",    0.157),
]

domain_decomp = [
    ("BDDC-LU",   1.375),
    ("BDDC-AMG",  0.350),
]

# --- colors ---
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
c0 = cycle[0]  # single-level
c1 = cycle[1]  # multigrid
c2 = cycle[2]  # domain-decomp

# --- combine all, then sort globally ---
all_data = [(name, val, c0) for name, val in single_level]
all_data += [(name, val, c1) for name, val in multigrid]
all_data += [(name, val, c2) for name, val in domain_decomp]
all_data = sorted(all_data, key=lambda x: x[1], reverse=True)

labels = [x[0] for x in all_data]
values = [x[1] for x in all_data]
colors = [x[2] for x in all_data]
x = list(range(len(labels)))

# --- plot ---
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(x, values, width=0.72, color=colors)

# labels
ax.set_ylabel("Overall metric")
ax.set_xlabel("Solver")
ax.set_title("Overall solver metric (cylinder)")

# ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs - 1)

# legend
legend_elements = [
    Patch(facecolor=c0, label="Single-level"),
    Patch(facecolor=c1, label="Multigrid"),
    Patch(facecolor=c2, label="Domain-Decomp"),
]
ax.legend(
    handles=legend_elements,
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper right",
)

# grid
ax.grid(True, axis="y")
ax.set_axisbelow(True)

# y-axis ticks
ax.set_ylim(0.0, 1.4)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])

# spacing
ax.set_xlim(-0.6, len(labels) - 0.4)
plt.tight_layout()
plt.margins(x=0.02, y=0.04)

plt.savefig("solver_bar.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_bar.svg", dpi=200, bbox_inches="tight")
plt.show()
import matplotlib.pyplot as plt
import niceplots
from matplotlib.patches import Patch

# --- style ---
plt.style.use(niceplots.get_style())

fs = 18
plt.rcParams.update({
    "font.size": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 2,
    "ytick.labelsize": fs - 1,
    "legend.fontsize": fs - 1,
})

# --- data ---
single_level = [
    ("Direct-LU", 0.197),
]

multigrid = [
    ("GMG-CP",  0.447),
    ("GMG-ASW", 1.434),
]

domain_decomp = [
    ("BDDC-LU",  1.503),
    ("BDDC-AMG", 0.838),
]

# --- colors ---
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
c0 = cycle[0]  # single-level
c1 = cycle[1]  # multigrid
c2 = cycle[2]  # domain-decomp

# --- combine and sort globally ---
all_data = [(name, val, c0) for name, val in single_level]
all_data += [(name, val, c1) for name, val in multigrid]
all_data += [(name, val, c2) for name, val in domain_decomp]
all_data = sorted(all_data, key=lambda x: x[1], reverse=True)

labels = [x[0] for x in all_data]
values = [x[1] for x in all_data]
colors = [x[2] for x in all_data]

x_all = list(range(len(labels)))

# --- plot ---
fig, ax = plt.subplots(figsize=(8.2, 4.6))
bars = ax.bar(x_all, values, width=0.62, color=colors)

# labels/title
ax.set_ylabel("Overall metric")
ax.set_xlabel("Solver")
ax.set_title("Overall solver metric (wing)")

# ticks
ax.set_xticks(x_all)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=fs - 2)

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
ax.set_ylim(0.0, 1.6)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

# spacing
ax.set_xlim(-0.6, len(labels) - 0.4)
plt.tight_layout()
plt.margins(x=0.05, y=0.05)

plt.savefig("solver_bar_wing.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_bar_wing.svg", dpi=200, bbox_inches="tight")
plt.show()
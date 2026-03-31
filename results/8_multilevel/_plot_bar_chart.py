import matplotlib.pyplot as plt
import niceplots
from matplotlib.patches import Patch

# --- style ---
fs = 20

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

plt.style.use(niceplots.get_style())

# --- data ---
solvers_no = ["Direct-LU", "DJ", "MCGS", "CP", "ILU(0)", "ILU(1)", "ILU(2)", "ASW"]
overall_no = [0.818, 0.125, 0.114, 0.128, 0.003, 0.003, 0.003, 0.458]

solvers_yes = ["GMG-CP", "GMG-ASW", "SA-AMG", "RN-AMG", "BDDC"]
overall_yes = [0.063, 0.178, 0.087, 0.115, 1.091]

# --- sort each group (descending = better visually) ---
no_sorted = sorted(zip(overall_no, solvers_no), reverse=True)
yes_sorted = sorted(zip(overall_yes, solvers_yes), reverse=True)

overall_no_sorted, solvers_no_sorted = zip(*no_sorted)
overall_yes_sorted, solvers_yes_sorted = zip(*yes_sorted)

# --- combine with gap ---
solvers = list(solvers_no_sorted) + [""] + list(solvers_yes_sorted)
overall = list(overall_no_sorted) + [0.0] + list(overall_yes_sorted)

# --- colors (default cycle, but grouped) ---
c0 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
c1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

colors = (
    [c0] * len(solvers_no_sorted) +
    ["white"] +
    [c1] * len(solvers_yes_sorted)
)

# --- plot ---
fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(solvers, overall, color=colors)

# separator
ax.axvline(len(solvers_no_sorted) - 0.5, linestyle="--", linewidth=1.0)

# labels
ax.set_ylabel("Overall metric")
ax.set_xlabel("Solver")
ax.set_title("Overall solver metric")

# ticks
ax.set_xticklabels(solvers, rotation=45, ha="right")

# legend (clean, same style as scatter)
legend_elements = [
    Patch(facecolor=c0, label="Single-level"),
    Patch(facecolor=c1, label="Multilevel"),
]
ax.legend(
    handles=legend_elements,
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="center",
    bbox_to_anchor=(0.4, 0.55)
)

# grid (match your scatter)
ax.grid(True, axis="y")

# tighten
plt.tight_layout()

# plt.savefig("solver_bar.png", dpi=200, bbox_inches="tight")
plt.savefig("solver_bar.svg", dpi=200, bbox_inches="tight")
plt.show()
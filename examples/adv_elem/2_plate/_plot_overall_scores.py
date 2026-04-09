import matplotlib.pyplot as plt
import niceplots
from matplotlib.patches import Patch

# -------------------------------------------------
# Style
# -------------------------------------------------
plt.style.use(niceplots.get_style())

fs = 20
plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 1,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

# -------------------------------------------------
# Plate data
# thick = t=1e-1, thin = t=1e-3
# score = min(KL_thick / elem_thick, KL_thin / elem_thin)
# None = no convergence / not computed
# -------------------------------------------------
data_by_cycle = {
    "V22": {
        "kl_thick": 9.0,
        "kl_thin": 9.0,
        "iter_data": {
            "RMR":      {"thick": 20, "thin": None, "group": "c0"},
            "MITC4":    {"thick": 6,  "thin": None, "group": "c0"},
            "MITC4-LP": {"thick": 7,  "thin": 117,  "group": "smoothp"},
            "MITC4-EP": {"thick": 10, "thin": None, "group": "smoothp"},
            "MITC9":    {"thick": None, "thin": None, "group": "c0"},
            "MIG2":     {"thick": 9,  "thin": 10,   "group": "c1"},
        },
    },
    "K22": {
        "kl_thick": 4.0,
        "kl_thin": 4.0,
        "iter_data": {
            "RMR":      {"thick": 11, "thin": 221, "group": "c0"},
            "MITC4":    {"thick": 6,  "thin": 188, "group": "c0"},
            "MITC4-LP": {"thick": 6,  "thin": 46,  "group": "smoothp"},
            "MITC4-EP": {"thick": 15, "thin": 98,  "group": "smoothp"},
            "MITC9":    {"thick": None, "thin": None, "group": "c0"},
            "MIG2":     {"thick": 5,  "thin": 6,   "group": "c1"},
        },
    },
}

# -------------------------------------------------
# Compute scores
# -------------------------------------------------
def compute_scores(cycle_key):
    cfg = data_by_cycle[cycle_key]
    kl_thick = cfg["kl_thick"]
    kl_thin = cfg["kl_thin"]

    scores = []
    excluded = []

    for name, d in cfg["iter_data"].items():
        thick = d["thick"]
        thin = d["thin"]

        if thick is None or thin is None:
            excluded.append(name)
            continue

        thick_ratio = kl_thick / thick
        thin_ratio = kl_thin / thin
        score = min(thick_ratio, thin_ratio)

        scores.append({
            "name": name,
            "score": score,
            "group": d["group"],
            "thick_ratio": thick_ratio,
            "thin_ratio": thin_ratio,
        })

    scores.sort(key=lambda row: row["score"], reverse=True)
    return scores, excluded


scores_v, excluded_v = compute_scores("V22")
scores_k, excluded_k = compute_scores("K22")

# -------------------------------------------------
# Print results
# -------------------------------------------------
print("\nV(2,2) scores:")
for row in scores_v:
    print(
        f"{row['name']:10s} : {row['score']:.4f}   "
        f"(thick={row['thick_ratio']:.4f}, thin={row['thin_ratio']:.4f})"
    )
if excluded_v:
    print("Excluded from V(2,2):", excluded_v)

print("\nK(2,2) scores:")
for row in scores_k:
    print(
        f"{row['name']:10s} : {row['score']:.4f}   "
        f"(thick={row['thick_ratio']:.4f}, thin={row['thin_ratio']:.4f})"
    )
if excluded_k:
    print("Excluded from K(2,2):", excluded_k)

# -------------------------------------------------
# Colors
# -------------------------------------------------
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map = {
    "c0": cycle[0],
    "smoothp": cycle[1],
    "c1": cycle[2],
}

# -------------------------------------------------
# Plot helper
# -------------------------------------------------
def plot_scores(ax, scores, title, ylabel=False):
    labels = [row["name"] for row in scores]
    values = [row["score"] for row in scores]
    colors = [color_map[row["group"]] for row in scores]
    x = list(range(len(labels)))

    ax.bar(x, values, width=0.62, color=colors)

    kl_line = ax.axhline(
        1.0, linestyle="--", linewidth=2, color="black", zorder=3
    )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    if ylabel:
        ax.set_ylabel(r"Score $S$")
    ax.set_xlabel("Plate element")

    return kl_line


# -------------------------------------------------
# Plot (1x2 with width ratios based on number of bars)
# -------------------------------------------------
nv = max(len(scores_v), 1)
nk = max(len(scores_k), 1)

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 5.8),
    sharey=True,
    gridspec_kw={"width_ratios": [nv, nk]},
)

line_v = plot_scores(axes[0], scores_v, r"$V(2,2)$-ASW", ylabel=True)
line_k = plot_scores(axes[1], scores_k, r"$K(2,2)$-ASW", ylabel=False)

# Make sure the KL reference line at y=1 is visible
all_vals = [row["score"] for row in scores_v] + [row["score"] for row in scores_k]
ymax = max(1.05, 1.12 * max(all_vals)) if all_vals else 1.05
axes[0].set_ylim(0.0, ymax)

# Legend
legend_elements = [
    Patch(facecolor=color_map["c0"], label=r"$C^0$"),
    Patch(facecolor=color_map["smoothp"], label=r"$C^0$-SmoothP"),
    Patch(facecolor=color_map["c1"], label=r"$C^1$"),
]

axes[1].legend(
    handles=legend_elements + [line_k],
    labels=[r"$C^0$", r"$C^0$-SmoothP", r"$C^1$", "KL ref"],
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper right",
)

plt.tight_layout()
plt.savefig("plate_solver_score_combined.png", dpi=200, bbox_inches="tight")
plt.savefig("plate_solver_score_combined.svg", dpi=200, bbox_inches="tight")
plt.show()
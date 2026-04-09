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
# Choose cycle type
# -------------------------------------------------
# cycle_type = "V11"   # options: "V11", "V22"
cycle_type = "V22"   # options: "V11", "V22"

# -------------------------------------------------
# Data for each cycle type
# EB is reference only (not plotted as a bar)
# -------------------------------------------------
data_by_cycle = {
    "V22": {
        "eb_thick": 7.0,
        "eb_thin": 7.0,
        "iter_data": {
            "TSR":     {"thick": 7,  "thin": 49, "group": "c0"},
            "TSR-LP":  {"thick": 4,  "thin": 11, "group": "smoothp"},
            "HRA":     {"thick": 6,  "thin": 24, "group": "c0"},
            "HRA-EP":  {"thick": 13, "thin": 14, "group": "smoothp"},
            "HHR":     {"thick": 4,  "thin": 7,  "group": "c1"},
            "HHD":     {"thick": 7,  "thin": 6,  "group": "c1"},
            "HIGD":    {"thick": 7,  "thin": 7,  "group": "c1"},
            "MIG":    {"thick": 9,  "thin": 7,  "group": "c1"},
            "ASGS":    {"thick": 5,  "thin": 11, "group": "c0"},
        },
    },

    "V11": {
        "eb_thick": 16.0,
        "eb_thin": 16.0,
        "iter_data": {
            "TSR":     {"thick": 9,  "thin": None, "group": "c0"},
            "TSR-LP":  {"thick": 5,  "thin": 25,   "group": "smoothp"},
            "HRA":     {"thick": 6,  "thin": None, "group": "c0"},
            "HRA-EP":  {"thick": 22, "thin": 28,   "group": "smoothp"},
            "HHR":     {"thick": 8,  "thin": 16,   "group": "c1"},
            "HHD":     {"thick": 16, "thin": 16,   "group": "c1"},
            "HIGD":    {"thick": 10, "thin": 10,   "group": "c1"},
            "MIG":    {"thick": 13, "thin": 10,   "group": "c1"},
            "ASGS":    {"thick": 6,  "thin": 41,   "group": "c0"},
        },
    },
}

cfg = data_by_cycle[cycle_type]
eb_thick = cfg["eb_thick"]
eb_thin = cfg["eb_thin"]
iter_data = cfg["iter_data"]

# -------------------------------------------------
# Compute scores
# Exclude methods with no thin-shell convergence
# -------------------------------------------------
scores = []
excluded = []

for name, d in iter_data.items():
    thick = d["thick"]
    thin = d["thin"]

    if thick is None or thin is None:
        excluded.append(name)
        continue

    thick_ratio = eb_thick / thick
    thin_ratio = eb_thin / thin
    score = min(thick_ratio, thin_ratio)

    scores.append((name, score, d["group"], thick_ratio, thin_ratio))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

print(f"\nMin-ratio normalized scores for {cycle_type}:")
for name, score, group, thick_ratio, thin_ratio in scores:
    print(
        f"{name:7s} : {score:.4f}   "
        f"[{group}]   "
        f"(thick={thick_ratio:.4f}, thin={thin_ratio:.4f})"
    )

if excluded:
    print("\nExcluded (no thin-shell convergence):")
    for name in excluded:
        print(f"  {name}")

# -------------------------------------------------
# Colors
# -------------------------------------------------
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map = {
    "c0": cycle[0],
    "smoothp": cycle[1],
    "c1": cycle[2],
}

labels = [row[0] for row in scores]
values = [row[1] for row in scores]
colors = [color_map[row[2]] for row in scores]
x = list(range(len(labels)))

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x, values, width=0.72, color=colors)

# EB baseline
ax.axhline(1.0, linestyle="--", linewidth=2, color="black", label="EB ref")

ax.set_ylabel(r"Score $S$")
ax.set_xlabel("Beam element")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs - 1)

ax.grid(True, axis="y")
ax.set_axisbelow(True)

ax.set_ylim(0.0, 1.1 * max(values))

legend_elements = [
    Patch(facecolor=color_map["c0"], label=r"$C^0$"),
    Patch(facecolor=color_map["smoothp"], label="$C^0$-SmoothP"),
    Patch(facecolor=color_map["c1"], label=r"$C^1$"),
]
ax.legend(
    handles=legend_elements + [ax.lines[0]],
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper right",
)

ax.set_xlim(-0.6, len(labels) - 0.4)

plt.tight_layout()
plt.margins(x=0.02, y=0.04)

plt.savefig(f"beam_element_score_bar_{cycle_type}.png", dpi=200, bbox_inches="tight")
plt.savefig(f"beam_element_score_bar_{cycle_type}.svg", dpi=200, bbox_inches="tight")
plt.show()
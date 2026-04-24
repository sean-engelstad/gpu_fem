import numpy as np
import matplotlib.pyplot as plt
import niceplots
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
fs = 20

plt.rcParams.update({
    "font.size": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 1,
    "ytick.labelsize": fs - 1,
    "legend.fontsize": fs - 3,
})

plt.style.use(niceplots.get_style())

fine_colors   = ["#2E86DE", "#E1B12C", "#00A878"]
coarse_colors = ["#8B0000", "#6A1B9A", "#FF7F0E"]

# ------------------------------------------------------------
# Basis
# ------------------------------------------------------------
def quad_bernstein(xi):
    return np.vstack([
        (1.0 - xi)**2,
        2.0 * xi * (1.0 - xi),
        xi**2,
    ])

def quad_c1_basis(xi):
    B = quad_bernstein(xi)
    N0 = 0.5 * B[0]
    N1 = 0.5 * B[0] + B[1] + 0.5 * B[2]
    N2 = 0.5 * B[2]
    return np.vstack((N0, N1, N2))

# ------------------------------------------------------------
# Interior prolongation blocks
# ------------------------------------------------------------
P_L = np.array([
    [0.75, 0.25, 0.00],
    [0.25, 0.75, 0.00],
    [0.00, 0.75, 0.25],
])

P_R = np.array([
    [0.25, 0.75, 0.00],
    [0.00, 0.75, 0.25],
    [0.00, 0.25, 0.75],
])

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def draw_matrix_text(fig, x, y, M, fontsize=15):
    rows = [
        f"{M[0,0]:.2f} {M[0,1]:.2f} {M[0,2]:.2f}",
        f"{M[1,0]:.2f} {M[1,1]:.2f} {M[1,2]:.2f}",
        f"{M[2,0]:.2f} {M[2,1]:.2f} {M[2,2]:.2f}",
    ]
    txt = (
        "⎡ " + rows[0] + " ⎤\n"
        "⎢ " + rows[1] + " ⎥\n"
        "⎣ " + rows[2] + " ⎦"
    )
    fig.text(
        x, y, txt,
        ha="center", va="center",
        fontsize=fontsize,
        family="monospace",
    )

def add_fig_arrow(fig, start, end, lw=2.5, ms=18):
    arrow = FancyArrowPatch(
        start, end,
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        color="black",
    )
    fig.add_artist(arrow)

# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------
gap = 0.045
xi = np.linspace(0.0, 1.0, 500)
N = quad_c1_basis(xi)

fineL_x = 0.5 * xi
fineR_x = 0.5 + gap + 0.5 * xi

coarse_left = 0.16
coarse_right = 0.88
coarse_mid = 0.5 * (coarse_left + coarse_right)
coarse_x = coarse_left + (coarse_right - coarse_left) * xi

# ------------------------------------------------------------
# Figure and subplots
# ------------------------------------------------------------
fig, (ax_fine, ax_coarse) = plt.subplots(
    2, 1,
    figsize=(8.8, 7.4),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18}
)

# ------------------------------------------------------------
# Fine subplot
# ------------------------------------------------------------
for i in range(3):
    ax_fine.plot(fineL_x, N[i], lw=3.0, color=fine_colors[i], solid_capstyle="round")
    ax_fine.plot(fineR_x, N[i], lw=3.0, color=fine_colors[i], solid_capstyle="round")

for xk in [0.0, 0.5, 0.5 + gap, 1.0 + gap]:
    ax_fine.axvline(xk, color="k", lw=0.7, alpha=0.10, zorder=0)

ax_fine.set_ylim(-0.02, 1.0)
ax_fine.set_yticks([0.0, 0.5, 1.0])
ax_fine.set_ylabel("Value")
ax_fine.grid(True, axis="y", alpha=0.12)
ax_fine.grid(True, axis="x", alpha=0.08)

fine_handles = [
    Line2D([0], [0], color=fine_colors[0], lw=3.0, label=r"$N_0^f(\xi)$"),
    Line2D([0], [0], color=fine_colors[1], lw=3.0, label=r"$N_1^f(\xi)$"),
    Line2D([0], [0], color=fine_colors[2], lw=3.0, label=r"$N_2^f(\xi)$"),
]
ax_fine.legend(
    handles=fine_handles,
    loc="upper right",
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    ncol=1,
)

ax_fine.spines["top"].set_visible(False)
ax_fine.spines["right"].set_visible(False)

# ------------------------------------------------------------
# Coarse subplot
# ------------------------------------------------------------
for i in range(3):
    ax_coarse.plot(coarse_x, N[i], lw=3.2, color=coarse_colors[i], solid_capstyle="round")

for xk in [coarse_left, coarse_mid, coarse_right]:
    ax_coarse.axvline(xk, color="k", lw=0.7, alpha=0.10, zorder=0)

ax_coarse.set_ylim(-0.02, 1.0)
ax_coarse.set_yticks([])
ax_coarse.set_xlabel(r"Parametric coordinate $\xi$")
ax_coarse.grid(True, axis="y", alpha=0.12)
ax_coarse.grid(True, axis="x", alpha=0.08)

coarse_handles = [
    Line2D([0], [0], color=coarse_colors[0], lw=3.2, label=r"$N_0^c(\xi)$"),
    Line2D([0], [0], color=coarse_colors[1], lw=3.2, label=r"$N_1^c(\xi)$"),
    Line2D([0], [0], color=coarse_colors[2], lw=3.2, label=r"$N_2^c(\xi)$"),
]
ax_coarse.legend(
    handles=coarse_handles,
    loc="lower right",
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    ncol=1,
)

ax_coarse.spines["top"].set_visible(False)
ax_coarse.spines["right"].set_visible(False)
ax_coarse.spines["left"].set_visible(False)

# shared x limits/ticks
ax_coarse.set_xlim(-0.02, 1.05 + gap)
ax_coarse.set_xticks([0.0, 0.5, 1.0])
ax_coarse.set_xticklabels(["0", "0.5", "1"])

# ------------------------------------------------------------
# Matrices in figure coordinates
# ------------------------------------------------------------
draw_matrix_text(fig, x=0.13, y=0.5, M=P_L, fontsize=15)
draw_matrix_text(fig, x=0.87, y=0.5, M=P_R, fontsize=15)

# ------------------------------------------------------------
# Arrows in figure coordinates (between subplots)
# ------------------------------------------------------------
add_fig_arrow(fig, start=(0.47, 0.43), end=(0.33, 0.55), lw=2.8, ms=22)
add_fig_arrow(fig, start=(0.57, 0.43), end=(0.67, 0.55), lw=2.8, ms=22)

plt.savefig("iga_clean.png", dpi=220, bbox_inches="tight")
plt.savefig("iga_clean.svg", dpi=220, bbox_inches="tight")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import niceplots
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
fs = 20

plt.rcParams.update({
    "font.size": fs + 1,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 1,
    "ytick.labelsize": fs - 1,
    "legend.fontsize": fs - 2,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.3,
})

plt.style.use(niceplots.get_style())

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def draw_square(ax, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, **kwargs):
    x = [xmin, xmax, xmax, xmin, xmin]
    y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(x, y, **kwargs)

def draw_axes_cross(ax, L=1.0, color="0.86", lw=1.0):
    ax.plot([-L, L], [0, 0], color=color, lw=lw, zorder=0)
    ax.plot([0, 0], [-L, L], color=color, lw=lw, zorder=0)

def scatter_pts(ax, pts, **kwargs):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

def draw_support_cross(ax, center=(0.0, 0.0), h=0.10, color="0.25", lw=1.7):
    x0, y0 = center
    ax.plot([x0 - h, x0 + h], [y0, y0], color=color, lw=lw, zorder=8)
    ax.plot([x0, x0], [y0 - h, y0 + h], color=color, lw=lw, zorder=8)

def connect_support(ax, center, neighbors, color="0.35", lw=1.5):
    x0, y0 = center
    for xn, yn in neighbors:
        ax.plot([x0, xn], [y0, yn], color=color, lw=lw, zorder=7)

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
col_nodes = "black"

# MITC colors (unchanged)
col_t1 = "#C44E52"   # muted red
col_t2 = "#4C956C"   # muted green
col_t3 = "#8172B2"   # muted purple

# Mixed-order IGA colors
col_w  = "#4C78A8"   # muted blue
col_tx = "#D17C59"   # soft burnt orange
col_ty = "#8A7C6D"   # warm gray-brown

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.1))
fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.19, wspace=0.28)

# ============================================================
# 1) MITC4
# ============================================================
ax = axes[0]

draw_square(ax, -1, 1, -1, 1, color="0.72", lw=1.25)
draw_axes_cross(ax, L=1.0, color="0.86", lw=1.0)

mitc4_nodes = np.array([
    [-1, -1], [1, -1], [1, 1], [-1, 1]
])

mitc4_e11g13 = np.array([
    [0.0, -1.0],
    [0.0,  1.0],
])

mitc4_e22g23 = np.array([
    [-1.0, 0.0],
    [ 1.0, 0.0],
])

mitc4_e12 = np.array([
    [0.0, 0.0],
])

scatter_pts(ax, mitc4_nodes, s=155, color=col_nodes, zorder=5)
scatter_pts(ax, mitc4_e11g13, s=155, color=col_t1, marker="s", zorder=6)
scatter_pts(ax, mitc4_e22g23, s=155, color=col_t2, marker="s", zorder=6)
scatter_pts(ax, mitc4_e12,    s=165, color=col_t3, marker="s", zorder=6)

ax.set_aspect("equal")
ax.set_xlim(-1.17, 1.17)
ax.set_ylim(-1.17, 1.17)
ax.axis("off")
ax.set_title("MITC4 element", fontsize=fs)

# ============================================================
# 2) MITC9
# ============================================================
ax = axes[1]

draw_square(ax, -1, 1, -1, 1, color="0.72", lw=1.25)
draw_axes_cross(ax, L=1.0, color="0.86", lw=1.0)

grid = np.array([-1.0, 0.0, 1.0])
mitc9_nodes = np.array([[x, y] for y in grid for x in grid])

a = 0.5770
b = 0.7745

mitc9_e12 = np.array([
    [-a, -a],
    [ a, -a],
    [-a,  a],
    [ a,  a],
])

mitc9_e11g13 = np.array([
    [-a, -b], [ a, -b],
    [-a,  0.0], [ a,  0.0],
    [-a,  b], [ a,  b],
])

mitc9_e22g23 = np.array([
    [-b, -a], [-b,  a],
    [ 0.0, -a], [ 0.0,  a],
    [ b, -a], [ b,  a],
])

scatter_pts(ax, mitc9_nodes, s=135, color=col_nodes, zorder=5)
scatter_pts(ax, mitc9_e11g13, s=135, color=col_t1, marker="s", zorder=6)
scatter_pts(ax, mitc9_e22g23, s=135, color=col_t2, marker="s", zorder=6)
scatter_pts(ax, mitc9_e12,    s=145, color=col_t3, marker="s", zorder=6)

ax.set_aspect("equal")
ax.set_xlim(-1.17, 1.17)
ax.set_ylim(-1.17, 1.17)
ax.axis("off")
ax.set_title("MITC9 element", fontsize=fs)

# shared legend for MITC4 + MITC9
mitc_handles = [
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor=col_nodes, markeredgecolor=col_nodes,
           markersize=10, label="nodes"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor=col_t1, markeredgecolor=col_t1,
           markersize=11, label=r"$e_{11},\,\gamma_{13}$"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor=col_t2, markeredgecolor=col_t2,
           markersize=11, label=r"$e_{22},\,\gamma_{23}$"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor=col_t3, markeredgecolor=col_t3,
           markersize=11, label=r"$e_{12}$"),
]

fig.legend(
    handles=mitc_handles,
    loc="lower center",
    bbox_to_anchor=(0.34, -0.08),
    ncol=2,
    frameon=True,
    columnspacing=1.1,
)

# ============================================================
# 3) Mixed-order IGA2 for RM plate
# ============================================================
ax = axes[2]

draw_square(ax, -1, 1, -1, 1, color="0.72", lw=1.25)
draw_axes_cross(ax, L=1.0, color="0.86", lw=1.0)

# w DOF: 3x3
w_coords = np.array([-1.0, 0.0, 1.0])
w_pts = np.array([[x, y] for y in w_coords for x in w_coords])

# theta_x: 2x3 at x = ±0.5, y = -1, 0, 1
tx_pts = np.array([
    [-0.5, -1.0], [ 0.5, -1.0],
    [-0.5,  0.0], [ 0.5,  0.0],
    [-0.5,  1.0], [ 0.5,  1.0],
])

# theta_y: 3x2 at y = ±0.5, x = -1, 0, 1
ty_pts = np.array([
    [-1.0, -0.5], [ 0.0, -0.5], [ 1.0, -0.5],
    [-1.0,  0.5], [ 0.0,  0.5], [ 1.0,  0.5],
])

# plot DOFs
scatter_pts(ax, w_pts,  s=155, color=col_w, zorder=5)
scatter_pts(ax, tx_pts, s=200, facecolors="none", edgecolors=col_tx,
            linewidths=2.8, marker="o", zorder=6)
scatter_pts(ax, ty_pts, s=200, facecolors="none", edgecolors=col_ty,
            linewidths=2.8, marker="o", zorder=6)

# center support marker + connections
center = (0.0, 0.0)
nearest_support = [
    (-0.5, 0.0), (0.5, 0.0),   # nearest theta_x
    (0.0, -0.5), (0.0, 0.5),   # nearest theta_y
]
connect_support(ax, center, nearest_support, color="0.33", lw=1.55)
draw_support_cross(ax, center=center, h=0.105, color="0.22", lw=1.7)

# optional small labels
ax.text(0.08, 0.14, r"$w$", color=col_w, fontsize=18)
ax.text(0.56, 0.03, r"$\theta_x$", color=col_tx, fontsize=18)
ax.text(0.04, 0.58, r"$\theta_y$", color=col_ty, fontsize=18)

ax.set_aspect("equal")
ax.set_xlim(-1.17, 1.17)
ax.set_ylim(-1.17, 1.17)
ax.axis("off")
ax.set_title("Mixed-order IGA", fontsize=fs)

# separate legend for mixed-order IGA2
iga_handles = [
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor=col_w, markeredgecolor=col_w,
           markersize=12, label=r"$w$"),
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor="none", markeredgecolor=col_tx,
           markeredgewidth=2.5, markersize=12, label=r"$\theta_x$"),
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor="none", markeredgecolor=col_ty,
           markeredgewidth=2.5, markersize=12, label=r"$\theta_y$"),
]

axes[2].legend(
    handles=iga_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=True,
    columnspacing=1.2,
)

plt.savefig("mitc_iga_mixed_order.png", dpi=300, bbox_inches="tight")
plt.savefig("mitc_iga_mixed_order.svg", dpi=300, bbox_inches="tight")
plt.show()
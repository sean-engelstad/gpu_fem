import matplotlib.pyplot as plt
import numpy as np
import niceplots
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
fs = 20

plt.rcParams.update({
    "font.size": fs + 1,
    "axes.labelsize": fs,
    "xtick.labelsize": fs - 1,
    "ytick.labelsize": fs - 1,
    "legend.fontsize": fs - 1,
})

plt.style.use(niceplots.get_style())

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
col_t1 = "#C44E52"   # e11, gamma13
col_t2 = "#4C956C"   # e22, gamma23
col_t3 = "#8172B2"   # e12
col_grid = "0.35"
col_box = "k"
col_arrow = "0.35"
col_nodes = "black"

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def draw_single_element(ax, x0=0.0, y0=0.0, L=1.0, **kwargs):
    x = [x0, x0 + L, x0 + L, x0, x0]
    y = [y0, y0, y0 + L, y0 + L, y0]
    ax.plot(x, y, **kwargs)

def draw_fine_patch(ax, x0=0.0, y0=0.0, L=1.0):
    draw_single_element(ax, x0, y0, L, color=col_box, lw=2.2)
    ax.plot([x0 + 0.5 * L, x0 + 0.5 * L], [y0, y0 + L], "--", color=col_grid, lw=1.4)
    ax.plot([x0, x0 + L], [y0 + 0.5 * L, y0 + 0.5 * L], "--", color=col_grid, lw=1.4)

# -----------------------------------------------------------------------------
# Node locations
# -----------------------------------------------------------------------------
def coarse_nodes_mitc4(x0=0.0, y0=0.0, L=1.0):
    return np.array([
        [x0,     y0],
        [x0 + L, y0],
        [x0 + L, y0 + L],
        [x0,     y0 + L],
    ])

def coarse_nodes_mitc9(x0=0.0, y0=0.0, L=1.0):
    xs = np.linspace(x0, x0 + L, 3)
    ys = np.linspace(y0, y0 + L, 3)
    return np.array([[x, y] for y in ys for x in xs])

def fine_nodes_mitc4_patch(x0=0.0, y0=0.0, L=1.0):
    # 2x2 MITC4 patch -> 3x3 nodal grid
    xs = np.linspace(x0, x0 + L, 3)
    ys = np.linspace(y0, y0 + L, 3)
    return np.array([[x, y] for y in ys for x in xs])

def fine_nodes_mitc9_patch(x0=0.0, y0=0.0, L=1.0):
    # 2x2 MITC9 patch -> 5x5 nodal grid (corners, midsides, centroids)
    xs = np.linspace(x0, x0 + L, 5)
    ys = np.linspace(y0, y0 + L, 5)
    return np.array([[x, y] for y in ys for x in xs])

# -----------------------------------------------------------------------------
# MITC4 tying points
# -----------------------------------------------------------------------------
def coarse_points_mitc4(x0=0.0, y0=0.0, L=1.0):
    pts_t1 = np.array([
        [x0 + 0.5 * L, y0 + 0.0 * L],
        [x0 + 0.5 * L, y0 + 1.0 * L],
    ])

    pts_t2 = np.array([
        [x0 + 0.0 * L, y0 + 0.5 * L],
        [x0 + 1.0 * L, y0 + 0.5 * L],
    ])

    pts_t3 = np.array([
        [x0 + 0.5 * L, y0 + 0.5 * L],
    ])

    return pts_t1, pts_t2, pts_t3

def fine_points_mitc4(x0=0.0, y0=0.0, L=1.0):
    h = 0.5 * L
    pts_t1, pts_t2, pts_t3 = [], [], []

    for j in range(2):
        for i in range(2):
            ex0 = x0 + i * h
            ey0 = y0 + j * h

            pts_t1.extend([
                [ex0 + 0.5 * h, ey0 + 0.0 * h],
                [ex0 + 0.5 * h, ey0 + 1.0 * h],
            ])
            pts_t2.extend([
                [ex0 + 0.0 * h, ey0 + 0.5 * h],
                [ex0 + 1.0 * h, ey0 + 0.5 * h],
            ])
            pts_t3.append([ex0 + 0.5 * h, ey0 + 0.5 * h])

    pts_t1 = np.unique(np.array(pts_t1), axis=0)
    pts_t2 = np.unique(np.array(pts_t2), axis=0)
    pts_t3 = np.unique(np.array(pts_t3), axis=0)
    return pts_t1, pts_t2, pts_t3

# -----------------------------------------------------------------------------
# MITC9 tying points
# -----------------------------------------------------------------------------
def coarse_points_mitc9(x0=0.0, y0=0.0, L=1.0):
    a = 0.5770
    b = 0.7745

    def map_pts(pts):
        pts = np.asarray(pts)
        x = x0 + 0.5 * L * (pts[:, 0] + 1.0)
        y = y0 + 0.5 * L * (pts[:, 1] + 1.0)
        return np.column_stack((x, y))

    pts_t1_local = np.array([
        [-a, -b], [ a, -b],
        [-a,  0.0], [ a,  0.0],
        [-a,  b], [ a,  b],
    ])

    pts_t2_local = np.array([
        [-b, -a], [-b,  a],
        [ 0.0, -a], [ 0.0,  a],
        [ b, -a], [ b,  a],
    ])

    pts_t3_local = np.array([
        [-a, -a],
        [ a, -a],
        [-a,  a],
        [ a,  a],
    ])

    return map_pts(pts_t1_local), map_pts(pts_t2_local), map_pts(pts_t3_local)

def fine_points_mitc9(x0=0.0, y0=0.0, L=1.0):
    h = 0.5 * L
    pts_t1, pts_t2, pts_t3 = [], [], []

    for j in range(2):
        for i in range(2):
            ex0 = x0 + i * h
            ey0 = y0 + j * h
            c1, c2, c3 = coarse_points_mitc9(ex0, ey0, h)
            pts_t1.extend(c1.tolist())
            pts_t2.extend(c2.tolist())
            pts_t3.extend(c3.tolist())

    pts_t1 = np.unique(np.round(np.array(pts_t1), 10), axis=0)
    pts_t2 = np.unique(np.round(np.array(pts_t2), 10), axis=0)
    pts_t3 = np.unique(np.round(np.array(pts_t3), 10), axis=0)
    return pts_t1, pts_t2, pts_t3

# -----------------------------------------------------------------------------
# Row plotting helper
# -----------------------------------------------------------------------------
def plot_transfer_row(ax, coarse_fun, fine_fun, row_label):
    xL, yL, L = 0.0, 0.0, 1.0
    gap = 1.25
    xR, yR = xL + L + gap, 0.0

    # patches
    draw_single_element(ax, xL, yL, L, color=col_box, lw=2.2)
    draw_fine_patch(ax, xR, yR, L)

    # nodes
    if row_label == "MITC4":
        c_nodes = coarse_nodes_mitc4(xL, yL, L)
        f_nodes = fine_nodes_mitc4_patch(xR, yR, L)
    else:
        c_nodes = coarse_nodes_mitc9(xL, yL, L)
        f_nodes = fine_nodes_mitc9_patch(xR, yR, L)

    ax.scatter(c_nodes[:, 0], c_nodes[:, 1], s=90, color=col_nodes, zorder=7)
    ax.scatter(f_nodes[:, 0], f_nodes[:, 1], s=62, color=col_nodes, zorder=7)

    # tying points
    c_t1, c_t2, c_t3 = coarse_fun(xL, yL, L)
    f_t1, f_t2, f_t3 = fine_fun(xR, yR, L)

    ax.scatter(c_t1[:, 0], c_t1[:, 1], s=130, color=col_t1, marker="s", zorder=5)
    ax.scatter(c_t2[:, 0], c_t2[:, 1], s=130, color=col_t2, marker="s", zorder=5)
    ax.scatter(c_t3[:, 0], c_t3[:, 1], s=145, color=col_t3, marker="s", zorder=5)

    ax.scatter(f_t1[:, 0], f_t1[:, 1], s=78, color=col_t1, marker="s", zorder=5)
    ax.scatter(f_t2[:, 0], f_t2[:, 1], s=78, color=col_t2, marker="s", zorder=5)
    ax.scatter(f_t3[:, 0], f_t3[:, 1], s=90, color=col_t3, marker="s", zorder=5)

    # coarse tying points overlaid on fine patch
    cR_t1, cR_t2, cR_t3 = coarse_fun(xR, yR, L)
    ax.scatter(cR_t1[:, 0], cR_t1[:, 1], s=220, facecolors="none", edgecolors=col_t1,
               linewidths=2.2, marker="s", zorder=6)
    ax.scatter(cR_t2[:, 0], cR_t2[:, 1], s=220, facecolors="none", edgecolors=col_t2,
               linewidths=2.2, marker="s", zorder=6)
    ax.scatter(cR_t3[:, 0], cR_t3[:, 1], s=235, facecolors="none", edgecolors=col_t3,
               linewidths=2.2, marker="s", zorder=6)

    # labels
    ax.text(xL + 0.5 * L, 1.08, f"{row_label} coarse", ha="center", va="bottom", fontsize=fs + 1)
    ax.text(xR + 0.5 * L, 1.08, f"{row_label} 2×2 fine", ha="center", va="bottom", fontsize=fs + 1)

    # transfer arrow
    ax.annotate(
        "",
        xy=(xR - 0.18, 0.5),
        xytext=(xL + 1.18, 0.5),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=col_arrow),
    )
    ax.text(
        xL + 1.00 + 0.5 * gap,
        0.56,
        r"intergrid transfer $\mathbf{P}$",
        ha="center",
        va="bottom",
        fontsize=17,
        color="0.25",
    )

        # coarse tying points overlaid on fine patch
    cR_t1, cR_t2, cR_t3 = coarse_fun(xR, yR, L)
    ax.scatter(cR_t1[:, 0], cR_t1[:, 1], s=220, facecolors="none", edgecolors=col_t1,
               linewidths=2.2, marker="s", zorder=6)
    ax.scatter(cR_t2[:, 0], cR_t2[:, 1], s=220, facecolors="none", edgecolors=col_t2,
               linewidths=2.2, marker="s", zorder=6)
    ax.scatter(cR_t3[:, 0], cR_t3[:, 1], s=235, facecolors="none", edgecolors=col_t3,
               linewidths=2.2, marker="s", zorder=6)

    # add callouts ONLY for MITC4 fine patch
    if row_label == "MITC4":
        # coarse tying label: use the bottom red coarse-on-fine point
        px, py = cR_t1[0]
        ax.annotate(
            "coarse tying",
            xy=(px + 0.03, py - 0.03),
            xytext=(px - 0.03, py - 0.12),
            fontsize=16,
            color="0.25",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color="0.25",
            ),
        )

        # fine tying label: use one nearby fine red tying point
        fx, fy = f_t1[1]
        ax.annotate(
            "fine tying",
            xy=(fx + 0.01, fy - 0.01),
            xytext=(fx + 0.02, fy - 0.07),
            fontsize=16,
            color="0.25",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color="0.25",
            ),
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.32, xR + 1.20)
    ax.set_ylim(-0.16, 1.18)
    ax.axis("off")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.0))

plot_transfer_row(
    axes[0],
    coarse_fun=coarse_points_mitc4,
    fine_fun=fine_points_mitc4,
    row_label="MITC4",
)

plot_transfer_row(
    axes[1],
    coarse_fun=coarse_points_mitc9,
    fine_fun=fine_points_mitc9,
    row_label="MITC9",
)

# shared legend
handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=col_nodes, markeredgecolor=col_nodes,
           markersize=8, label="nodes"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor=col_t1, markeredgecolor=col_t1,
           markersize=10, label=r"$e_{11},\,\gamma_{13}$"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor=col_t2, markeredgecolor=col_t2,
           markersize=10, label=r"$e_{22},\,\gamma_{23}$"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor=col_t3, markeredgecolor=col_t3,
           markersize=10, label=r"$e_{12}$"),
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=4,
    frameon=False,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("mitc_coarse_fine_transfer.png", dpi=300, bbox_inches="tight")
plt.savefig("mitc_coarse_fine_transfer.svg", dpi=300, bbox_inches="tight")
# plt.show()
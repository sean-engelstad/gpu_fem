import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def draw_single_element(ax, x0=0.0, y0=0.0, L=1.0, **kwargs):
    x = [x0, x0 + L, x0 + L, x0, x0]
    y = [y0, y0, y0 + L, y0 + L, y0]
    ax.plot(x, y, **kwargs)


def draw_fine_patch(ax, x0=0.0, y0=0.0, L=1.0):
    # outer boundary
    draw_single_element(ax, x0, y0, L, color="k", lw=2.5)

    # interior 2x2 lines
    ax.plot([x0 + 0.5 * L, x0 + 0.5 * L], [y0, y0 + L], "--", color="tab:blue", lw=1.5)
    ax.plot([x0, x0 + L], [y0 + 0.5 * L, y0 + 0.5 * L], "--", color="tab:blue", lw=1.5)


# -----------------------------------------------------------------------------
# Tying points
# -----------------------------------------------------------------------------
def coarse_points(x0=0.0, y0=0.0, L=1.0):
    # y+- edge centroids: gamma_13, e_11
    pts_y = np.array([
        [x0 + 0.5 * L, y0 + 0.0 * L],   # bottom
        [x0 + 0.5 * L, y0 + 1.0 * L],   # top
    ])

    # x+- edge centroids: gamma_23, e_22
    pts_x = np.array([
        [x0 + 0.0 * L, y0 + 0.5 * L],   # left
        [x0 + 1.0 * L, y0 + 0.5 * L],   # right
    ])

    # centroid: gamma_12
    pts_c = np.array([
        [x0 + 0.5 * L, y0 + 0.5 * L],
    ])

    return pts_y, pts_x, pts_c


def fine_points(x0=0.0, y0=0.0, L=1.0):
    h = 0.5 * L

    pts_y = []
    pts_x = []
    pts_c = []

    for j in range(2):
        for i in range(2):
            ex0 = x0 + i * h
            ey0 = y0 + j * h

            # y+- edge centroids: gamma_13, e_11
            pts_y.extend([
                [ex0 + 0.5 * h, ey0 + 0.0 * h],   # bottom
                [ex0 + 0.5 * h, ey0 + 1.0 * h],   # top
            ])

            # x+- edge centroids: gamma_23, e_22
            pts_x.extend([
                [ex0 + 0.0 * h, ey0 + 0.5 * h],   # left
                [ex0 + 1.0 * h, ey0 + 0.5 * h],   # right
            ])

            # centroid: gamma_12
            pts_c.append([ex0 + 0.5 * h, ey0 + 0.5 * h])

    # remove duplicates from shared edges
    pts_y = np.unique(np.array(pts_y), axis=0)
    pts_x = np.unique(np.array(pts_x), axis=0)
    pts_c = np.unique(np.array(pts_c), axis=0)

    return pts_y, pts_x, pts_c


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))

# left coarse element
xL, yL, L = 0.0, 0.0, 1.0
draw_single_element(ax, xL, yL, L, color="k", lw=2.5)

# right fine patch
gap = 1.2
xR, yR = xL + L + gap, 0.0
draw_fine_patch(ax, xR, yR, L)

# points
c_y, c_x, c_c = coarse_points(xL, yL, L)
f_y, f_x, f_c = fine_points(xR, yR, L)

# colors
col_y = "tab:red"      # gamma_13 / e_11
col_x = "tab:green"    # gamma_23 / e_22
col_c = "tab:purple"   # gamma_12

# coarse markers
ax.scatter(c_y[:, 0], c_y[:, 1], s=130, color=col_y, zorder=5, label=r"$\gamma_{13},\,e_{11}$ tying points")
ax.scatter(c_x[:, 0], c_x[:, 1], s=130, color=col_x, zorder=5, label=r"$\gamma_{23},\,e_{22}$ tying points")
ax.scatter(c_c[:, 0], c_c[:, 1], s=150, color=col_c, marker="s", zorder=5, label=r"$\gamma_{12}$ tying points")

# fine markers
ax.scatter(f_y[:, 0], f_y[:, 1], s=90, color=col_y, zorder=5)
ax.scatter(f_x[:, 0], f_x[:, 1], s=90, color=col_x, zorder=5)
ax.scatter(f_c[:, 0], f_c[:, 1], s=110, color=col_c, marker="s", zorder=5)

# titles above each patch
ax.text(xL + 0.5 * L, 1.10, "Coarse element", ha="center", va="bottom", fontsize=18)
ax.text(xR + 0.5 * L, 1.10, "2×2 fine elements", ha="center", va="bottom", fontsize=18)

# annotations for coarse element only
ax.text(xL + 0.58, yL - 0.08, r"$\gamma_{13},\,e_{11}$", color=col_y, fontsize=16, ha="left", va="top")
ax.text(xL + 0.58, yL + 1.03, r"$\gamma_{13},\,e_{11}$", color=col_y, fontsize=16, ha="left", va="bottom")

ax.text(xL - 0.05, yL + 0.58, r"$\gamma_{23},\,e_{22}$", color=col_x, fontsize=16, ha="right", va="bottom")
ax.text(xL + 1.05, yL + 0.58, r"$\gamma_{23},\,e_{22}$", color=col_x, fontsize=16, ha="left", va="bottom")

ax.text(xL + 0.56, yL + 0.56, r"$\gamma_{12}$", color=col_c, fontsize=16, ha="left", va="bottom")

# arrow indicating transfer comparison
ax.annotate(
    "",
    xy=(xR - 0.18, 0.5),
    xytext=(xL + 1.18, 0.5),
    arrowprops=dict(arrowstyle="->", lw=2.0, color="0.35")
)
ax.text(xL + 1.08 + 0.5 * gap, 0.56, r"intergrid transfer $\mathbf{P}$", ha="center", va="bottom", fontsize=14, color="0.25")

# styling
ax.set_aspect("equal")
ax.set_xlim(-0.35, xR + 1.25)
ax.set_ylim(-0.18, 1.22)
ax.axis("off")

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig("mitc_tying_points.svg", dpi=400)
# plt.show()
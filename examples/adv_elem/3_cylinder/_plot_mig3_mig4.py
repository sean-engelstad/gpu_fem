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
    "legend.fontsize": fs - 3,
    "legend.handlelength": 1.1,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.35,
})

plt.style.use(niceplots.get_style())

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rect_grid(xs, ys):
    return np.array([[x, y] for y in ys for x in xs], dtype=float)

def scatter_pts(ax, pts, **kwargs):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

def draw_square(ax, xmin=-1, xmax=1, ymin=-1, ymax=1, **kwargs):
    ax.plot([xmin, xmax], [ymin, ymin], **kwargs)
    ax.plot([xmax, xmax], [ymin, ymax], **kwargs)
    ax.plot([xmax, xmin], [ymax, ymax], **kwargs)
    ax.plot([xmin, xmin], [ymax, ymin], **kwargs)

def draw_center_marker(ax, center=(0.0, 0.0), h=0.07, color="k", lw=1.8):
    x0, y0 = center
    ax.plot([x0 - h, x0 + h], [y0, y0], color=color, lw=lw, zorder=20)
    ax.plot([x0, x0], [y0 - h, y0 + h], color=color, lw=lw, zorder=20)

def connect(ax, center, targets, color="0.2", lw=1.6):
    x0, y0 = center
    for pt in targets:
        if pt is None:
            continue
        x1, y1 = pt
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=10)

def nearest_same_y(points, center, side):
    x0, y0 = center
    pts = np.asarray(points)
    pts = pts[np.isclose(pts[:, 1], y0)]
    if side == "left":
        pts = pts[pts[:, 0] < x0]
        if len(pts) == 0:
            return None
        return tuple(pts[np.argmax(pts[:, 0])])
    else:
        pts = pts[pts[:, 0] > x0]
        if len(pts) == 0:
            return None
        return tuple(pts[np.argmin(pts[:, 0])])

def nearest_same_x(points, center, side):
    x0, y0 = center
    pts = np.asarray(points)
    pts = pts[np.isclose(pts[:, 0], x0)]
    if side == "down":
        pts = pts[pts[:, 1] < y0]
        if len(pts) == 0:
            return None
        return tuple(pts[np.argmax(pts[:, 1])])
    else:
        pts = pts[pts[:, 1] > y0]
        if len(pts) == 0:
            return None
        return tuple(pts[np.argmin(pts[:, 1])])

def midpoint_array(vals):
    vals = np.asarray(vals, dtype=float)
    return 0.5 * (vals[:-1] + vals[1:])

def offset_pts(pts, dx=0.0, dy=0.0):
    pts = np.asarray(pts, dtype=float).copy()
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts

# ------------------------------------------------------------
# Colors and marker settings
# ------------------------------------------------------------
col_u  = "#C44E52"   # red
col_v  = "#4C956C"   # green
col_w  = "black"     # black
col_tx = "#8172B2"   # purple
col_ty = "#8A6D3B"   # brown

ms_sq = 125
ms_w  = 125
edge_lw = 2.3

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
fig.subplots_adjust(left=0.04, right=0.985, top=0.88, bottom=0.22, wspace=0.20)

# ============================================================
# MIG3
# exact embedded coordinates from user
# ============================================================
ax = axes[0]
draw_square(ax, -1, 1, -1, 1, color="0.75", lw=1.2)

u_x  = np.array([-1.0, -1/3,  1/3,  1.0])
u_y  = np.array([-2/3,  0.0,  2/3])

v_x  = np.array([-2/3,  0.0,  2/3])
v_y  = np.array([-1.0, -1/3,  1/3,  1.0])

w_x  = np.array([-2/3,  0.0,  2/3])
w_y  = np.array([-2/3,  0.0,  2/3])

tx_x = np.array([-1/3,  1/3])
tx_y = np.array([-2/3,  0.0,  2/3])

ty_x = np.array([-2/3,  0.0,  2/3])
ty_y = np.array([-1/3,  1/3])

u_pts  = rect_grid(u_x,  u_y)
v_pts  = rect_grid(v_x,  v_y)
w_pts  = rect_grid(w_x,  w_y)
tx_pts = rect_grid(tx_x, tx_y)
ty_pts = rect_grid(ty_x, ty_y)

# offsets:
# u / thx overlap on x-edges -> offset in y
# v / thy overlap on y-edges -> offset in x
u_plot  = offset_pts(u_pts,  dx=0.0,   dy=-0.028)
tx_plot = offset_pts(tx_pts, dx=0.0,   dy= 0.028)

v_plot  = offset_pts(v_pts,  dx=-0.028, dy=0.0)
ty_plot = offset_pts(ty_pts, dx= 0.028, dy=0.0)

scatter_pts(ax, u_plot,  s=ms_sq, color=col_u, marker="s", zorder=3)
scatter_pts(ax, v_plot,  s=ms_sq, color=col_v, marker="s", zorder=3)
scatter_pts(ax, w_pts,   s=ms_w,  color=col_w, marker="o", zorder=5)

scatter_pts(ax, tx_plot, s=ms_sq, facecolors="none", edgecolors=col_tx,
            linewidths=edge_lw, marker="s", zorder=4)
scatter_pts(ax, ty_plot, s=ms_sq, facecolors="none", edgecolors=col_ty,
            linewidths=edge_lw, marker="s", zorder=4)

center = (0.0, 0.0)

connect(ax, center, [
    nearest_same_y(u_pts, center, "left"),
    nearest_same_y(u_pts, center, "right"),
], color=col_u, lw=1.7)

connect(ax, center, [
    nearest_same_x(v_pts, center, "down"),
    nearest_same_x(v_pts, center, "up"),
], color=col_v, lw=1.7)

connect(ax, center, [
    nearest_same_y(tx_pts, center, "left"),
    nearest_same_y(tx_pts, center, "right"),
], color=col_tx, lw=1.7)

connect(ax, center, [
    nearest_same_x(ty_pts, center, "down"),
    nearest_same_x(ty_pts, center, "up"),
], color=col_ty, lw=1.7)

draw_center_marker(ax, center=center, h=0.075, color="k", lw=1.8)

ax.text(0.04, 0.04, r"$w$", color=col_w, fontsize=18)
# ax.text(0.50, -0.86, r"$u$", color=col_u, fontsize=18)
# ax.text(-0.90, 0.48, r"$v$", color=col_v, fontsize=18)
# ax.text(0.50, 0.03, r"$\theta_x$", color=col_tx, fontsize=18)
# ax.text(0.03, 0.52, r"$\theta_y$", color=col_ty, fontsize=18)
# central w already labeled

# (u, θx) pair to the RIGHT of center → offset in y (vertical split)
# (u, θx) pair to the RIGHT of center → shift right + vertical split
pt_right = nearest_same_y(u_pts, center, "right")
if pt_right is not None:
    x0, y0 = pt_right

    dx = 0.06      # push label to the right
    dy = 0.028     # SAME as your DOF offset

    # u (red) slightly below
    ax.text(x0 + dx + 0.02, y0 - dy + 0.02, r"$u$",
            color=col_u, fontsize=16, ha="left", va="top")

    # θx (purple) slightly above
    ax.text(x0 + dx + 0.01, y0 + dy - 0.02, r"$\theta_x$",
            color=col_tx, fontsize=16, ha="left", va="bottom")


# (v, θy) pair ABOVE center → shift up + horizontal split
pt_up = nearest_same_x(v_pts, center, "up")
if pt_up is not None:
    x0, y0 = pt_up

    dx = 0.028     # SAME as DOF offset
    dy = 0.06      # push label upward

    # v (green) slightly left
    ax.text(x0 - dx, y0 + dy + 0.02, r"$v$",
            color=col_v, fontsize=16, ha="right", va="bottom")

    # θy (brown) slightly right
    ax.text(x0 + dx, y0 + dy, r"$\theta_y$",
            color=col_ty, fontsize=16, ha="left", va="bottom")


ax.set_title("MIG3 element", fontsize=fs)
ax.set_aspect("equal")
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-1.12, 1.12)
ax.axis("off")

# ============================================================
# MIG4
# highest-order evenly spaced, then embedded lower-order fields
# ============================================================
ax = axes[1]
draw_square(ax, -1, 1, -1, 1, color="0.75", lw=1.2)

# (u,v,w) = (2x4, 3x3, 3x2), (thx,thy) = (2x2, 3x1)

# highest order in x: p=3 -> 4 points
# highest order in y: p=4 -> 5 points
x4 = np.linspace(-1, 1, 4)   # p=3
x3 = midpoint_array(x4)      # p=2

y5 = np.linspace(-1, 1, 5)   # p=4
y4 = midpoint_array(y5)      # p=3
y3 = midpoint_array(y4)      # p=2

u_x  = x3          # p=2
u_y  = y5          # p=4

v_x  = x4          # p=3
v_y  = y4          # p=3

w_x  = x4          # p=3
w_y  = y3          # p=2

tx_x = midpoint_array(w_x)   # x-edge midpoints of w
tx_y = w_y

ty_x = w_x
ty_y = midpoint_array(w_y)   # y-edge midpoints of w

u_pts  = rect_grid(u_x,  u_y)
v_pts  = rect_grid(v_x,  v_y)
w_pts  = rect_grid(w_x,  w_y)
tx_pts = rect_grid(tx_x, tx_y)
ty_pts = rect_grid(ty_x, ty_y)

# offsets:
# u / thx overlap on x-edges -> offset in y
# v / thy overlap on y-edges -> offset in x
u_plot  = offset_pts(u_pts,  dx=0.0,   dy=-0.024)
tx_plot = offset_pts(tx_pts, dx=0.0,   dy= 0.024)

v_plot  = offset_pts(v_pts,  dx=-0.024, dy=0.0)
ty_plot = offset_pts(ty_pts, dx= 0.024, dy=0.0)

scatter_pts(ax, u_plot,  s=ms_sq, color=col_u, marker="s", zorder=3)
scatter_pts(ax, v_plot,  s=ms_sq, color=col_v, marker="s", zorder=3)
scatter_pts(ax, w_pts,   s=ms_w,  color=col_w, marker="o", zorder=5)

scatter_pts(ax, tx_plot, s=ms_sq, facecolors="none", edgecolors=col_tx,
            linewidths=edge_lw, marker="s", zorder=4)
scatter_pts(ax, ty_plot, s=ms_sq, facecolors="none", edgecolors=col_ty,
            linewidths=edge_lw, marker="s", zorder=4)

center = tuple(w_pts[np.argmin(np.sum(w_pts**2, axis=1))])

connect(ax, center, [
    nearest_same_y(u_pts, center, "left"),
    nearest_same_y(u_pts, center, "right"),
], color=col_u, lw=1.7)

connect(ax, center, [
    nearest_same_x(v_pts, center, "down"),
    nearest_same_x(v_pts, center, "up"),
], color=col_v, lw=1.7)

connect(ax, center, [
    nearest_same_y(tx_pts, center, "left"),
    nearest_same_y(tx_pts, center, "right"),
], color=col_tx, lw=1.7)

connect(ax, center, [
    nearest_same_x(ty_pts, center, "down"),
    nearest_same_x(ty_pts, center, "up"),
], color=col_ty, lw=1.7)

draw_center_marker(ax, center=center, h=0.072, color="k", lw=1.8)

ax.text(center[0] + 0.04, center[1] + 0.04, r"$w$", color=col_w, fontsize=18)
# ax.text(0.52, -0.88, r"$u$", color=col_u, fontsize=18)
# ax.text(-0.92, 0.58, r"$v$", color=col_v, fontsize=18)
# ax.text(0.16, 0.06, r"$\theta_x$", color=col_tx, fontsize=18)
# ax.text(-0.78, -0.24, r"$\theta_y$", color=col_ty, fontsize=18)
# central w already labeled

# (u, θx) pair to the RIGHT of center → shift right + vertical split
pt_right = nearest_same_y(u_pts, center, "right")
if pt_right is not None:
    x0, y0 = pt_right

    dx = 0.06      # push label to the right
    dy = 0.028     # SAME as your DOF offset

    # u (red) slightly below
    ax.text(x0 + dx + 0.02, y0 - dy + 0.02, r"$u$",
            color=col_u, fontsize=16, ha="left", va="top")

    # θx (purple) slightly above
    ax.text(x0 + dx + 0.01, y0 + dy - 0.02, r"$\theta_x$",
            color=col_tx, fontsize=16, ha="left", va="bottom")


# (v, θy) pair ABOVE center → shift up + horizontal split
pt_up = nearest_same_x(v_pts, center, "up")
if pt_up is not None:
    x0, y0 = pt_up

    dx = 0.028     # SAME as DOF offset
    dy = 0.06      # push label upward

    # v (green) slightly left
    ax.text(x0 - dx, y0 + dy + 0.02, r"$v$",
            color=col_v, fontsize=16, ha="right", va="bottom")

    # θy (brown) slightly right
    ax.text(x0 + dx, y0 + dy, r"$\theta_y$",
            color=col_ty, fontsize=16, ha="left", va="bottom")


ax.set_title("MIG4 element", fontsize=fs)
ax.set_aspect("equal")
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-1.12, 1.12)
ax.axis("off")

# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------
handles = [
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor=col_u, markeredgecolor=col_u,
           markersize=10, label=r"$u$"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor=col_v, markeredgecolor=col_v,
           markersize=10, label=r"$v$"),
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor=col_w, markeredgecolor=col_w,
           markersize=9, label=r"$w$"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor="none", markeredgecolor=col_tx,
           markeredgewidth=2.2, markersize=10, label=r"$\theta_x$"),
    Line2D([0], [0], marker="s", linestyle="None",
           markerfacecolor="none", markeredgecolor=col_ty,
           markeredgewidth=2.2, markersize=10, label=r"$\theta_y$"),
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=5,
    frameon=True,
    columnspacing=1.25,
)

plt.savefig("mig3_mig4_one_element_embedded_v2.png", dpi=300, bbox_inches="tight")
plt.savefig("mig3_mig4_one_element_embedded_v2.svg", dpi=300, bbox_inches="tight")
plt.show()
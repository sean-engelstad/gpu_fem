import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-14:
        return v
    return v / n

def add_vector(
    ax,
    p0,
    v,
    label=None,
    color="k",
    lw=3.0,
    scale=1.0,
    text_shift=None,
    fontsize=16,
    arrow_length_ratio=0.14,
):
    """
    Draw a 3D vector from p0 in direction v.
    """
    p0 = np.asarray(p0)
    v = np.asarray(v) * scale

    ax.quiver(
        p0[0], p0[1], p0[2],
        v[0], v[1], v[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=arrow_length_ratio
    )

    if label is not None:
        pend = p0 + v
        if text_shift is None:
            text_shift = np.array([0.02, 0.02, 0.02])
        ax.text(
            pend[0] + text_shift[0],
            pend[1] + text_shift[1],
            pend[2] + text_shift[2],
            label,
            color=color,
            fontsize=fontsize
        )

def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# ------------------------------------------------------------
# Undeformed midsurface X(xi1, xi2)
# ------------------------------------------------------------
def X_map(xi1, xi2):
    """
    Mildly curved undeformed shell midsurface.
    """
    z = 0.18 * xi1**2 + 0.08 * xi2**2 + 0.03 * xi1 * xi2
    return np.array([xi1, xi2, z])

def dX_dxi1(xi1, xi2):
    return np.array([1.0, 0.0, 2.0 * 0.18 * xi1 + 0.03 * xi2])

def dX_dxi2(xi1, xi2):
    return np.array([0.0, 1.0, 2.0 * 0.08 * xi2 + 0.03 * xi1])

# ------------------------------------------------------------
# Displacement field u(xi1, xi2)
# ------------------------------------------------------------
def u_map(xi1, xi2):
    """
    Example displacement field for visualization.
    """
    u1 = 0.10 * xi1
    u2 = -0.05 * xi2 + 0.03 * xi1 * xi2
    u3 = 0.18 * np.sin(0.9 * np.pi * xi1) * np.sin(0.8 * np.pi * xi2)
    return np.array([u1, u2, u3])

def du_dxi1(xi1, xi2):
    du1 = 0.10
    du2 = 0.03 * xi2
    du3 = 0.18 * (0.9 * np.pi) * np.cos(0.9 * np.pi * xi1) * np.sin(0.8 * np.pi * xi2)
    return np.array([du1, du2, du3])

def du_dxi2(xi1, xi2):
    du1 = 0.0
    du2 = -0.05 + 0.03 * xi1
    du3 = 0.18 * np.sin(0.9 * np.pi * xi1) * (0.8 * np.pi) * np.cos(0.8 * np.pi * xi2)
    return np.array([du1, du2, du3])

# ------------------------------------------------------------
# Deformed midsurface x(xi1, xi2) = X + u
# ------------------------------------------------------------
def x_map(xi1, xi2):
    return X_map(xi1, xi2) + u_map(xi1, xi2)

def dx_dxi1(xi1, xi2):
    return dX_dxi1(xi1, xi2) + du_dxi1(xi1, xi2)

def dx_dxi2(xi1, xi2):
    return dX_dxi2(xi1, xi2) + du_dxi2(xi1, xi2)

# ------------------------------------------------------------
# Surface plotting
# ------------------------------------------------------------
def plot_surface(ax, surf_fun, color_surface="lightgray", alpha=0.5):
    n = 51
    xi1_vals = np.linspace(-1.0, 1.0, n)
    xi2_vals = np.linspace(-1.0, 1.0, n)

    XX = np.zeros((n, n))
    YY = np.zeros((n, n))
    ZZ = np.zeros((n, n))

    for i, xi1 in enumerate(xi1_vals):
        for j, xi2 in enumerate(xi2_vals):
            p = surf_fun(xi1, xi2)
            XX[j, i] = p[0]
            YY[j, i] = p[1]
            ZZ[j, i] = p[2]

    ax.plot_surface(
        XX, YY, ZZ,
        rstride=1, cstride=1,
        color=color_surface,
        edgecolor="none",
        alpha=alpha,
        shade=True
    )

def plot_multiple_isolines(
    ax,
    surf_fun,
    xi1_fixed_vals=None,
    xi2_fixed_vals=None,
    color_xi1="tab:blue",
    color_xi2="tab:orange",
    lw=1.5,
    ls="--",
):
    """
    Plot multiple xi^1 and xi^2 isolines.
    xi1_fixed_vals: values of xi1 with xi2 varying
    xi2_fixed_vals: values of xi2 with xi1 varying
    """
    if xi1_fixed_vals is None:
        xi1_fixed_vals = [-0.75, -0.35, 0.0, 0.35, 0.75]
    if xi2_fixed_vals is None:
        xi2_fixed_vals = [-0.75, -0.35, 0.0, 0.35, 0.75]

    s = np.linspace(-1.0, 1.0, 300)

    # xi1 = const, xi2 varies
    for c in xi1_fixed_vals:
        pts = np.array([surf_fun(c, si) for si in s])
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            linestyle=ls, color=color_xi1, lw=lw, alpha=0.95
        )

    # xi2 = const, xi1 varies
    for c in xi2_fixed_vals:
        pts = np.array([surf_fun(si, c) for si in s])
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            linestyle=ls, color=color_xi2, lw=lw, alpha=0.95
        )

# ------------------------------------------------------------
# Main figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Representative parametric point: centered in the patch
xi1p = 0.0
xi2p = 0.0

# Surface points
X0 = X_map(xi1p, xi2p)
x0 = x_map(xi1p, xi2p)
u0 = u_map(xi1p, xi2p)

# Basis vectors
A1 = dX_dxi1(xi1p, xi2p)
A2 = dX_dxi2(xi1p, xi2p)
A3 = normalize(np.cross(A1, A2))

a1 = dx_dxi1(xi1p, xi2p)
a2 = dx_dxi2(xi1p, xi2p)
a3 = normalize(np.cross(a1, a2))

# Normalize tangents for display
basis_scale = 0.50
A1p = normalize(A1)
A2p = normalize(A2)
a1p = normalize(a1)
a2p = normalize(a2)

# Normal arrows shorter than xi^3 dashed guide lines
normal_arrow_scale = 0.42
xi3_scale = 0.90

# Plot surfaces
plot_surface(ax1, X_map, color_surface="lightsteelblue", alpha=0.55)
plot_surface(ax2, x_map, color_surface="lightcoral", alpha=0.50)

# Plot multiple dashed isolines
plot_multiple_isolines(ax1, X_map)
plot_multiple_isolines(ax2, x_map)

# Mark representative points
ax1.scatter(X0[0], X0[1], X0[2], color="k", s=48, zorder=10)
ax2.scatter(x0[0], x0[1], x0[2], color="k", s=48, zorder=10)

# ------------------------------------------------------------
# Dashed xi^3 lines (longer and separate from A3 / a3)
# ------------------------------------------------------------
ax1.plot(
    [X0[0], X0[0] + xi3_scale * A3[0]],
    [X0[1], X0[1] + xi3_scale * A3[1]],
    [X0[2], X0[2] + xi3_scale * A3[2]],
    color="tab:green",
    lw=1.8,
    linestyle="--"
)
ax1.text(
    X0[0] + 1.10 * xi3_scale * A3[0] + 0.02,
    X0[1] + 1.10 * xi3_scale * A3[1] + 0.01,
    X0[2] + 1.10 * xi3_scale * A3[2] + 0.03,
    r"$\xi^3$",
    color="tab:green",
    fontsize=16
)

ax2.plot(
    [x0[0], x0[0] + xi3_scale * a3[0]],
    [x0[1], x0[1] + xi3_scale * a3[1]],
    [x0[2], x0[2] + xi3_scale * a3[2]],
    color="tab:green",
    lw=1.8,
    linestyle="--"
)
ax2.text(
    x0[0] + 1.10 * xi3_scale * a3[0] + 0.02,
    x0[1] + 1.10 * xi3_scale * a3[1] + 0.01,
    x0[2] + 1.10 * xi3_scale * a3[2] + 0.03,
    r"$\xi^3$",
    color="tab:green",
    fontsize=16
)

# ------------------------------------------------------------
# Labels for xi^1 and xi^2
# Put them on selected isolines away from the center
# ------------------------------------------------------------
p_xi1_u = X_map(0.75, 0.0)   # on xi2 = 0 line
p_xi2_u = X_map(0.0, -0.75)  # on xi1 = 0 line
ax1.text(
    p_xi1_u[0] - 0.02, p_xi1_u[1] + 0.04, p_xi1_u[2] + 0.04,
    r"$\xi^1$", color="tab:orange", fontsize=16
)
ax1.text(
    p_xi2_u[0] - 0.02, p_xi2_u[1] + 0.02, p_xi2_u[2] + 0.08,
    r"$\xi^2$", color="tab:blue", fontsize=16
)

p_xi1_d = x_map(0.75, 0.0)
p_xi2_d = x_map(0.0, -0.75)
ax2.text(
    p_xi1_d[0] + 0.1, p_xi1_d[1] + 0.04, p_xi1_d[2] + 0.04,
    r"$\xi^1$", color="tab:orange", fontsize=16
)
ax2.text(
    p_xi2_d[0] - 0.02, p_xi2_d[1] + 0.02, p_xi2_d[2] + 0.08,
    r"$\xi^2$", color="tab:blue", fontsize=16
)

# ------------------------------------------------------------
# Basis vectors: stand out more
# ------------------------------------------------------------
add_vector(
    ax1, X0, A1p,
    label=r"$\mathbf{A}_1$",
    color="purple", scale=basis_scale, lw=4.0,
    text_shift=np.array([0.03, 0.01, 0.02]), fontsize=17
)
add_vector(
    ax1, X0, A2p,
    label=r"$\mathbf{A}_2$",
    color="darkorange", scale=basis_scale, lw=4.0,
    text_shift=np.array([0.01, 0.03, 0.02]), fontsize=17
)
add_vector(
    ax1, X0, A3,
    label=r"$\mathbf{A}_3$",
    color="forestgreen", scale=normal_arrow_scale, lw=4.0,
    text_shift=np.array([0.02, 0.01, -0.03]), fontsize=17
)

add_vector(
    ax2, x0, a1p,
    label=r"$\mathbf{a}_1$",
    color="purple", scale=basis_scale, lw=4.0,
    text_shift=np.array([0.03, 0.01, 0.02]), fontsize=17
)
add_vector(
    ax2, x0, a2p,
    label=r"$\mathbf{a}_2$",
    color="darkorange", scale=basis_scale, lw=4.0,
    text_shift=np.array([0.01, 0.03, 0.02]), fontsize=17
)
add_vector(
    ax2, x0, a3,
    label=r"$\mathbf{a}_3$",
    color="forestgreen", scale=normal_arrow_scale, lw=4.0,
    text_shift=np.array([0.02, 0.01, -0.03]), fontsize=17
)

# ------------------------------------------------------------
# Position vectors X and x from origin
# ------------------------------------------------------------
origin = np.zeros(3)

add_vector(
    ax1, origin, X0,
    label=r"$\mathbf{X}$",
    color="black", lw=2.5, scale=1.0,
    text_shift=np.array([-0.12, -0.05, 0.08]),
    fontsize=16
)
add_vector(
    ax2, origin, x0,
    label=r"$\mathbf{x}$",
    color="black", lw=2.5, scale=1.0,
    text_shift=np.array([-0.12, -0.05, 0.03]),
    fontsize=16
)

# ------------------------------------------------------------
# Displacement vector u = x - X
# ------------------------------------------------------------
ax2.scatter(X0[0], X0[1], X0[2], color="gray", s=24, alpha=0.9)
ax2.plot(
    [X0[0], x0[0]],
    [X0[1], x0[1]],
    [X0[2], x0[2]],
    color="red",
    lw=1.4,
    linestyle="--",
    alpha=0.8
)
# add_vector(
#     ax2, X0, u0,
#     label=r"$\mathbf{u}$",
#     color="red", lw=3.0, scale=1.0,
#     text_shift=np.array([0.03, 0.03, 0.03]),
#     fontsize=16
# )

# ------------------------------------------------------------
# Cosmetics
# ------------------------------------------------------------
for ax in [ax1, ax2]:
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.set_zlabel(r"$X_3$", fontsize=14)
    ax.view_init(elev=26, azim=-55)
    ax.grid(False)
    set_axes_equal(ax)

# ------------------------------------------------------------
# Cosmetics
# ------------------------------------------------------------
for ax in [ax1, ax2]:
    ax.view_init(elev=26, azim=-55)
    ax.grid(False)
    set_axes_equal(ax)
    ax.set_axis_off()

# ax1.set_title("Undeformed shell", fontsize=17, pad=16)
# ax2.set_title("Deformed shell", fontsize=17, pad=16)

plt.tight_layout()
plt.savefig("_shell_theory.png", dpi=400)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "figure.figsize": (12.5, 3.4),
    "lines.linewidth": 2.0,
})

fig, axes = plt.subplots(1, 2)
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# ============================================================
# 1) TRANSVERSE SHEAR LOCKING
# one beam element:
# dashed = no transverse shear
# solid  = exaggerated transverse shear distortion
# ============================================================
# ax = axes[0]
# ax.set_title("Transverse shear locking")

# # flatter beam centerline
# x = np.linspace(0.0, 3.2, 400)
# y = 0.22 * (1.0 - ((x - 1.6) / 1.6) ** 2)

# # tangent / normal
# dy = np.gradient(y, x)
# L = np.sqrt(1.0 + dy**2)
# tx = 1.0 / L
# ty = dy / L
# nx = -ty
# ny = tx

# t = 0.34

# # dashed: true/no-shear beam
# x_top_d = x + 0.5 * t * nx
# y_top_d = y + 0.5 * t * ny
# x_bot_d = x - 0.5 * t * nx
# y_bot_d = y - 0.5 * t * ny

# # solid: locked beam with strong transverse shear distortion
# s = (x - x.min()) / (x.max() - x.min())
# shear = 0.26 * (2.0 * s - 1.0)

# x_top_s = x + 0.5 * t * nx + shear * tx
# y_top_s = y + 0.5 * t * ny + shear * ty
# x_bot_s = x - 0.5 * t * nx - shear * tx
# y_bot_s = y - 0.5 * t * ny - shear * ty

# ax.plot(x_top_d, y_top_d, "k--", lw=1.7)
# ax.plot(x_bot_d, y_bot_d, "k--", lw=1.7)

# solid_poly = np.column_stack([x_top_s, y_top_s]).tolist() + \
#              np.column_stack([x_bot_s[::-1], y_bot_s[::-1]]).tolist()
# ax.add_patch(Polygon(solid_poly, closed=True, facecolor="0.88", edgecolor="none"))

# ax.plot(x_top_s, y_top_s, "k-", lw=2.2)
# ax.plot(x_bot_s, y_bot_s, "k-", lw=2.2)

# for xp in [x[0], x[len(x)//2], x[-1]]:
#     yp = np.interp(xp, x, y)
#     mp = np.interp(xp, x, dy)
#     lp = np.sqrt(1.0 + mp**2)
#     txp, typ = 1.0 / lp, mp / lp
#     nxp, nyp = -typ, txp

#     p1 = np.array([xp - 0.5 * t * nxp, yp - 0.5 * t * nyp])
#     p2 = np.array([xp + 0.5 * t * nxp, yp + 0.5 * t * nyp])
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.4)

# for xp in [x[0], x[len(x)//2], x[-1]]:
#     yp = np.interp(xp, x, y)
#     mp = np.interp(xp, x, dy)
#     lp = np.sqrt(1.0 + mp**2)
#     txp, typ = 1.0 / lp, mp / lp
#     nxp, nyp = -typ, txp
#     sp = 0.26 * (2.0 * ((xp - x.min()) / (x.max() - x.min())) - 1.0)

#     p1 = np.array([xp - 0.5 * t * nxp - sp * txp, yp - 0.5 * t * nyp - sp * typ])
#     p2 = np.array([xp + 0.5 * t * nxp + sp * txp, yp + 0.5 * t * nyp + sp * typ])
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=1.7)

# ax.set_xlim(-0.35, 3.55)
# ax.set_ylim(y_bot_s.min() - 0.08, y_top_s.max() + 0.10)

# ax = axes[0]
# ax.set_title("Transverse shear locking")

# x = np.linspace(0.0, 3.2, 400)

# # dashed: larger bending deflection
# y_d = 0.22 * (1.0 - ((x - 1.6) / 1.6) ** 2)

# # solid: smaller bending deflection due to locking
# y_s = 0.16 * (1.0 - ((x - 1.6) / 1.6) ** 2)

# # dashed tangent / normal
# dy_d = np.gradient(y_d, x)
# L_d = np.sqrt(1.0 + dy_d**2)
# tx_d = 1.0 / L_d
# ty_d = dy_d / L_d
# nx_d = -ty_d
# ny_d = tx_d

# # solid tangent / normal
# dy_s = np.gradient(y_s, x)
# L_s = np.sqrt(1.0 + dy_s**2)
# tx_s = 1.0 / L_s
# ty_s = dy_s / L_s
# nx_s = -ty_s
# ny_s = tx_s

# t = 0.34

# # dashed: true/no-shear beam
# x_top_d = x + 0.5 * t * nx_d
# y_top_d = y_d + 0.5 * t * ny_d
# x_bot_d = x - 0.5 * t * nx_d
# y_bot_d = y_d - 0.5 * t * ny_d

# # solid: locked beam with smaller bending deflection + shear distortion
# s = (x - x.min()) / (x.max() - x.min())
# shear = 0.16 * (2.0 * s - 1.0)

# x_top_s = x + 0.5 * t * nx_s + shear * tx_s
# y_top_s = y_s + 0.5 * t * ny_s + shear * ty_s
# x_bot_s = x - 0.5 * t * nx_s - shear * tx_s
# y_bot_s = y_s - 0.5 * t * ny_s - shear * ty_s

# ax.plot(x_top_d, y_top_d, "k--", lw=1.7)
# ax.plot(x_bot_d, y_bot_d, "k--", lw=1.7)

# solid_poly = np.column_stack([x_top_s, y_top_s]).tolist() + \
#              np.column_stack([x_bot_s[::-1], y_bot_s[::-1]]).tolist()
# ax.add_patch(Polygon(solid_poly, closed=True, facecolor="0.88", edgecolor="none"))

# ax.plot(x_top_s, y_top_s, "k-", lw=2.2)
# ax.plot(x_bot_s, y_bot_s, "k-", lw=2.2)

# # dashed sections
# for xp in [x[0], x[len(x)//2], x[-1]]:
#     yp = np.interp(xp, x, y_d)
#     mp = np.interp(xp, x, dy_d)
#     lp = np.sqrt(1.0 + mp**2)
#     txp, typ = 1.0 / lp, mp / lp
#     nxp, nyp = -typ, txp

#     p1 = np.array([xp - 0.5 * t * nxp, yp - 0.5 * t * nyp])
#     p2 = np.array([xp + 0.5 * t * nxp, yp + 0.5 * t * nyp])
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.4)

# # solid sections
# for xp in [x[0], x[len(x)//2], x[-1]]:
#     yp = np.interp(xp, x, y_s)
#     mp = np.interp(xp, x, dy_s)
#     lp = np.sqrt(1.0 + mp**2)
#     txp, typ = 1.0 / lp, mp / lp
#     nxp, nyp = -typ, txp
#     sp = 0.16 * (2.0 * ((xp - x.min()) / (x.max() - x.min())) - 1.0)

#     p1 = np.array([xp - 0.5 * t * nxp - sp * txp, yp - 0.5 * t * nyp - sp * typ])
#     p2 = np.array([xp + 0.5 * t * nxp + sp * txp, yp + 0.5 * t * nyp + sp * typ])
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=1.7)

# ax.set_xlim(-0.35, 3.55)
# ax.set_ylim(min(y_bot_d.min(), y_bot_s.min()) - 0.08,
#             max(y_top_d.max(), y_top_s.max()) + 0.10)

ax = axes[0]
ax.set_title("Transverse shear locking")

x = np.linspace(0.0, 3.2, 400)

# cantilever beam: fixed at left, bending upward to the right
# dashed = larger true deflection
# solid  = smaller locked deflection
A_d = 0.055
A_s = 0.035
y_d = A_d * x**2
y_s = A_s * x**2

# dashed tangent / normal
dy_d = np.gradient(y_d, x)
L_d = np.sqrt(1.0 + dy_d**2)
tx_d = 1.0 / L_d
ty_d = dy_d / L_d
nx_d = -ty_d
ny_d = tx_d

# solid tangent / normal
dy_s = np.gradient(y_s, x)
L_s = np.sqrt(1.0 + dy_s**2)
tx_s = 1.0 / L_s
ty_s = dy_s / L_s
nx_s = -ty_s
ny_s = tx_s

t = 0.34

# dashed: true/no-shear beam
x_top_d = x + 0.5 * t * nx_d
y_top_d = y_d + 0.5 * t * ny_d
x_bot_d = x - 0.5 * t * nx_d
y_bot_d = y_d - 0.5 * t * ny_d

# solid: locked beam with smaller bending deflection + transverse shear distortion
s = (x - x.min()) / (x.max() - x.min())
shear = 0.12 * s   # zero at the fixed end, grows toward the free end

x_top_s = x + 0.5 * t * nx_s + shear * tx_s
y_top_s = y_s + 0.5 * t * ny_s + shear * ty_s
x_bot_s = x - 0.5 * t * nx_s - shear * tx_s
y_bot_s = y_s - 0.5 * t * ny_s - shear * ty_s

# dashed beam outline
# ax.plot(x_top_d, y_top_d, "k--", lw=1.7)
# ax.plot(x_bot_d, y_bot_d, "k--", lw=1.7)
# dashed beam outline
ax.plot(x_top_d, y_top_d, "k--", lw=1.7, alpha=0.55)
ax.plot(x_bot_d, y_bot_d, "k--", lw=1.7, alpha=0.55)

# solid filled beam
solid_poly = np.column_stack([x_top_s, y_top_s]).tolist() + \
             np.column_stack([x_bot_s[::-1], y_bot_s[::-1]]).tolist()
ax.add_patch(Polygon(solid_poly, closed=True, facecolor="0.88", edgecolor="none"))

# solid beam outline
ax.plot(x_top_s, y_top_s, "k-", lw=2.2)
ax.plot(x_bot_s, y_bot_s, "k-", lw=2.2)

# section lines: fixed end, middle, free end
for xp in [x[0], x[len(x)//2], x[-1]]:
    # dashed section
    yp = np.interp(xp, x, y_d)
    mp = np.interp(xp, x, dy_d)
    lp = np.sqrt(1.0 + mp**2)
    txp, typ = 1.0 / lp, mp / lp
    nxp, nyp = -typ, txp
    p1 = np.array([xp - 0.5 * t * nxp, yp - 0.5 * t * nyp])
    p2 = np.array([xp + 0.5 * t * nxp, yp + 0.5 * t * nyp])
    # ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.4)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.4, alpha=0.55)

for xp in [x[0], x[len(x)//2], x[-1]]:
    # solid section
    yp = np.interp(xp, x, y_s)
    mp = np.interp(xp, x, dy_s)
    lp = np.sqrt(1.0 + mp**2)
    txp, typ = 1.0 / lp, mp / lp
    nxp, nyp = -typ, txp
    sp = 0.12 * ((xp - x.min()) / (x.max() - x.min()))
    p1 = np.array([xp - 0.5 * t * nxp - sp * txp, yp - 0.5 * t * nyp - sp * typ])
    p2 = np.array([xp + 0.5 * t * nxp + sp * txp, yp + 0.5 * t * nyp + sp * typ])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=1.7)

# fixed wall at left
xwall = -0.08
ax.plot([xwall, xwall], [-0.25, 0.45], "k-", lw=2.0)
for yy in np.linspace(-0.22, 0.42, 8):
    ax.plot([xwall - 0.08, xwall], [yy - 0.04, yy], "k-", lw=1.0)


# --- local axes (xi^1 along beam, xi^3 through thickness) ---
origin = np.array([0.4, 0.05])
scale = 0.6

# xi^1 (along beam)
ax.arrow(origin[0], origin[1], scale, 0.0,
         head_width=0.04, head_length=0.08, fc='k', ec='k', lw=1.5)
ax.text(origin[0] + scale + 0.1, origin[1] - 0.02, r'$\xi^1$')

# xi^3 (through thickness, vertical-ish)
ax.arrow(origin[0], origin[1], 0.0, scale*0.7,
         head_width=0.04, head_length=0.08, fc='k', ec='k', lw=1.5)
ax.text(origin[0] - 0.08, origin[1] + scale*0.7 + 0.1, r'$\xi^3$')

# --- thickness label t ---
mid = len(x) // 2
xp = x[mid]
yp = y_s[mid]
mp = dy_s[mid]
lp = np.sqrt(1.0 + mp**2)
txp, typ = 1.0 / lp, mp / lp
nxp, nyp = -typ, txp

p_bot = np.array([xp - 0.5 * t * nxp, yp - 0.5 * t * nyp])
p_top = np.array([xp + 0.5 * t * nxp, yp + 0.5 * t * nyp])

ax.plot([p_bot[0], p_top[0]], [p_bot[1], p_top[1]], "k-", lw=1.3)
ax.text((p_bot[0]+p_top[0])/2 + 0.05,
        (p_bot[1]+p_top[1])/2,
        r'$t$')


ax.set_xlim(-0.25, 3.55)
ax.set_ylim(min(y_bot_d.min(), y_bot_s.min()) - 0.10,
            max(y_top_d.max(), y_top_s.max()) + 0.12)

# ============================================================
# 2) PARASITIC SHEAR MEMBRANE LOCKING
# flatter element to reduce vertical whitespace
# ============================================================
# ax = axes[1]
# ax.set_title("Parasitic shear membrane locking")

# W, H = 1.6, 0.55

# quad_d = np.array([
#     [0.0, 0.0],
#     [W,   0.0],
#     [W,   H],
#     [0.0, H],
# ])

# quad_s = np.array([
#     [0.00, 0.00],
#     [1.68, 0.05],
#     [1.92, 0.58],
#     [0.18, 0.53],
# ])

# ax.plot(
#     [quad_d[0,0], quad_d[1,0], quad_d[2,0], quad_d[3,0], quad_d[0,0]],
#     [quad_d[0,1], quad_d[1,1], quad_d[2,1], quad_d[3,1], quad_d[0,1]],
#     "k--", lw=1.7
# )

# ax.add_patch(Polygon(quad_s, closed=True, facecolor="0.88", edgecolor="k", linewidth=2.2))

# ax.plot([0.8, 0.8], [0.0, H], "k--", lw=1.3)
# ax.plot([0.0, W], [0.5 * H, 0.5 * H], "k--", lw=1.3)

# def interp(Pa, Pb, s):
#     return (1.0 - s) * Pa + s * Pb

# p_bot = interp(quad_s[0], quad_s[1], 0.5)
# p_top = interp(quad_s[3], quad_s[2], 0.5)
# p_left = interp(quad_s[0], quad_s[3], 0.5)
# p_right = interp(quad_s[1], quad_s[2], 0.5)

# ax.plot([p_bot[0], p_top[0]], [p_bot[1], p_top[1]], "k-", lw=1.4)
# ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], "k-", lw=1.4)

# ax.set_xlim(-0.2, 2.05)
# ax.set_ylim(-0.08, 0.72)
# ============================================================
# 2) PARASITIC SHEAR MEMBRANE LOCKING
# planar unstructured quad mesh
# dashed = intended bending-like deformation
# solid  = same deformation + parasitic in-plane shear
# ============================================================
# ============================================================
# 2) PARASITIC SHEAR MEMBRANE LOCKING
# ring-like planar strip, similar orientation to earlier version
# dashed = intended bending-like deformation
# solid  = same deformation + parasitic in-plane shear
# ============================================================
ax = axes[1]
ax.set_title("Parasitic shear membrane locking")

# logical mesh: 4 x 3 quads
nu, nv = 5, 4
u = np.linspace(0.0, 1.0, nu)
v = np.linspace(-0.32, 0.32, nv)
U, V = np.meshgrid(u, v)

# mildly unstructured logical mesh
Uu = U + 0.025 * np.array([
    [ 0.00,  0.02, -0.01,  0.01,  0.00],
    [ 0.00, -0.01,  0.02, -0.02,  0.00],
    [ 0.00,  0.01, -0.02,  0.01,  0.00],
    [ 0.00, -0.02,  0.01, -0.01,  0.00],
])
Vu = V + 0.015 * np.array([
    [ 0.00, -0.01,  0.00,  0.01,  0.00],
    [ 0.01,  0.00, -0.01,  0.00, -0.01],
    [-0.01,  0.01,  0.00, -0.01,  0.01],
    [ 0.00,  0.00,  0.01,  0.00,  0.00],
])

# base strip size
L = 2.45
H = 0.82
X0 = L * Uu
Y0 = H * Vu

# dashed: larger "true" ring-like deformation
center_bulge_d = 0.22 * np.sin(np.pi * Uu)
end_sag_d = -0.10 * (2.0 * Uu - 1.0) ** 2
Xd = X0
Yd = Y0 + center_bulge_d + end_sag_d

# solid: reduced overall deflection due to locking
center_bulge_s = 0.12 * np.sin(np.pi * Uu)
end_sag_s = -0.05 * (2.0 * Uu - 1.0) ** 2
Xbase_s = X0
Ybase_s = Y0 + center_bulge_s + end_sag_s

# add parasitic in-plane shear/skew to the solid mesh
# Xs = Xbase_s + 0.18 * Vu * (0.35 + 0.95 * np.sin(np.pi * Uu)) \
#         + 0.02 * np.sin(2.5 * np.pi * Uu) * (1.0 - (Vu / 0.32) ** 2)
# Ys = Ybase_s

Xs = Xbase_s + 0.28 * Vu * (0.45 + 1.20 * np.sin(np.pi * Uu)) \
        + 0.035 * np.sin(2.5 * np.pi * Uu) * (1.0 - (Vu / 0.32) ** 2)
Ys = Ybase_s

# dashed ideal mesh (lighter / more transparent)
for j in range(nv):
    ax.plot(Xd[j, :], Yd[j, :], "k--", lw=1.3, alpha=0.55)
for i in range(nu):
    ax.plot(Xd[:, i], Yd[:, i], "k--", lw=1.3, alpha=0.55)

# solid locked mesh as filled quads
for j in range(nv - 1):
    for i in range(nu - 1):
        quad = np.array([
            [Xs[j, i],     Ys[j, i]],
            [Xs[j, i+1],   Ys[j, i+1]],
            [Xs[j+1, i+1], Ys[j+1, i+1]],
            [Xs[j+1, i],   Ys[j+1, i]],
        ])
        ax.add_patch(Polygon(quad, closed=True, facecolor="0.88", edgecolor="k", linewidth=1.4))

# tight limits
xmin = min(Xd.min(), Xs.min())
xmax = max(Xd.max(), Xs.max())
ymin = min(Yd.min(), Ys.min())
ymax = max(Yd.max(), Ys.max())

# --- in-plane axes (xi^1, xi^2) ---
origin = np.array([xmin + 0.15, ymin + 0.15])
scale = 0.6

# xi^1 (horizontal-ish)
ax.arrow(origin[0], origin[1], scale, 0.0,
         head_width=0.04, head_length=0.08, fc='k', ec='k', lw=1.5)
ax.text(origin[0] + scale + 0.1, origin[1] - 0.05, r'$\xi^1$')

# xi^2 (in-plane vertical)
ax.arrow(origin[0], origin[1], 0.0, scale,
         head_width=0.04, head_length=0.08, fc='k', ec='k', lw=1.5)
ax.text(origin[0] - 0.08, origin[1] + scale + 0.1, r'$\xi^2$')


ax.set_xlim(xmin - 0.08, xmax + 0.08)
ax.set_ylim(ymin - 0.08, ymax + 0.08)


fig.tight_layout(pad=0.4, w_pad=1.2)
plt.savefig("_shell_locking.svg", dpi=400)
plt.show()
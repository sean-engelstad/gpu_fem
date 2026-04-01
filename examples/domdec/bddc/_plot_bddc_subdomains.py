import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.ticker import NullFormatter, NullLocator


def cylinder_map(x, th, R=1.0):
    return np.array([x, R * np.cos(th), R * np.sin(th)], dtype=float)


def build_nodes(nex=8, net=8, R=1.0, L=2.0, theta0=0.0, theta1=0.5*np.pi):
    x = np.linspace(0.0, L, nex + 1)
    th = np.linspace(theta0, theta1, net + 1)

    X = np.zeros((nex + 1, net + 1))
    Y = np.zeros((nex + 1, net + 1))
    Z = np.zeros((nex + 1, net + 1))

    for i in range(nex + 1):
        for j in range(net + 1):
            p = cylinder_map(x[i], th[j], R=R)
            X[i, j], Y[i, j], Z[i, j] = p

    return X, Y, Z, x, th


def inset_element_quads_with_ids(x_nodes, th_nodes, R, inset_frac=0.16):
    dx = x_nodes[1] - x_nodes[0]
    dth = th_nodes[1] - th_nodes[0]

    quads = []
    elem_ids = []
    nex = len(x_nodes) - 1
    net = len(th_nodes) - 1

    for i in range(nex):
        for j in range(net):
            x0 = x_nodes[i] + inset_frac * dx
            x1 = x_nodes[i + 1] - inset_frac * dx
            t0 = th_nodes[j] + inset_frac * dth
            t1 = th_nodes[j + 1] - inset_frac * dth

            v00 = cylinder_map(x0, t0, R)
            v10 = cylinder_map(x1, t0, R)
            v11 = cylinder_map(x1, t1, R)
            v01 = cylinder_map(x0, t1, R)

            quads.append(np.vstack([v00, v10, v11, v01]))
            elem_ids.append((i, j))

    return quads, elem_ids


def style_axes_clean(ax):
    ax.grid(False)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.zaxis.set_major_locator(NullLocator())
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def get_subdomain_id(elem_i, elem_j, sx, st):
    return elem_i // sx, elem_j // st


def build_subdomain_vertex_points(x_nodes, th_nodes, R, sx, st):
    nex = len(x_nodes) - 1
    net = len(th_nodes) - 1
    nsx = nex // sx
    nst = net // st

    pts = []
    for ii in range(nsx + 1):
        for jj in range(nst + 1):
            i = ii * sx
            j = jj * st
            pts.append(cylinder_map(x_nodes[i], th_nodes[j], R))
    return np.array(pts)


def bowed_edge_curve(xa, tha, xb, thb, R, inward_dx=0.0, inward_dth=0.0, npts=80):
    """
    Build an in-plane bowed curve from endpoint A to B.
    The curve matches endpoints exactly, and bows inward with
    shape amp * sin(pi*s), so displacement is zero at the vertices.
    """
    s = np.linspace(0.0, 1.0, npts)
    bow = np.sin(np.pi * s)

    x = (1.0 - s) * xa + s * xb + inward_dx * bow
    th = (1.0 - s) * tha + s * thb + inward_dth * bow

    return np.array([cylinder_map(xx, tt, R) for xx, tt in zip(x, th)])


def build_subdomain_bowed_curves(x_nodes, th_nodes, R, nsub_x, nsub_t,
                                 bow_x_frac=0.30, bow_t_frac=0.30, npts=80):
    """
    For each subdomain, draw 4 inward-bowed curves connecting the true
    subdomain vertices. The inward displacement is zero at the endpoints
    and maximal at midspan.
    """
    nex = len(x_nodes) - 1
    net = len(th_nodes) - 1

    sx = nex // nsub_x
    st = net // nsub_t

    dx = x_nodes[1] - x_nodes[0]
    dth = th_nodes[1] - th_nodes[0]

    curves = []
    curve_sub_ids = []

    for si in range(nsub_x):
        for sj in range(nsub_t):
            sid = si * nsub_t + sj

            i0 = si * sx
            i1 = (si + 1) * sx
            j0 = sj * st
            j1 = (sj + 1) * st

            xL = x_nodes[i0]
            xR = x_nodes[i1]
            tB = th_nodes[j0]
            tT = th_nodes[j1]

            # left edge: bow inward in +x
            curves.append(
                bowed_edge_curve(
                    xL, tB, xL, tT, R,
                    inward_dx=bow_x_frac * dx,
                    inward_dth=0.0,
                    npts=npts
                )
            )
            curve_sub_ids.append(sid)

            # right edge: bow inward in -x
            curves.append(
                bowed_edge_curve(
                    xR, tB, xR, tT, R,
                    inward_dx=-bow_x_frac * dx,
                    inward_dth=0.0,
                    npts=npts
                )
            )
            curve_sub_ids.append(sid)

            # bottom edge: bow inward in +theta
            curves.append(
                bowed_edge_curve(
                    xL, tB, xR, tB, R,
                    inward_dx=0.0,
                    inward_dth=bow_t_frac * dth,
                    npts=npts
                )
            )
            curve_sub_ids.append(sid)

            # top edge: bow inward in -theta
            curves.append(
                bowed_edge_curve(
                    xL, tT, xR, tT, R,
                    inward_dx=0.0,
                    inward_dth=-bow_t_frac * dth,
                    npts=npts
                )
            )
            curve_sub_ids.append(sid)

    return curves, curve_sub_ids


# ===== MAIN =====

# nex, net = 8, 8
nex, net = 12, 12
nsub_x, nsub_t = 4, 4
sx = nex // nsub_x
st = net // nsub_t

R, L = 1.0, 2.0

X, Y, Z, x_nodes, th_nodes = build_nodes(nex, net, R, L)
# inset_frac = 0.16
inset_frac = 0.1
quads, elem_ids = inset_element_quads_with_ids(x_nodes, th_nodes, R, inset_frac=inset_frac)

num_sub = nsub_x * nsub_t
cmap = plt.get_cmap("tab20", num_sub)

# color each element by subdomain
elem_colors = []
for (i, j) in elem_ids:
    si, sj = get_subdomain_id(i, j, sx, st)
    sid = si * nsub_t + sj
    rgba = list(cmap(sid))
    rgba[3] = 0.35
    elem_colors.append(tuple(rgba))


# primal vertices
vertex_pts = build_subdomain_vertex_points(x_nodes, th_nodes, R, sx, st)


all_nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
# mark subdomain vertex indices

vertex_set = set()
edge_set = set()

# vertices = subdomain corners
for ii in range(nsub_x + 1):
    for jj in range(nsub_t + 1):
        i = ii * sx
        j = jj * st
        vertex_set.add((i, j))

# edge/interface nodes = nodes on subdomain grid lines, excluding vertices
for i in range(nex + 1):
    for j in range(net + 1):
        on_sub_x_line = (i % sx == 0)
        on_sub_t_line = (j % st == 0)

        if (on_sub_x_line or on_sub_t_line) and (i, j) not in vertex_set:
            edge_set.add((i, j))

interior_pts = []
edge_pts = []

for i in range(nex + 1):
    for j in range(net + 1):
        pt = [X[i, j], Y[i, j], Z[i, j]]

        if (i, j) in vertex_set:
            continue
        elif (i, j) in edge_set:
            edge_pts.append(pt)
        else:
            interior_pts.append(pt)

interior_pts = np.array(interior_pts)
edge_pts = np.array(edge_pts)

# bowed in-plane subdomain edge curves
# frac = 0.33
frac = 0.16
curves, curve_sub_ids = build_subdomain_bowed_curves(
    x_nodes, th_nodes, R, nsub_x, nsub_t,
    bow_x_frac=frac,
    bow_t_frac=frac,
    npts=100
)

curve_colors = [cmap(sid) for sid in curve_sub_ids]

# plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
style_axes_clean(ax)

# colored inset elements
elem_poly = Poly3DCollection(
    quads,
    facecolors=elem_colors,
    edgecolors=(0.30, 0.30, 0.30, 0.90),
    linewidths=1.1,
)
ax.add_collection3d(elem_poly)

# bowed subdomain boundary curves
curve_collection = Line3DCollection(
    curves,
    colors=curve_colors,
    linewidths=3.2,
)
ax.add_collection3d(curve_collection)

# interior nodes
# ax.scatter(
#     interior_pts[:, 0], interior_pts[:, 1], interior_pts[:, 2],
#     s=18,
#     c="#4f6272",      # muted blue-gray
#     edgecolors="none",
#     alpha=0.9,
#     depthshade=False,
#     zorder=10,
# )

# interior nodes
# interior nodes -> subtle dark fill
ax.scatter(
    interior_pts[:, 0], interior_pts[:, 1], interior_pts[:, 2],
    s=22,
    c="#222222",
    edgecolors="none",
    alpha=0.4,
    depthshade=False,
    zorder=10,
)

# edge/interface nodes -> medium-dark with black outline
ax.scatter(
    edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2],
    s=28,
    c="#6f6f6f",
    edgecolors="black",
    linewidths=0.45,
    alpha=0.7,
    depthshade=False,
    zorder=20,
)

# primal vertices -> strongest
ax.scatter(
    vertex_pts[:, 0], vertex_pts[:, 1], vertex_pts[:, 2],
    s=90,
    c="black",
    edgecolors="black",
    linewidths=0.8,
    depthshade=False,
    zorder=30,
)

# ax.set_proj_type('ortho')

ax.set_zlim(0, R)
ax.set_ylim(0, R)
ax.set_xlim(0, L)

ax.set_box_aspect((L, R, R))
ax.view_init(elev=20, azim=60)

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0, 0, 1, 1])

plt.savefig(
    "bddc_cylinder.png",
    dpi=400,
    bbox_inches='tight',
    pad_inches=0
)

plt.savefig(
    "bddc_cylinder.svg",
    dpi=400,
    bbox_inches='tight',
    pad_inches=0
)


# plt.savefig("bddc_cylinder.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
# plt.savefig("bddc_cylinder.svg", dpi=400, bbox_inches="tight", pad_inches=0.01)
plt.show()
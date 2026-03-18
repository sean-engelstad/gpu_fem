import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import NullFormatter, NullLocator

# your AMG aggregation imports
import sys
sys.path.append("_src/")
from poisson import poisson_2d_csr
from csr_aggregation import greedy_serial_aggregation_csr


def cylinder_map(x, th, R=1.0):
    return np.array([x, R * np.cos(th), R * np.sin(th)], dtype=float)


def build_nodes(nex=6, net=6, R=1.0, L=2.0, theta0=0.0, theta1=0.5*np.pi):
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


def ij_to_k(i, j, net):
    # flatten (i,j) with indexing="ij"
    return i * (net + 1) + j


def build_aggregation_on_parameter_grid(nex, net, threshold=0.25):
    """
    Build a structured graph in (x, theta) using the same Poisson stencil,
    then run your greedy aggregation on the node graph.
    """
    nx = nex + 1
    ny = net + 1

    Lx = 1.0
    Ly = 1.0
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)

    A_free = poisson_2d_csr(nx, ny, hx, hy)
    aggregate_ind = greedy_serial_aggregation_csr(A_free, threshold=threshold)

    return aggregate_ind.reshape((nx, ny))


# ===== MAIN =====

nex, net = 6, 6
R, L = 1.0, 2.0

X, Y, Z, x_nodes, th_nodes = build_nodes(nex, net, R, L)
quads, elem_ids = inset_element_quads_with_ids(x_nodes, th_nodes, R, inset_frac=0.1)

# AMG aggregation on structured parameter-space node graph
aggregate_ind = build_aggregation_on_parameter_grid(nex, net, threshold=0.25)
num_agg = int(np.max(aggregate_ind) + 1)
print(f"{num_agg=}")

# pick a categorical colormap
cmap = plt.get_cmap("tab20", num_agg)

# color each element by the most common aggregate among its 4 corner nodes
elem_colors = []
for (i, j) in elem_ids:
    agg_corners = [
        aggregate_ind[i, j],
        aggregate_ind[i + 1, j],
        aggregate_ind[i, j + 1],
        aggregate_ind[i + 1, j + 1],
    ]
    agg_corners = np.asarray(agg_corners, dtype=int)
    elem_agg = np.bincount(agg_corners).argmax()
    rgba = list(cmap(elem_agg))
    rgba[3] = 0.35  # transparency
    elem_colors.append(tuple(rgba))

# node colors
node_colors = np.zeros(((nex + 1) * (net + 1), 4))
node_xyz = np.zeros(((nex + 1) * (net + 1), 3))

ctr = 0
for i in range(nex + 1):
    for j in range(net + 1):
        node_xyz[ctr, :] = [X[i, j], Y[i, j], Z[i, j]]
        node_colors[ctr, :] = cmap(int(aggregate_ind[i, j]))
        ctr += 1

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
style_axes_clean(ax)

# colored inset elements
elem_poly = Poly3DCollection(
    quads,
    facecolors=elem_colors,
    edgecolors=(0.25, 0.25, 0.25, 0.9),
    linewidths=1.0,
)
ax.add_collection3d(elem_poly)

# colored nodes
ax.scatter(
    node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
    s=65,
    c=node_colors,
    edgecolors="black",
    linewidths=0.45,
    depthshade=False,
)

ax.set_box_aspect((L, R, R))
ax.view_init(elev=20, azim=60)

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0, 0, 1, 1])

plt.margins(x=0.05, y=0.05)

plt.savefig("agg_cylinder.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
plt.show()
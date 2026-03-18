import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import NullFormatter, NullLocator


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


def inset_element_quads(x_nodes, th_nodes, R, inset_frac=0.16):
    dx = x_nodes[1] - x_nodes[0]
    dth = th_nodes[1] - th_nodes[0]

    quads = []
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
    return quads


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


def cf_mask(i, j):
    # SIMPLE AMG SPLIT (replace this with your actual one if needed)
    # standard AMG coarsening
    # return (i + j) % 2 == 0
    # A2 coarsening, a bit stronger
    return i % 2 == 0 and j % 2 == 0


# ===== MAIN =====

nex, net = 6, 6
# nex, net = 7, 7
R, L = 1.0, 2.0

X, Y, Z, x_nodes, th_nodes = build_nodes(nex, net, R, L)

gray_quads = inset_element_quads(x_nodes, th_nodes, R, inset_frac=0.1)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
style_axes_clean(ax)

# Gray inset elements
elem_poly = Poly3DCollection(
    gray_quads,
    facecolors=(0.7, 0.7, 0.7, 0.5),
    edgecolors=(0.3, 0.3, 0.3, 1.0),
)
ax.add_collection3d(elem_poly)

# Split nodes
xc, yc, zc = [], [], []
xf, yf, zf = [], [], []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if cf_mask(i, j):
            xc.append(X[i, j]); yc.append(Y[i, j]); zc.append(Z[i, j])
        else:
            xf.append(X[i, j]); yf.append(Y[i, j]); zf.append(Z[i, j])

# Plot nodes
ax.scatter(xf, yf, zf, s=40, c="orange", edgecolors="black", label="F")
ax.scatter(xc, yc, zc, s=60, c="blue", edgecolors="black", label="C")

ax.set_box_aspect((L, R, R))
ax.view_init(elev=20, azim=60)
# ax.legend()
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.05, 0.8),  # INSIDE the axes
    frameon=False,
    borderpad=0.2,
    handletextpad=0.3,
    labelspacing=0.2
)

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0, 0, 1, 1])  # fill entire figure

# plt.show()
# plt.savefig("cf_cylinder.svg", dpi=400)
# plt.savefig("cf_cylinder.png", dpi=400)
plt.savefig("cf_cylinder.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# ------------------------------------------------------------
# Style (matching your script)
# ------------------------------------------------------------
# fs = 20
fs = 22

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs -2,
})

plt.style.use(niceplots.get_style())

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
p = 2                  # quadratic
nxe = 4                # number of elements
u = np.linspace(0.0, 1.0, 1200)

# Open uniform knot vector for quadratic C^1 IGA
knots = np.concatenate((
    np.zeros(p + 1),
    np.arange(1, nxe) / nxe,
    np.ones(p + 1),
))

nglob = nxe + p
elem_conn = [[i, i + 1, i + 2] for i in range(nxe)]

# Nondegenerate knot spans = actual elements
ndegen_knots = knots[p:-p]
nelems = len(ndegen_knots) - 1

print(f"knots         = {knots}")
print(f"ndegen_knots  = {ndegen_knots}")
print(f"elem_conn     = {elem_conn}")
print(f"nglob         = {nglob}")

# ------------------------------------------------------------
# Local Bernstein basis on parent interval xi in [0,1]
# ------------------------------------------------------------
def quad_bernstein(xi):
    B0 = (1.0 - xi)**2
    B1 = 2.0 * xi * (1.0 - xi)
    B2 = xi**2
    B = np.vstack((B0, B1, B2))
    return B

# ------------------------------------------------------------
# Assemble global IGA basis from element Bernstein pieces
# ------------------------------------------------------------
global_basis = np.zeros((nglob, u.size))
local_basis = np.zeros((nelems, 3, u.size))

for ielem in range(nelems):
    knot1 = ndegen_knots[ielem]
    knot2 = ndegen_knots[ielem + 1]
    h = knot2 - knot1

    xi = (u - knot1) / h
    mask = (xi >= 0.0) & (xi <= 1.0)
    xi_eval = np.clip(xi, 0.0, 1.0)

    B = quad_bernstein(xi_eval)

    # Quadratic C^1 assembly
    N1_B0 = 0.0 if ielem == 0 else 0.5
    N2_B0 = 1.0 if ielem == 0 else 0.5

    N1_B2 = 0.0 if ielem == nelems - 1 else 0.5
    N0_B2 = 1.0 if ielem == nelems - 1 else 0.5

    N0 = N2_B0 * B[0]
    N1 = N1_B0 * B[0] + B[1] + N1_B2 * B[2]
    N2 = N0_B2 * B[2]

    N0 *= mask
    N1 *= mask
    N2 *= mask

    local_basis[ielem, 0, :] = N0
    local_basis[ielem, 1, :] = N1
    local_basis[ielem, 2, :] = N2

    g0, g1, g2 = elem_conn[ielem]
    global_basis[g0, :] += N0
    global_basis[g1, :] += N1
    global_basis[g2, :] += N2

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 9))
ax0, ax1 = axes

# use ONLY 3 colors for local basis functions, repeated for each element
cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
local_colors = [cycle_colors[0], cycle_colors[1], cycle_colors[2]]

# ------------------------------
# Top: local element basis with spacing between elements
# ------------------------------
gap = 0.10                    # artificial gap between elements in top plot
elem_plot_width = 1.0         # width allotted to each element in top plot


# elem_colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
# # elem_colors = [elem_colors[0], elem_colors[2], elem_colors[3]]
cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
elem_colors = [cycle_colors[0], cycle_colors[1], cycle_colors[2]]

for ielem in range(nelems):
    x_local = np.linspace(0.0, 1.0, u.size)
    knot1 = ndegen_knots[ielem]
    knot2 = ndegen_knots[ielem + 1]
    mask = (u >= knot1) & (u <= knot2)

    # plot only over the actual support of this element
    xi_plot = (u[mask] - knot1) / (knot2 - knot1)

    # spaced-out horizontal coordinate for visualization
    x_plot = ielem * (elem_plot_width + gap) + xi_plot * elem_plot_width

    ax0.plot(x_plot, local_basis[ielem, 0, mask], lw=2.8, color=elem_colors[0])
    ax0.plot(x_plot, local_basis[ielem, 1, mask], lw=2.8, color=elem_colors[1])
    ax0.plot(x_plot, local_basis[ielem, 2, mask], lw=2.8, color=elem_colors[2])

    # element label
    xC = ielem * (elem_plot_width + gap) + 0.5 * elem_plot_width
    ax0.text(
        xC, 1.08, rf"$e={ielem}$",
        ha="center", va="bottom", fontsize=16
    )

    # faint boundaries around each visualized element block
    xL = ielem * (elem_plot_width + gap)
    xR = xL + elem_plot_width
    ax0.axvline(xL, color="k", lw=0.7, alpha=0.20)
    ax0.axvline(xR, color="k", lw=0.7, alpha=0.20)

# nice ticks centered on each element
xticks = [ielem * (elem_plot_width + gap) + 0.5 * elem_plot_width for ielem in range(nelems)]
xticklabels = [rf"$e={ielem}$" for ielem in range(nelems)]
ax0.set_xticks(xticks)
ax0.set_xticklabels(xticklabels)

ax0.set_ylabel("Value")
ax0.set_ylim(-0.02, 1.15)
ax0.set_xlim(-0.02, nelems * elem_plot_width + (nelems - 1) * gap + 0.02)
ax0.grid(True)
ax0.set_title(r"Element basis functions for $C^1$ (p=2) IGA")


# small legend for local basis numbering
ax0.plot([], [], color=elem_colors[0], lw=2.8, label=r"$N_0^e(u)$")
ax0.plot([], [], color=elem_colors[1], lw=2.8, label=r"$N_1^e(u)$")
ax0.plot([], [], color=elem_colors[2], lw=2.8, label=r"$N_2^e(u)$")
ax0.legend(
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.9)
)

# ------------------------------
# Bottom: global assembled IGA basis
# ------------------------------
for iglob in range(nglob):
    ax1.plot(
        u, global_basis[iglob],
        lw=3.0,
        color=cycle_colors[iglob % len(cycle_colors)],
        label=rf"$N_{{{iglob}}}(\xi)$"
    )

for xk in ndegen_knots:
    ax1.axvline(xk, color="k", lw=0.7, alpha=0.25)

ax1.set_xlabel(r"Parametric coordinate $\xi$")
ax1.set_ylabel("Value")
ax1.set_ylim(-0.02, 1.15)
ax1.set_xlim(0.0, 1.0)
ax1.grid(True)
ax1.set_title(r"Global basis functions for $C^1$ (p=2) IGA")
ax1.legend(
    ncol=min(nglob, 3),
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    loc="upper center",
)

plt.tight_layout()
plt.savefig("iga2_basis_assembly_spaced.png", dpi=200, bbox_inches="tight")
plt.savefig("iga2_basis_assembly_spaced.svg", dpi=200, bbox_inches="tight")
plt.show()
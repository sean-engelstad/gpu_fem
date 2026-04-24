# plot global IGA basis functions for 4 elements
# using local span-based basis evaluation like _plate.py
# shown in 2x2 subplots for p = 1, 2, 3, 4

import numpy as np
import matplotlib.pyplot as plt
import niceplots

# ------------------------------------------------------------
# style (same flavor as previous script)
# ------------------------------------------------------------
fs = 24

plt.rcParams.update({
    "font.size": fs + 2,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs - 2,
})

plt.style.use(niceplots.get_style())

# ------------------------------------------------------------
# Cox-de Boor span search
# ------------------------------------------------------------
def find_span(n_ctrl, degree, u, U):
    if u >= U[-1] - 1e-12:
        return n_ctrl - 1
    low = degree
    high = len(U) - degree - 1
    mid = (low + high) // 2
    while True:
        if u < U[mid]:
            high = mid
        elif u >= U[mid + 1]:
            low = mid
        else:
            return mid
        mid = (low + high) // 2

# ------------------------------------------------------------
# nonzero basis functions + first derivatives
# ------------------------------------------------------------
def basis_functions_and_derivatives(span, u, degree, U, n_deriv=1):
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    ndu = np.zeros((degree + 1, degree + 1))
    ndu[0, 0] = 1.0

    for j in range(1, degree + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    N = ndu[:, degree].copy()

    ders = np.zeros((n_deriv + 1, degree + 1))
    a = np.zeros((2, degree + 1))

    for r in range(degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n_deriv + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else degree - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]

            ders[k, r] = d
            s1, s2 = s2, s1

    for k in range(1, n_deriv + 1):
        factor = degree
        for j in range(2, k + 1):
            factor *= (degree - j + 1)
        ders[k, :] *= factor

    return N, ders[1]

# ------------------------------------------------------------
# open uniform knot vector
# ------------------------------------------------------------
def make_open_uniform_knots(nxe, p):
    return [0.0] * (p + 1) + [i / nxe for i in range(1, nxe)] + [1.0] * (p + 1)

# ------------------------------------------------------------
# assemble global basis by sweeping over nonzero knot spans
# ------------------------------------------------------------
def build_global_basis(nxe, p, n_per_elem=120):
    knots = make_open_uniform_knots(nxe, p)
    n_ctrl = nxe + p

    u_plot = []
    basis_plot = [[] for _ in range(n_ctrl)]

    ielem = 0
    for iknot in range(len(knots) - 1):
        knot1 = knots[iknot]
        knot2 = knots[iknot + 1]
        dknot = knot2 - knot1

        if abs(dknot) <= 1e-12:
            continue

        # local element points
        uvec = np.linspace(knot1, knot2, n_per_elem)

        # avoid exact right endpoint except for very last element
        if knot2 < 1.0 - 1e-12:
            ueval = uvec.copy()
            ueval[-1] = knot2 - 1e-12
        else:
            ueval = uvec.copy()

        # span is constant over this nondegenerate knot span
        span = find_span(n_ctrl, p, 0.5 * (knot1 + knot2), knots)

        # local active basis values on this element
        Nvals = np.zeros((p + 1, uvec.shape[0]))
        for iu, uu in enumerate(ueval):
            N, _ = basis_functions_and_derivatives(span, uu, p, knots, n_deriv=1)
            Nvals[:, iu] = N[:]

        # append element contribution into global basis arrays
        # active globals are span-p, ..., span
        active_globals = [span - p + a for a in range(p + 1)]

        for iglob in range(n_ctrl):
            if iglob in active_globals:
                aloc = active_globals.index(iglob)
                vals = Nvals[aloc, :]
            else:
                vals = np.full(uvec.shape, np.nan)

            basis_plot[iglob].append(vals)

        u_plot.append(uvec)
        ielem += 1

    u_all = np.concatenate(u_plot)
    global_basis = np.array([np.concatenate(parts) for parts in basis_plot])

    return np.array(knots), u_all, global_basis

# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
nxe = 4
plist = [1, 2, 3, 4]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for iax, p in enumerate(plist):
    ax = axes[iax]

    knots, u_all, global_basis = build_global_basis(nxe, p, n_per_elem=160)
    n_ctrl = nxe + p
    elem_bounds = np.unique(knots)

    for iglob in range(n_ctrl):
        ax.plot(
            u_all,
            global_basis[iglob],
            lw=3.0,
            color=cycle_colors[iglob % len(cycle_colors)],
            label=rf"$N_{{{iglob}}}(u)$",
        )

    # for xk in elem_bounds:
    #     ax.axvline(xk, color="k", lw=0.7, alpha=0.20)

    # for e in range(nxe):
    #     xc = (e + 0.5) / nxe
    #     ax.text(
    #         xc, 1.06, rf"$e={e}$",
    #         ha="center", va="bottom", fontsize=16
    #     )

    cont = p - 1
    ax.set_title(rf"$p={p}$ global basis ($C^{cont}$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.15)
    ax.grid(True)

    # ax.legend(
    #     ncol=min(3, n_ctrl),
    #     frameon=True,
    #     facecolor="white",
    #     framealpha=1.0,
    #     loc="upper center",
    # )

axes[0].set_ylabel("Value")
axes[2].set_ylabel("Value")
axes[2].set_xlabel("Parametric coordinate u")
axes[3].set_xlabel("Parametric coordinate u")

plt.tight_layout()
plt.savefig("iga_basis_orders_1_to_4.png", dpi=200, bbox_inches="tight")
plt.savefig("iga_basis_orders_1_to_4.svg", dpi=200, bbox_inches="tight")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import niceplots
import math


def build_hist_from_sparse_log(points, final_iter=None, final_val=None):
    """
    Build a full per-iteration history from sparse log points using
    log-space interpolation between reported iterations.
    """
    pts = list(points)
    if final_iter is not None and final_val is not None:
        pts.append((final_iter, final_val))

    pts = sorted(pts, key=lambda x: x[0])

    max_it = pts[-1][0]
    hist = np.zeros(max_it + 1, dtype=float)

    for k in range(len(pts) - 1):
        i0, y0 = pts[k]
        i1, y1 = pts[k + 1]

        if y0 <= 0.0 or y1 <= 0.0:
            raise ValueError("Residual values must be positive.")

        if i1 == i0:
            hist[i0] = y0
            continue

        logy0 = np.log10(y0)
        logy1 = np.log10(y1)

        for i in range(i0, i1 + 1):
            t = (i - i0) / (i1 - i0)
            hist[i] = 10.0 ** ((1.0 - t) * logy0 + t * logy1)

    return hist


def plot_histories(histories, title=None, y_floor=1e-16, save_png=None):
    if not histories:
        print("No histories to plot.")
        return

    plt.style.use(niceplots.get_style())

    fig, ax = plt.subplots(figsize=(8, 6))

    ymin_global = 1e100
    ymax_global = 0.0

    for h in histories:
        y = np.maximum(h["hist"], y_floor)
        it = np.arange(len(y))

        ymin_global = min(ymin_global, np.min(y))
        ymax_global = max(ymax_global, np.max(y))

        ax.semilogy(it, y, label=h["label"], linewidth=2)

    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("Residual norm")

    # if title is not None:
    #     ax.set_title(title)

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.5)
    ax.axhline(1e-6, linestyle="--", linewidth=1.0, color="gray", alpha=0.5)

    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")

    pmax = math.ceil(math.log10(ymax_global))
    pmin = math.floor(math.log10(ymin_global))
    powers = list(range(pmin, pmax + 1))
    yticks = [10.0 ** p for p in powers]

    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"$10^{{{p}}}$" for p in powers])
    ax.set_ylim(10.0 ** pmin, 10.0 ** pmax)

    ax.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=200, bbox_inches="tight")
        print(f"Saved {save_png}")
    else:
        plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # BDDC with wraparound, t = 1e-1
    # ------------------------------------------------------------
    wrap_t1em1_points = [
        (0, 1.55078493e+02),
        (5, 4.45434226e+01),
        (10, 5.79095399e+00),
        (15, 4.85866811e-01),
        (20, 4.59382927e-02),
        (25, 5.08642575e-03),
        (30, 5.60434888e-04),
        (35, 4.00207666e-05),
        (40, 7.10366604e-06),
        (45, 5.55786970e-07),
    ]
    wrap_t1em1_hist = build_hist_from_sparse_log(
        wrap_t1em1_points,
        final_iter=46,
        final_val=5.557869698e-07,
    )

    # ------------------------------------------------------------
    # BDDC with wraparound, t = 1e-3
    # ------------------------------------------------------------
    wrap_t1em3_points = [
        (0, 2.01347753e+02),
        (5, 1.16251515e+02),
        (10, 5.87773029e+01),
        (15, 2.22376489e+01),
        (20, 7.51235468e+00),
        (25, 2.95905037e+00),
        (30, 1.22084376e+00),
        (35, 4.71972142e-01),
        (40, 1.72410462e-01),
        (45, 6.39682585e-02),
        (50, 2.26133310e-02),
        (55, 8.59617011e-03),
        (60, 3.13524743e-03),
        (65, 1.13828787e-03),
        (70, 4.22034367e-04),
        (75, 1.53613922e-04),
        (80, 5.78971574e-05),
        (85, 2.16043567e-05),
        (90, 7.93527147e-06),
        (95, 3.03221472e-06),
    ]
    wrap_t1em3_hist = build_hist_from_sparse_log(
        wrap_t1em3_points,
        final_iter=98,
        final_val=1.973703976e-06,
    )

    # ------------------------------------------------------------
    # BDDC without wraparound, t = 1e-1
    # ------------------------------------------------------------
    nowrap_t1em1_points = [
        (0, 5.17766025e+02),
        (5, 6.49604013e+02),
        (10, 7.14572810e+02),
        (15, 6.39992129e+02),
        (20, 4.98819018e+02),
        (25, 4.34278705e+02),
        (30, 3.52425852e+02),
        (35, 2.98449506e+02),
        (40, 2.55144281e+02),
        (45, 2.19697978e+02),
        (50, 1.85685365e+02),
        (55, 1.54495982e+02),
        (60, 1.30132245e+02),
        (65, 1.12347885e+02),
        (70, 1.01072281e+02),
        (75, 9.16597105e+01),
        (80, 8.21562218e+01),
        (85, 7.17737028e+01),
        (90, 6.05429559e+01),
        (95, 5.22032239e+01),
        (100, 4.57527401e+01),
        (105, 3.84115311e+01),
        (110, 3.18916859e+01),
        (115, 2.62335417e+01),
        (120, 2.18751871e+01),
        (125, 1.82633400e+01),
        (130, 1.49731268e+01),
        (135, 1.27109277e+01),
        (140, 1.07406898e+01),
        (145, 9.11075730e+00),
    ]
    nowrap_t1em1_hist = build_hist_from_sparse_log(
        nowrap_t1em1_points,
        final_iter=146,
        final_val=7.90185305e+00,
    )

    # ------------------------------------------------------------
    # BDDC without wraparound, t = 1e-3
    # ------------------------------------------------------------
    nowrap_t1em3_points = [
        (0, 8.66037064e+01),
        (5, 2.80314932e+02),
        (10, 4.33352510e+02),
        (15, 4.91586784e+02),
        (20, 5.24083539e+02),
        (25, 5.41864944e+02),
        (30, 5.49793085e+02),
        (35, 5.52549067e+02),
        (40, 5.55329545e+02),
        (45, 5.58685615e+02),
        (50, 5.60813551e+02),
        (55, 5.64896649e+02),
        (60, 5.67838602e+02),
        (65, 5.72771478e+02),
        (70, 5.77264319e+02),
        (75, 5.80971886e+02),
        (80, 5.85917323e+02),
        (85, 5.89197928e+02),
        (90, 5.91630978e+02),
        (95, 5.93687528e+02),
        (100, 5.96286197e+02),
        (105, 5.98656601e+02),
        (110, 6.01406954e+02),
        (115, 6.02537177e+02),
        (120, 6.04186017e+02),
        (125, 6.06133853e+02),
        (130, 6.07629634e+02),
        (135, 6.08844382e+02),
        (140, 6.08782792e+02),
        (145, 6.09238983e+02),
    ]
    nowrap_t1em3_hist = build_hist_from_sparse_log(
        nowrap_t1em3_points,
        final_iter=146,
        final_val=6.09173434e+02,
    )

    histories = [
        {"label": r"wraparound, $t=10^{-1}$", "hist": wrap_t1em1_hist},
        {"label": r"wraparound, $t=10^{-3}$", "hist": wrap_t1em3_hist},
        {"label": r"no wraparound, $t=10^{-1}$", "hist": nowrap_t1em1_hist},
        {"label": r"no wraparound, $t=10^{-3}$", "hist": nowrap_t1em3_hist},
    ]

    plot_histories(
        histories,
        title="BDDC Krylov convergence: wraparound vs no wraparound",
        save_png="bddc_wraparound_thickness_compare.png",
    )
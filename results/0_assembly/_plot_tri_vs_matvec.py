import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

plt.style.use(niceplots.get_style())

filename = "csv/tri_matvec.csv"
savefig = "out/tri_vs_all_times.svg"

# filename = "csv/tri_ilu0_matvec.csv"
# savefig = "out/tri_ilu0_vs_all_times.svg"

df = pd.read_csv(filename)

# numeric cleanup
for col in ["nxe", "tri_solve_ms", "mat_vec_ms", "fact_ms"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["dof"] = 6 * (df["nxe"] + 1) ** 2

# BIG fonts
fs = 24
plt.rcParams.update({
    "font.size": fs,
    "axes.titlesize": fs + 4,
    "axes.labelsize": fs + 2,
    "xtick.labelsize": fs - 2,
    "ytick.labelsize": fs - 2,
    "legend.fontsize": fs - 4,
    "axes.linewidth": 2.5,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

gpus = df["GPU"].unique()

# -----------------------
# Common x-limits for both subplots
# -----------------------
x_all = df["dof"].dropna().to_numpy()
xmin = 0.9 * x_all.min()
xmax = 1.15 * x_all.max()

for i, gpu in enumerate(gpus):
    ax = axes[i]

    sub = df[df["GPU"] == gpu].copy()
    sub = sub.sort_values("dof")

    x = sub["dof"].to_numpy()

    line_fact, = ax.plot(x, sub["fact_ms"], "o-", linewidth=3, markersize=8, label="ILU factor")
    line_tri,  = ax.plot(x, sub["tri_solve_ms"], "s-", linewidth=3, markersize=8, label="Tri solve")
    line_mv,   = ax.plot(x, sub["mat_vec_ms"], "^-", linewidth=3, markersize=8, label="SpMV")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)

    ax.set_title(f"({chr(97+i)}) {gpu}")
    ax.grid(True, which="major", alpha=0.3)

    if i == 0:
        ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Degrees of Freedom")

    # -----------------------
    # Callouts at nxe = 300
    # -----------------------
    callout_row = sub[sub["nxe"] == 300].copy()

    if not callout_row.empty:
        callout_row = callout_row.iloc[0]
        x_call = callout_row["dof"]
        x_text = x_call / 5 # move left in log-space

        def add_callout(yval, line, yscale, mode=0):
            if pd.isna(yval):
                return

            if mode == 0 and not ("ilu0" in filename):
                label = f"{yval:.0f} ms"
            elif mode == 0 and "ilu0" in filename:
                label = f"{yval:.2f} ms"
            elif mode == 1:
                label = f"{yval:.2f} ms"
            else:
                label = f"{yval:.3f} ms"

            ax.text(
                x_text,
                yval * yscale,
                label,
                color=line.get_color(),
                fontsize=fs - 4,
                va="center",
                ha="left",
            )

        if "ilu0" in filename:
            offset = 1.25
        else:
            offset = 1.35

        add_callout(callout_row["fact_ms"], line_fact, offset, mode=0)
        add_callout(callout_row["tri_solve_ms"], line_tri, 1.18, mode=1)
        add_callout(callout_row["mat_vec_ms"], line_mv, 1.35, mode=2)

axes[0].legend()

plt.tight_layout()
plt.savefig(savefig, dpi=400)
plt.show()
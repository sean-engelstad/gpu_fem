import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

plt.style.use(niceplots.get_style())

savefig = "out/tri_both_vs_all_times.svg"

# -----------------------
# Read both datasets
# -----------------------
files = {
    "ILU(0)": "csv/tri_ilu0_matvec.csv",
    "LU": "csv/tri_matvec.csv",
}

dfs = {}
for fill, filename in files.items():
    df = pd.read_csv(filename)

    for col in ["nxe", "tri_solve_ms", "mat_vec_ms", "fact_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dof"] = 6 * (df["nxe"] + 1) ** 2
    dfs[fill] = df

# -----------------------
# BIG fonts
# -----------------------
fs = 26
plt.rcParams.update({
    "font.size": fs,
    "axes.titlesize": fs + 4,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
    "axes.linewidth": 2.5,
})

# -----------------------
# Layout: 2x2
# rows = fill, cols = hardware
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

fills = ["ILU(0)", "LU"]
gpus = ["3060Ti", "A100"]

# -----------------------
# Common x-limits across everything
# -----------------------
x_all = []
for df in dfs.values():
    x_all.extend(df["dof"].dropna().to_list())

x_all = np.array(x_all)
xmin = 0.9 * x_all.min()
xmax = 1.15 * x_all.max()

panel_labels = ["(a)", "(b)", "(c)", "(d)"]
ipanel = 0

for i_fill, fill in enumerate(fills):
    df = dfs[fill]

    for i_gpu, gpu in enumerate(gpus):
        ax = axes[i_fill, i_gpu]

        sub = df[df["GPU"] == gpu].copy()
        sub = sub.sort_values("dof")

        x = sub["dof"].to_numpy()

        line_fact, = ax.plot(
            x, sub["fact_ms"], "o-", linewidth=3, markersize=8, label="Factor"
        )
        line_tri, = ax.plot(
            x, sub["tri_solve_ms"], "s-", linewidth=3, markersize=8, label="Tri solve"
        )
        line_mv, = ax.plot(
            x, sub["mat_vec_ms"], "^-", linewidth=3, markersize=8, label="SpMV"
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)

        ax.set_title(f"{panel_labels[ipanel]} {fill} - {gpu}")
        ipanel += 1

        ax.grid(True, which="major", alpha=0.3)

        if i_gpu == 0:
            ax.set_ylabel("Time (ms)")
        if i_fill == 1:
            ax.set_xlabel("Degrees of Freedom")

        # -----------------------
        # Callouts at nxe = 300
        # -----------------------
        callout_row = sub[sub["nxe"] == 300].copy()

        if not callout_row.empty:
            callout_row = callout_row.iloc[0]
            x_call = callout_row["dof"]
            x_text = x_call / 3.0

            def add_callout(yval, line, yscale, mode=0):
                if pd.isna(yval):
                    return

                # label formatting
                if mode == 0 and fill == "LU":
                    label = f"{yval:.0f} ms"
                elif mode == 0 and fill == "ILU(0)":
                    label = f"{yval:.2f} ms"
                elif mode == 1:
                    label = f"{yval:.2f} ms"
                else:
                    label = f"{yval:.3f} ms"

                y_text = yval * yscale

                ax.annotate(
                    label,
                    xy=(x_call, yval),
                    xytext=(x_text, y_text),
                    textcoords="data",
                    color=line.get_color(),
                    fontsize=fs - 4,
                    ha="left",     # anchor left edge of text
                    va="center",
                    bbox=dict(     # invisible box → gives edge for arrow to hit
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.9,
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=line.get_color(),
                        linewidth=1.8,
                        shrinkA=1,   # THIS is key → pulls line off text edge
                        shrinkB=0,
                    ),
                )

            if fill == "ILU(0)":
                fact_offset = 3.0
            else:
                fact_offset = 2

            

            add_callout(callout_row["fact_ms"], line_fact, fact_offset, mode=0)
            offset = 1.1 if fill == "ILU(0)" else 1.1
            offset *= 2
            add_callout(callout_row["tri_solve_ms"], line_tri, offset, mode=1)
            offset = 1.1 if fill == "ILU(0)" else 1.1
            offset *= 2
            add_callout(callout_row["mat_vec_ms"], line_mv, offset, mode=2)

# -----------------------
# One legend only
# -----------------------
axes[0, 0].legend(loc="upper left")

# plt.margins(x=0.03, y=0.03)

plt.tight_layout()
plt.savefig(savefig, dpi=400)
# plt.show()
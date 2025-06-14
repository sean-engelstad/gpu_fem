import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import colorsys

# Utility: desaturate a color (keep same hue/luminance)
def desaturate(rgb, factor=0.3):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, l, s * factor)

# Grid setup
nrows, ncols = 6, 6
grid = np.full((nrows, ncols), -1)
thread_ids = np.arange(32)
grid.flat[:32] = thread_ids

# Add wraparound values 0,1,2,3 at the end
extra_ids = [0, 1, 2, 3]
grid.flat[32:] = extra_ids

# Row = thread ID // 4
orig_rows = np.full_like(grid, -1)
orig_rows[grid >= 0] = grid[grid >= 0] // 4

# Mark wraparound cells
is_wraparound = np.full_like(grid, False, dtype=bool)
is_wraparound.flat[32:] = True

# Thread group colors
group_colors = sns.color_palette("tab10", 8)

fig, axes = plt.subplots(1, 2, figsize=(12, 7))

for ax_i, label in enumerate(["Thread IDs (with wraparound)", "Original Row = Thread ID // 4"]):
    ax = axes[ax_i]
    for i in range(nrows):
        for j in range(ncols):
            tid = grid[i, j]
            row_id = orig_rows[i, j]
            wrap = is_wraparound[i, j]

            if tid >= 0:
                group_id = tid // 4
                base_color = group_colors[group_id]

                if wrap:
                    fill_color = desaturate(base_color, 0.3)
                    edge_color = 'black'
                    line_style = 'dashed'
                    line_width = 2
                    text_color = 'black'
                else:
                    fill_color = base_color
                    edge_color = 'black'
                    line_style = 'solid'
                    line_width = 1
                    text_color = 'white'

                text_value = tid if ax_i == 0 else row_id
                ax.text(j, i, f"{text_value}", ha="center", va="center",
                        color=text_color, fontsize=12, weight="bold")
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                    linestyle=line_style
                ))
            else:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='lightgray',
                    edgecolor='black'
                ))

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.set_title(label)

plt.tight_layout()
plt.savefig("nodal_blocks_highlighted.svg", dpi=400)

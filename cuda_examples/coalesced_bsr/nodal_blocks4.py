import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import colorsys

def desaturate(rgb, factor=0.3):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, l, s * factor)

# Grid layout
n_blocks = 4
block_rows, block_cols = 6, 6
total_rows, total_cols = n_blocks * block_rows, block_cols
grid = np.full((total_rows, total_cols), -1)
wraparound = np.full_like(grid, False, dtype=bool)

# Fill thread IDs: reset at warp start AND wrap to 0 when >31
for warp_start_row in range(0, total_rows, 8):
    tid = 0
    for i in range(warp_start_row, min(warp_start_row + 8, total_rows)):
        for j in range(total_cols):
            grid[i, j] = tid
            tid += 1
            if tid > 31:
                tid = 0
                wraparound[i, j] = True

# Compute original row = thread ID // 4
orig_rows = np.full_like(grid, -1)
orig_rows[grid >= 0] = grid[grid >= 0] // 4

group_colors = sns.color_palette("tab10", 32)

fig, axes = plt.subplots(1, 2, figsize=(14, 14))

for ax_i, label in enumerate(["Thread IDs (wrap at 32 and per warp)", "Original Row = Thread ID // 4"]):
    ax = axes[ax_i]
    for i in range(total_rows):
        for j in range(total_cols):
            tid = grid[i, j]
            group_id = tid // 4
            base_color = group_colors[group_id % len(group_colors)]
            is_wrap = wraparound[i, j]

            fill_color = desaturate(base_color) if is_wrap else base_color
            edge_color = 'black' if is_wrap else '0.3'  # soft edge unless wraparound
            line_style = 'dashed' if is_wrap else 'solid'
            line_width = 2 if is_wrap else 0.5
            text_color = 'black' if is_wrap else 'white'

            text_value = tid if ax_i == 0 else group_id
            ax.text(j, i, f"{text_value}", ha="center", va="center",
                    color=text_color, fontsize=10, weight="bold")
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=line_width,
                linestyle=line_style
            ))

    # Nodal block boundaries (red solid)
    for b in range(1, n_blocks):
        y = b * block_rows - 0.5
        ax.axhline(y=y, color='red', linewidth=2, linestyle='solid')

    # Warp boundaries (blue dashed)
    for warp_y in range(8, total_rows, 8):
        ax.axhline(y=warp_y - 0.5, color='blue', linewidth=2, linestyle='dashed')

    # Y-axis labels: reset 0–7 in each warp
    yticks = []
    ylabels = []
    for y in range(total_rows):
        yticks.append(y)
        ylabels.append(str(y % 8))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)

    ax.set_xlim(-0.5, total_cols - 0.5)
    ax.set_ylim(total_rows - 0.5, -0.5)
    ax.set_xticks(range(total_cols))
    ax.set_xticklabels([])
    ax.set_aspect('equal')
    ax.set_title(label)

plt.tight_layout()
plt.savefig("coalesced_write_pattern_fixed.svg", dpi=300)
plt.show()

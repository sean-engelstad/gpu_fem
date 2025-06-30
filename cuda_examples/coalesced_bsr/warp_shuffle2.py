import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Data from your kernel printout
threads = list(range(32))
loc_val = [0,1,2,3,0,0,6,7,12,13,14,15,0,0,18,19,
           24,25,26,27,0,0,30,31,36,37,38,39,0,0,42,43]
off_val = [4,5,0,0,4,5,10,11,8,9,10,11,16,17,0,0,
           16,17,22,23,20,21,22,23,28,29,0,0,28,29,34,35]
mask = [1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,
        0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,1]
val = [loc_val[i] if mask[i] else off_val[i] for i in threads]

# Plot layout
fig, ax = plt.subplots(figsize=(20, 4))
ax.set_xlim(-1, 32)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
loc_color = '#add8e6'
off_color = '#f4cccc'
val_color1 = '#4a86e8'
val_color0 = '#cc0000'

# Plot boxes
for i in threads:
    # loc_val (top)
    ax.add_patch(patches.Rectangle((i-0.4, 1.2), 0.8, 0.8, facecolor=loc_color, edgecolor='black'))
    ax.text(i, 1.6, str(loc_val[i]), ha='center', va='center', fontsize=8)

    # final val (middle)
    color = val_color1 if mask[i] else val_color0
    ax.add_patch(patches.Rectangle((i-0.4, 0.2), 0.8, 0.8, facecolor=color, edgecolor='black'))
    ax.text(i, 0.6, str(val[i]), ha='center', va='center', fontsize=8, color='white')

    # off_val (bottom)
    ax.add_patch(patches.Rectangle((i-0.4, -0.8), 0.8, 0.8, facecolor=off_color, edgecolor='black'))
    ax.text(i, -0.4, str(off_val[i]), ha='center', va='center', fontsize=8)

    # Add arrow from src (loc vs off) to val
    y_from = 1.2 if mask[i] else -0.8
    y_to = 0.2
    ax.annotate('', xy=(i, y_to), xytext=(i, y_from),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Labels
ax.text(-1, 1.6, 'loc_val', fontsize=10, va='center')
ax.text(-1, 0.6, 'final val', fontsize=10, va='center')
ax.text(-1, -0.4, 'off_val', fontsize=10, va='center')

plt.tight_layout()
plt.show()

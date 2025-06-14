import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Config
val_mask_bits = 0b11001111000011001111000011001111
src_lanes = [0,1,2,3,  0,1,6,7,  4,5,6,7,
             8,9,10,11,  8,9,14,15,  12,13,14,15,
             16,17,18,19, 16,17,22,23]

# Setup
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-2, 35)
ax.set_ylim(-1, 32)
ax.invert_yaxis()

for t in range(32):
    use_val1 = (val_mask_bits >> t) & 1
    src = src_lanes[t]

    color = 'blue' if use_val1 else 'red'
    label = 'val1' if use_val1 else 'val2'
    ax.text(-1.5, t, f"Thread {t}", va='center', fontsize=8)
    ax.plot([src], [t], marker='o', color=color)
    ax.plot([t], [t], marker='s', color=color, label=label if t == 0 else None)

    if src != t:
        ax.annotate("",
                    xy=(t, t), xytext=(src, t),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    connectionstyle=f"arc3,rad={0.25 if src < t else -0.25}"),
                    annotation_clip=False)

# Grid and legend
ax.set_yticks(range(32))
ax.set_yticklabels([f"{t}" for t in range(32)])
ax.set_xticks([])
ax.set_title("__shfl_sync lane communication\nBlue = val1 (mask=1), Red = val2 (mask=0)", fontsize=14)
ax.legend(loc='lower right')
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.set_xlabel("← Source lane →")

plt.tight_layout()
plt.show()

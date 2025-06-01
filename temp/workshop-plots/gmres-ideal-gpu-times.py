import matplotlib.pyplot as plt
import pandas as pd
import niceplots

# GMRES class definition
class GMRES:
    def __init__(self, name, fill, times):
        self.name = name
        self.fill = fill
        self.times = times

# Runtime data
AMD_LU = 1.6643
ND_LU = 1.4677

# colors = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]
# colors = six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
colors = six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]


gmres_list = [
    GMRES("AMD-GMRES", fill=[5, 7, 9, 11, 15, 25], times=[]),
    # GMRES("RCM-GMRES", fill=[5, 7, 9, 11, 15, 25], times=[]),
    # GMRES("Qord-GMRES",  fill=[5, 7, 9, 11, 15, 25], times=[]),
    GMRES("CPU_baseline-LU"), fill=[5*i+1 for i in range(5)] + [25], times=[3.6], # time to beat
]

plt.style.use(niceplots.get_style())
# Plotting
plt.figure(figsize=(10, 7))
for i, gmres in enumerate(gmres_list):
    plt.plot(gmres.fill, gmres.times, color=colors[i], marker='o', label=gmres.name)

# LU baselines
plt.axhline(AMD_LU, color=colors[3], linestyle="-", label="AMD-LU")
plt.axhline(ND_LU, color=colors[4], linestyle="-", label="ND-LU")

plt.margins(x=0.1, y=0.05)
plt.xlabel("ILU Fill Level")
plt.ylabel("Runtime (s)")
# plt.title("GMRES Runtime vs Fill Level (TACS CPU, 8 Procs)")
plt.legend()
# plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("out/gmres-ucrm-cpu-times.png", dpi=400)

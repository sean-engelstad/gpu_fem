from src._cfe_elem import *
import matplotlib.pyplot as plt
import numpy as np
import niceplots

order = 1
# order = 2
# order = 3

gps = get_chebyshev_gps(order)
xi = np.linspace(-1.0, 1.0, 100)

plt.style.use(niceplots.get_style())


Nks = []
for ibasis in range(order + 1):
    Nks += [
        np.array([chebyshev_basis_fcn(ibasis, _xi, order) for _xi in xi])
    ]
    plt.plot(xi, Nks[-1], label=f"N{ibasis}")

plt.legend()
plt.xlabel("Xi")
plt.ylabel("CFE Basis")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

# Example usage
soln = read_from_csv("csv/plate_soln.csv")
# soln = read_from_csv("plate_loads.csv")
# print(soln)

num_nodes = soln.shape[0]//6
ned = int(num_nodes**0.5 - 1)

# regen mesh here
nxe = ned; nye = ned; Lx = 2.0; Ly = 1.0
x = np.linspace(0.0, Lx, nxe+1)
y = np.linspace(0.0, Ly, nye+1)
X, Y = np.meshgrid(x, y)

w_disps = soln[2::6] # only show w disps in plot
W = np.reshape(w_disps, newshape=X.shape)


# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, W, cmap='viridis', edgecolor='none')

# Add labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('SOLN (Z-axis)')
ax.set_title('3D Surface Plot')

# Add a color bar to show the mapping of colors to values
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

ax.set_zscale('symlog')

# Show the plot
plt.show()
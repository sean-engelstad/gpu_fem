""" develop algorithm for parallel graph coloring (demonstrate on CPU serial here first) """

import numpy as np
import matplotlib.pyplot as plt

""" inputs """
# plate problem,
nxe = 4
# nxe = 16
# nxe = 128

nx = nxe + 1
N = nx**2

# batch fraction for randomized parallel coloring
# randomized smaller batches should have less parallel update conflicts => resulting in more parallel and lower # of colors
batch_frac = 0.2

# maybe look at probabilities of nodes in batch being adjacent or connected based on batch size?
# maybe it can still settle and decrease colors later?

""" create element connectivity and init colors """
colors = np.zeros(N)

elem_conn = []
for ielem in range(nxe**2):
    ixe, iye = ielem % nxe, ielem // nxe
    _node = iye * nx + ixe
    nodes = [_node, _node + 1, _node + nx + 1, _node + nx]
    elem_conn += [nodes]

# also need to convert this to list of adjacent nodes.. in dict?
adj_nodes_dict = {inode:[] for inode in range(N)}
for loc_nodes in elem_conn:
    for inode in loc_nodes:
        for jnode in loc_nodes:
            if jnode == inode: continue # don't include node in itself the adjacency list
            adj_nodes_dict[inode] += [jnode]

# now make unique
for inode in range(N):
    adj_nodes_dict[inode] = list(np.unique(np.array(adj_nodes_dict[inode])))

# print(f"{adj_nodes_dict=}")
# exit()

""" demo the coloring algorithm as if in parallel """

def get_min_color(_adj_colors):
    for _color in range(1, 20):
        if not(_color in _adj_colors): return _color
    return None

batch_size = int(N * batch_frac)

for i_outer in range(100):

    print(f"{i_outer=}")

    # choose random batch (no repeats)
    rand_batch = np.random.permutation(N)[:batch_size]

    # copy out current colors since in parallel you can't see changes of other procs of threads, etc.
    current_colors = colors.copy()

    # now plot each iteration..
    COLOR_MAT = np.reshape(current_colors.copy(), (nx, nx))
    plt.imshow(COLOR_MAT, cmap='jet')
    plt.show()

    for inode in rand_batch:
        adj_nodes = adj_nodes_dict[inode]
        adj_colors = current_colors[adj_nodes]
        colors[inode] = get_min_color(adj_colors)

    


        

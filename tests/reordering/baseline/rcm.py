import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import matplotlib.pyplot as plt

# Original adjacency matrix (example with non-trivial RCM permutation)
A = np.array([
    [0, 1, 0, 1, 1, 0, 0],  # node 0
    [1, 0, 1, 0, 0, 0, 0],  # node 1
    [0, 1, 0, 1, 0, 1, 0],  # node 2
    [1, 0, 1, 0, 0, 1, 1],  # node 3
    [1, 0, 0, 0, 0, 0, 1],  # node 4
    [0, 0, 1, 1, 0, 0, 0],  # node 5
    [0, 0, 0, 1, 1, 0, 0]   # node 6
])

def compute_bandwidth(matrix):
    """Compute the bandwidth of a sparse matrix."""
    rows, cols = matrix.nonzero()
    return np.max(np.abs(rows - cols)) if len(rows) > 0 else 0

# Convert to CSR format
csr = csr_matrix(A)

# Compute Reverse Cuthill-McKee permutation
perm = reverse_cuthill_mckee(csr)

# Apply permutation to rows and columns
csr_perm = csr[perm, :][:, perm]

# Print the 5 requested outputs
print("orig rowp:", csr.indptr.tolist())
print("orig cols:", csr.indices.tolist())
print("RCM perm:", perm.tolist())

newrows = csr_perm.indptr.tolist()
newcols = csr_perm.indices.tolist()
newcols2 = []
# they didn't resort newcols by default (C++ does, so have to do that here)
for i in range(7):
    start = newrows[i]
    end = newrows[i+1]
    newcols2 += list(np.sort(np.array(newcols[start:end])))
print("new rowp:", newrows)
print("new cols:", newcols2)

# Compute and print bandwidths
orig_bw = compute_bandwidth(csr)
new_bw = compute_bandwidth(csr_perm)
print("Original bandwidth:", orig_bw)
print("RCM-reduced bandwidth:", new_bw)

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(A, cmap="Greys", interpolation="none")
plt.title("Original Sparsity")
plt.xlabel("Column")
plt.ylabel("Row")

plt.subplot(1, 2, 2)
plt.imshow(csr_perm.toarray(), cmap="Greys", interpolation="none")
plt.title("RCM-Reordered Sparsity")
plt.xlabel("Column")
plt.ylabel("Row")

plt.tight_layout()
plt.show()

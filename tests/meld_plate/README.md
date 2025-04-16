# MELD plate test

- [ ] [Iterative large SVD](https://epubs.siam.org/doi/epdf/10.1137/0915047)
- [ ] [fast efficient 3x3 SVD](https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf)
    * going to use this one as it says it can handle case with rank-deficient matrices better and is fast for 3x3 (no branching on GPU)

* currently getting large displacements in some nodes
    * suspect I need an iterative SVD solve to have more robustness to ill-conditioned matrices (esp. for wing cases)
- [ ] implement my own iterative 3x3 SVD solve first in python
- [ ] then implement 3x3 SVD solve into the meld.cuh code
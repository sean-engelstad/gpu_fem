import numpy as np
import scipy.sparse as sp


class ColoredTwodimElementAddSchwarz:
    """
    Colored 2D Additive Schwarz smoother for node-block matrices.

    Key idea
    --------
    - Same patch construction as standard 2x2 elem-ASW:
        * one anchor patch per node
        * if a full coupled_size x coupled_size patch fits, use it
        * otherwise fall back to a 1-node patch on the boundary
    - Patches are assigned colors on the structured anchor grid
    - Within one color: additive update
    - Across colors: multiplicative / Gauss-Seidel-like update
      (defect is updated after each color sweep)

    For coupled_size = 2, using color_shape=(3,2) gives 6 colors.
    Any color_shape=(cx, cy) with cx >= 2 and cy >= 2 guarantees that
    same-color anchor patches are not adjacent on the structured grid.
    For 2x2 patches, that also ensures same-color full patches do not overlap.
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int = 2,
        coupled_size: int = 2,
        omega: float = 0.7,
        iters: int = 1,
        color_shape=(3, 2),  # default = 6 colors
    ):
        assert sp.isspmatrix_csr(K) or sp.isspmatrix_bsr(K), "K must be CSR or BSR"

        self.K = K.tocsr()
        self.N = self.K.shape[0]

        self.block_dim = int(block_dim)
        assert self.N % self.block_dim == 0
        self.nnodes = self.N // self.block_dim

        self.nx = int(nx)
        self.ny = int(ny)
        assert self.nx * self.ny == self.nnodes, (
            f"nx*ny must equal nnodes: {self.nx}*{self.ny} != {self.nnodes}"
        )

        self.coupled_size = int(coupled_size)
        assert self.coupled_size >= 1

        self.omega = float(omega)
        self.iters = int(iters)

        cx, cy = color_shape
        self.color_shape = (int(cx), int(cy))
        assert self.color_shape[0] >= 2 and self.color_shape[1] >= 2, (
            "Need color_shape=(cx,cy) with cx>=2 and cy>=2 so no adjacent patches share a color"
        )
        self.ncolors = self.color_shape[0] * self.color_shape[1]

        # cache
        self._sch_nodes = []         # list[np.ndarray] : patch node lists
        self._invK = []              # list[np.ndarray] : inverse per patch
        self._patch_colors = []      # list[int]        : color id per patch
        self._color_groups = []      # list[list[int]]  : patch ids per color

        self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(
        cls,
        assembler,
        omega: float = 0.7,
        iters: int = 1,
        coupled_size: int = 2,
        color_shape=(3, 2),
    ):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=assembler.nny if hasattr(assembler, "nny") else assembler.nnx,
            block_dim=assembler.dof_per_node,
            coupled_size=coupled_size,
            omega=omega,
            iters=iters,
            color_shape=color_shape,
        )

    def _patch_color(self, ix: int, iy: int) -> int:
        """
        Structured coloring of anchor grid.

        With color_shape=(cx,cy), color = (ix mod cx) + cx*(iy mod cy).

        If cx>=2 and cy>=2, then horizontally, vertically, and diagonally
        adjacent anchors always have different colors.

        For coupled_size=2 full patches, that implies same-color patches do not overlap.
        """
        cx, cy = self.color_shape
        return (ix % cx) + cx * (iy % cy)

    def rebuild_patch_inverses(self):
        """
        Precompute patch node lists, patch colors, and inverted patch matrices.
        One patch per node (anchor).
        """
        bs = self.block_dim
        K = self.K

        self._sch_nodes = []
        self._invK = []
        self._patch_colors = []
        self._color_groups = [[] for _ in range(self.ncolors)]

        def _find_block(i_node: int, j_node: int):
            """Return dense (bs,bs) block K_{i_node,j_node} if present, else zeros."""
            r0 = bs * i_node
            c0 = bs * j_node
            blk = np.zeros((bs, bs), dtype=float)

            for rr in range(bs):
                gr = r0 + rr
                for p in range(K.indptr[gr], K.indptr[gr + 1]):
                    gc = K.indices[p]
                    if c0 <= gc < c0 + bs:
                        blk[rr, gc - c0] = K.data[p]
            return blk

        pid = 0
        for iy in range(self.ny):
            for ix in range(self.nx):
                anchor = ix + self.nx * iy

                full_fits = (
                    (ix + self.coupled_size <= self.nx) and
                    (iy + self.coupled_size <= self.ny)
                )

                if full_fits:
                    nodes = []
                    for j in range(self.coupled_size):
                        for i in range(self.coupled_size):
                            nodes.append((ix + i) + self.nx * (iy + j))
                    nodes = np.array(nodes, dtype=int)
                else:
                    # boundary fallback: 1-node patch
                    nodes = np.array([anchor], dtype=int)

                color = self._patch_color(ix, iy)

                self._sch_nodes.append(nodes)
                self._patch_colors.append(color)
                self._color_groups[color].append(pid)

                # assemble dense local matrix
                m = len(nodes) * bs
                Kloc = np.zeros((m, m), dtype=float)

                for a, na in enumerate(nodes):
                    ia0 = bs * a
                    for b, nb in enumerate(nodes):
                        ib0 = bs * b
                        Kloc[ia0:ia0 + bs, ib0:ib0 + bs] = _find_block(na, nb)

                self._invK.append(np.linalg.inv(Kloc))
                pid += 1

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        assert rhs.shape == (self.N,)
        soln = np.zeros_like(rhs)
        defect = rhs.copy()
        self.smooth_defect(soln, defect)
        return soln

    def smooth_defect(self, soln: np.ndarray, defect: np.ndarray):
        """
        Colored multiplicative-ASW-style smoothing:

            for iter:
                for color in colors:
                    additive sweep over all patches in this color
                    update soln and defect immediately

        So:
        - additive inside each color
        - multiplicative across colors
        """
        soln = np.asarray(soln)
        defect = np.asarray(defect)

        assert soln.shape == (self.N,)
        assert defect.shape == (self.N,)

        if len(self._invK) == 0:
            self.rebuild_patch_inverses()

        bs = self.block_dim

        for _ in range(self.iters):
            for color in range(self.ncolors):
                pids = self._color_groups[color]
                if len(pids) == 0:
                    continue

                dsoln_color = np.zeros_like(soln)

                # additive sweep over this color
                for pid in pids:
                    nodes = self._sch_nodes[pid]
                    invKp = self._invK[pid]

                    # gather local defect
                    dloc = np.concatenate([defect[bs * n: bs * (n + 1)] for n in nodes])

                    # local solve
                    uloc = invKp @ dloc

                    # scatter-add
                    for a, n in enumerate(nodes):
                        dsoln_color[bs * n: bs * (n + 1)] += (
                            self.omega * uloc[bs * a: bs * (a + 1)]
                        )

                # multiplicative update after each color
                soln += dsoln_color
                defect -= self.K.dot(dsoln_color)

        return
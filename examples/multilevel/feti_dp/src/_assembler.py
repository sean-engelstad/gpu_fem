import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

def build_csr_from_conn(conn, nnodes, symmetric=True):
    """
    Build CSR sparsity (rowp, cols, nnzb) from element connectivity.

    Parameters
    ----------
    conn : list[list[int]]
        Element connectivity: each entry is a list of global node indices in that element.
    nnodes : int
        Total number of nodes.
    symmetric : bool
        If True, add both (i,j) and (j,i). For most FE stiffness matrices, True.

    Returns
    -------
    rowp : np.ndarray shape (nnodes+1,)
    cols : np.ndarray shape (nnzb,)
    nnzb : int
    """
    # adjacency sets per row
    adj = [set() for _ in range(nnodes)]

    for elem_nodes in conn:
        # add all node-to-node couplings inside this element
        for a in elem_nodes:
            if a < 0 or a >= nnodes:
                raise ValueError(f"Node index {a} out of bounds 0..{nnodes-1}")
            row_set = adj[a]
            for b in elem_nodes:
                row_set.add(b)
                if symmetric:
                    # redundant given looping over all a anyway, but harmless; kept for clarity
                    pass

    # convert adjacency to CSR (sorted columns per row)
    rowp = np.zeros(nnodes + 1, dtype=np.int64)
    cols_list = []
    nnzb = 0
    for i in range(nnodes):
        row_cols = sorted(adj[i])
        cols_list.extend(row_cols)
        nnzb += len(row_cols)
        rowp[i + 1] = nnzb

    cols = np.array(cols_list, dtype=np.int64)
    return rowp, cols, nnzb


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


class Subdomain2DAssembler:
    """
    Structured-grid assembler for standard C0 plate elements on a unit square mesh.
    Intended for Q1 Reissner-Mindlin (4-node quad), dof per node = 3: [w, thx, thy].

    Subdomain convention:
      - interior nodes of a subdomain = nodes owned only by that subdomain
        plus any Dirichlet boundary nodes in that subdomain
      - interface nodes of a subdomain = nodes shared by 2+ subdomains,
        excluding Dirichlet boundary nodes

    The class can assemble:
      - global BSR matrix / RHS
      - local subdomain BSR matrices / RHS in local numbering
      - local restrictions for interior/interface unknowns
      - local block partitions A_II, A_IG, A_GI, A_GG
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E: float = 70e9,
        nu: float = 0.3,
        thick: float = 1.0e-2,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        nxs: int = 1,   # num subdomains in x-dir
        nys: int = 1,   # num subdomains in y-dir
        geometry:str='plate',
        radius:float=1.0,
    ):
        self.element = ELEMENT
        self.nxe = int(nxe)
        self.length = float(length)
        self.width = float(width)
        self.load_fcn = load_fcn
        self.clamped = bool(clamped)
        self.E = E
        self.nu = nu
        self.thick = thick
        self.geometry = geometry
        self.radius = radius

        self.element.clamped = self.clamped

        # ---- element expectations ----
        if hasattr(self.element, "ORDER"):
            assert self.element.ORDER == 1, \
                "SubdomainPlateAssembler is for ORDER=1 (Q1) elements."
        assert self.element.nodes_per_elem == 4, \
            "Q1 plate element must have 4 nodes per element."

        # ---- mesh sizes ----
        self.nnx = self.nxe + 1
        self.nny = self.nxe + 1
        self.nnodes = self.nnx * self.nny

        self.dof_per_node = int(self.element.dof_per_node)
        self.N = self.dof_per_node * self.nnodes
        self.num_elements = self.nxe * self.nxe

        # ---- subdomain layout ----
        self.nxs = int(nxs)
        self.nys = int(nys)
        assert self.nxs >= 1 and self.nys >= 1

        self.nnxs = int(np.ceil(self.nxe / self.nxs))   # elems per subdomain in x
        self.nnys = int(np.ceil(self.nxe / self.nys))   # elems per subdomain in y
        self.num_subdomains = self.nxs * self.nys

        # ---- grid spacing ----
        self.dx = self.length / self.nxe
        self.dy = self.width / self.nxe
        self.dth = self.dy / self.radius # for cylinder

        # ---- storage ----
        self.kmat = None
        self.force = None
        self.u = None

        # ------------------------------------------------------------
        # global boundary nodes
        # ------------------------------------------------------------
        self.bcs = []
        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx
            if ix == 0 or iy == 0 or ix == self.nnx - 1 or iy == self.nny - 1:
                self.bcs.append(inode)
        self.bcs = np.array(self.bcs, dtype=int)
        self.bc_set = set(self.bcs.tolist())

        # ------------------------------------------------------------
        # global element node connectivity
        # element nodes ordered as:
        # 0:(ex,ey), 1:(ex+1,ey), 2:(ex+1,ey+1), 3:(ex,ey+1)
        # ------------------------------------------------------------
        self.conn = []
        for ey in range(self.nxe):
            for ex in range(self.nxe):
                n0 = ex + self.nnx * ey
                n1 = (ex + 1) + self.nnx * ey
                # swapped order from plate case
                n2 = ex + self.nnx * (ey + 1)
                n3 = (ex + 1) + self.nnx * (ey + 1)
                self.conn.append(np.array([n0, n1, n2, n3], dtype=int))

        # ------------------------------------------------------------
        # global dof connectivity
        # ------------------------------------------------------------
        dpn = self.dof_per_node
        self.dof_conn = []
        for loc in self.conn:
            dofs = np.empty(4 * dpn, dtype=int)
            k = 0
            for n in loc:
                base = dpn * n
                for a in range(dpn):
                    dofs[k] = base + a
                    k += 1
            self.dof_conn.append(dofs)

        # ------------------------------------------------------------
        # assign each element to a subdomain
        # ------------------------------------------------------------
        self.sd_conn = {i_sd: [] for i_sd in range(self.num_subdomains)}

        for ielem in range(self.num_elements):
            ixe = ielem % self.nxe
            iye = ielem // self.nxe

            ixs = min(ixe // self.nnxs, self.nxs - 1)
            iys = min(iye // self.nnys, self.nys - 1)
            i_sd = int(ixs + iys * self.nxs)

            self.sd_conn[i_sd].append(self.conn[ielem])

        # ------------------------------------------------------------
        # local subdomain node maps and reduced connectivity
        #   sd_node_map[i_sd]      : global node -> local node
        #   sd_node_inv_map[i_sd]  : local node -> global node
        # ------------------------------------------------------------
        self.sd_nnodes = {i_sd: 0 for i_sd in range(self.num_subdomains)}
        self.sd_node_map = {}
        self.sd_node_inv_map = {}
        self.sd_nodes = {}
        self.sd_red_conn = {}

        for i_sd in range(self.num_subdomains):
            _sd_conn = self.sd_conn[i_sd]

            if len(_sd_conn) == 0:
                self.sd_nnodes[i_sd] = 0
                self.sd_node_map[i_sd] = {}
                self.sd_node_inv_map[i_sd] = {}
                self.sd_red_conn[i_sd] = []
                continue

            full_sd_nodes = np.unique(np.concatenate(_sd_conn, axis=0))
            num_sd_nodes = full_sd_nodes.shape[0]
            self.sd_nnodes[i_sd] = num_sd_nodes
            self.sd_nodes[i_sd] = full_sd_nodes

            node_map = {}
            node_inv_map = {}

            for lnode, gnode in enumerate(full_sd_nodes):
                node_map[int(gnode)] = int(lnode)
                node_inv_map[int(lnode)] = int(gnode)

            red_conn = []
            for loc_conn in _sd_conn:
                red_loc_conn = np.array([node_map[int(gnode)] for gnode in loc_conn], dtype=int)
                red_conn.append(red_loc_conn)

            self.sd_node_map[i_sd] = node_map
            self.sd_node_inv_map[i_sd] = node_inv_map
            self.sd_red_conn[i_sd] = red_conn

        # ------------------------------------------------------------
        # global BSR sparsity at node level
        # ------------------------------------------------------------
        self.rowp, self.cols, self.nnzb = build_csr_from_conn(self.conn, self.nnodes)

        # ------------------------------------------------------------
        # local subdomain BSR sparsity at node level
        # ------------------------------------------------------------
        self.sd_rowp = {}
        self.sd_cols = {}
        self.sd_nnzb = {}

        for i_sd in range(self.num_subdomains):
            _rowp, _cols, _nnzb = build_csr_from_conn(
                self.sd_red_conn[i_sd], self.sd_nnodes[i_sd]
            )
            self.sd_rowp[i_sd] = _rowp
            self.sd_cols[i_sd] = _cols
            self.sd_nnzb[i_sd] = _nnzb

        # ------------------------------------------------------------
        # node ownership across subdomains
        # ------------------------------------------------------------
        self.node_to_subdomains = {inode: [] for inode in range(self.nnodes)}
        for i_sd in range(self.num_subdomains):
            for gnode in self.sd_node_map[i_sd].keys():
                self.node_to_subdomains[gnode].append(i_sd)

        # ------------------------------------------------------------
        # classify local subdomain nodes:
        #   interior includes Dirichlet nodes
        #   interface excludes Dirichlet nodes
        # ------------------------------------------------------------
        self.sd_interior_nodes = {}
        self.sd_interface_nodes = {}

        for i_sd in range(self.num_subdomains):
            interior = []
            interface = []

            for lnode in range(self.sd_nnodes[i_sd]):
                gnode = self.sd_node_inv_map[i_sd][lnode]
                nowners = len(self.node_to_subdomains[gnode])

                if nowners <= 1:
                    interior.append(lnode)
                else:
                    if gnode in self.bc_set:
                        interior.append(lnode)
                    else:
                        interface.append(lnode)

            self.sd_interior_nodes[i_sd] = np.array(interior, dtype=int)
            self.sd_interface_nodes[i_sd] = np.array(interface, dtype=int)

        self.interface_nodes_global = np.array(
            sorted(
                gnode for gnode, owners in self.node_to_subdomains.items()
                if len(owners) > 1 and gnode not in self.bc_set
            ),
            dtype=int
        )
        self.interface_dof_global = self._node_list_to_dofs(self.interface_nodes_global)
        # print(f"{self.interface_nodes_global=}")

        # ------------------------------------------------------------
        # placeholders for assembled objects
        # ------------------------------------------------------------
        self.data = None

        self.sd_ndof = {}
        self.sd_data = {}
        self.sd_force = {}
        self.sd_kmat = {}

        self.sd_interior_dofs = {}
        self.sd_interface_dofs = {}
        self.sd_R_interior = {}
        self.sd_R_interface = {}

        self.sd_A_II = {}
        self.sd_A_IG = {}
        self.sd_A_GI = {}
        self.sd_A_GG = {}

    # =================================================================
    # helpers
    # =================================================================

    def _elem_xpts_from_loc(self, loc_conn: np.ndarray):
        """Return elem_xpts of length 12: [x,y,0] per local node."""
        elem_xpts = np.zeros(12, dtype=np.double)
        
        return elem_xpts
    
    def _elem_xpts_from_loc(self, loc_conn: np.ndarray):
        """Return elem_xpts of length 12: [x,y,0] per local node, consistent with element node order."""
        elem_xpts = np.zeros(12, dtype=np.double)

        if self.geometry == 'plate':
            for lnode, gnode in enumerate(loc_conn):
                ix = gnode % self.nnx
                iy = gnode // self.nnx
                elem_xpts[3 * lnode + 0] = ix * self.dx
                elem_xpts[3 * lnode + 1] = iy * self.dy
                elem_xpts[3 * lnode + 2] = 0.0

        elif self.geometry == 'cylinder':
            for lnode, gnode in enumerate(loc_conn):
                ix = gnode % self.nnx
                iy = gnode // self.nnx
                phi = iy * self.dth
                elem_xpts[3*lnode + 0] = ix * self.dx
                elem_xpts[3*lnode + 1] = -self.radius * np.cos(phi)
                elem_xpts[3*lnode + 2] = self.radius * np.sin(phi)

        elif self.geometry == 'sphere':
            for lnode, gnode in enumerate(loc_conn):
                ix = gnode % self.nnx
                iy = gnode // self.nnx
                dphi = self.dx / self.radius
                dth = self.dy / self.radius
                phi = ix * dphi
                th = np.pi/2 - iy * dth
                elem_xpts[3*lnode + 0] = self.radius * np.sin(th) * np.cos(phi)
                elem_xpts[3*lnode + 1] = self.radius * np.sin(th) * np.sin(phi)
                elem_xpts[3*lnode + 2] = self.radius * np.cos(th)
        
        return elem_xpts

    def _node_list_to_dofs(self, node_list: np.ndarray) -> np.ndarray:
        """Expand local node ids to local dof ids."""
        dpn = self.dof_per_node
        return np.array(
            [dpn * inode + a for inode in node_list for a in range(dpn)],
            dtype=int
        )

    def _make_restriction(self, selected_dofs: np.ndarray, ndof_total: int):
        """
        Build boolean restriction matrix R such that:
            x_sub = R @ x
        where x has length ndof_total and x_sub has length len(selected_dofs).
        """
        m = len(selected_dofs)
        rows = np.arange(m, dtype=int)
        cols = np.asarray(selected_dofs, dtype=int)
        vals = np.ones(m, dtype=np.double)
        return sp.csr_matrix((vals, (rows, cols)), shape=(m, ndof_total))

    def _get_bc_dofs_per_node(self):
        """Return constrained dof ids within a node."""
        if self.clamped:
            return [0, 1, 2]
        return [0]

    # =================================================================
    # BC application
    # =================================================================

    def _apply_bcs(self):
        """
        Global BC application.
        - simply-supported: w=0 on boundary
        - clamped: w=thx=thy=0 on boundary
        """
        dpn = self.dof_per_node
        bc_dofs = self._get_bc_dofs_per_node()

        for node in self.bcs:
            for idof in bc_dofs:
                for colp in range(self.rowp[node], self.rowp[node + 1]):
                    bc = self.cols[colp]
                    self.data[colp, idof, :] = 0.0
                    if bc == node:
                        self.data[colp, idof, idof] = 1.0
                self.force[dpn * node + idof] = 0.0

    def _apply_bcs_subdomain(self, i_sd: int):
        """Apply physical Dirichlet BCs to one subdomain matrix in local numbering."""
        dpn = self.dof_per_node
        bc_dofs = self._get_bc_dofs_per_node()

        rowp = self.sd_rowp[i_sd]
        cols = self.sd_cols[i_sd]
        data = self.sd_data[i_sd]
        force = self.sd_force[i_sd]
        node_map = self.sd_node_map[i_sd]   # global -> local

        for gnode in self.bcs:
            if int(gnode) not in node_map:
                continue

            lnode = node_map[int(gnode)]
            for idof in bc_dofs:
                for colp in range(rowp[lnode], rowp[lnode + 1]):
                    bc = cols[colp]
                    data[colp, idof, :] = 0.0
                    if bc == lnode:
                        data[colp, idof, idof] = 1.0
                force[dpn * lnode + idof] = 0.0

    # =================================================================
    # assembly
    # =================================================================

    def _assemble_system(self, bcs: bool = True):
        """Assemble the global finite element system."""
        print("ASSEMBLE SYSTEM")
        dpn = self.dof_per_node
        self.data = np.zeros((self.nnzb, dpn, dpn), dtype=np.double)
        self.force = np.zeros(self.N, dtype=np.double)

        for ielem in range(self.num_elements):
            loc_conn = self.conn[ielem]
            loc_dofs = self.dof_conn[ielem]
            elem_xpts = self._elem_xpts_from_loc(loc_conn)

            kelem = self.element.get_kelem(self.E, self.nu, self.thick, elem_xpts)
            felem = self.element.get_felem(mag=self.load_fcn, elem_xpts=elem_xpts)

            # scatter element stiffness into global node-BSR storage
            for lbr, br in enumerate(loc_conn):
                for colp in range(self.rowp[br], self.rowp[br + 1]):
                    bc = self.cols[colp]
                    hit = np.where(loc_conn == bc)[0]
                    if hit.size == 0:
                        continue
                    lbc = int(hit[0])

                    self.data[colp, :, :] += kelem[
                        dpn * lbr:dpn * (lbr + 1),
                        dpn * lbc:dpn * (lbc + 1)
                    ]

            np.add.at(self.force, loc_dofs, felem)

        if bcs:
            self._apply_bcs()

        self.kmat = sp.bsr_matrix(
            (self.data, self.cols, self.rowp),
            shape=(self.N, self.N)
        )
        print("\tdone with ASSEMBLY")

    def _assemble_subdomain_systems(self, bcs: bool = True):
        """Assemble all local subdomain systems in local subdomain numbering."""
        print("ASSEMBLE SUBDOMAIN SYSTEMS")
        dpn = self.dof_per_node

        self.sd_ndof = {
            i_sd: dpn * self.sd_nnodes[i_sd] for i_sd in range(self.num_subdomains)
        }

        self.sd_data = {
            i_sd: np.zeros((self.sd_nnzb[i_sd], dpn, dpn), dtype=np.double)
            for i_sd in range(self.num_subdomains)
        }

        self.sd_force = {
            i_sd: np.zeros(self.sd_ndof[i_sd], dtype=np.double)
            for i_sd in range(self.num_subdomains)
        }

        self.sd_kmat = {}

        for i_sd in range(self.num_subdomains):
            sd_nelems = len(self.sd_conn[i_sd])
            rowp = self.sd_rowp[i_sd]
            cols = self.sd_cols[i_sd]
            data = self.sd_data[i_sd]
            force = self.sd_force[i_sd]

            for ielem in range(sd_nelems):
                loc_conn = self.sd_conn[i_sd][ielem]          # global node ids
                loc_red_conn = self.sd_red_conn[i_sd][ielem]  # local node ids

                loc_red_dofs = np.array(
                    [dpn * inode + a for inode in loc_red_conn for a in range(dpn)],
                    dtype=int
                )

                elem_xpts = self._elem_xpts_from_loc(loc_conn)
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, elem_xpts)
                felem = self.element.get_felem(mag=self.load_fcn, elem_xpts=elem_xpts)

                # scatter into local BSR
                for lbr, br in enumerate(loc_red_conn):
                    for colp in range(rowp[br], rowp[br + 1]):
                        bc = cols[colp]
                        hit = np.where(loc_red_conn == bc)[0]
                        if hit.size == 0:
                            continue
                        lbc = int(hit[0])

                        data[colp, :, :] += kelem[
                            dpn * lbr:dpn * (lbr + 1),
                            dpn * lbc:dpn * (lbc + 1)
                        ]

                np.add.at(force, loc_red_dofs, felem)

            if bcs:
                self._apply_bcs_subdomain(i_sd)

            self.sd_kmat[i_sd] = sp.bsr_matrix(
                (data, cols, rowp),
                shape=(self.sd_ndof[i_sd], self.sd_ndof[i_sd])
            )
        print("\tdone with ASSEMBLY SUBDOMAINS")


    def build_subdomain_interface_blocks(self):
        """
        Build for each subdomain:
          - interior/interface dof lists
          - restriction operators
          - local matrix blocks A_II, A_IG, A_GI, A_GG

        Convention:
          - interior includes Dirichlet nodes
          - interface excludes Dirichlet nodes
        """
        if len(self.sd_kmat) != self.num_subdomains:
            self._assemble_subdomain_systems(bcs=True)

        self.sd_interior_dofs = {}
        self.sd_interface_dofs = {}
        self.sd_R_interior = {}
        self.sd_R_interface = {}

        self.sd_A_II = {}
        self.sd_A_IG = {}
        self.sd_A_GI = {}
        self.sd_A_GG = {}

        for i_sd in range(self.num_subdomains):
            ndof = self.sd_ndof[i_sd]
            A = self.sd_kmat[i_sd].tocsr()

            I_nodes = self.sd_interior_nodes[i_sd]
            G_nodes = self.sd_interface_nodes[i_sd]

            # print(f"{i_sd=} {I_nodes=} {G_nodes=}")

            I_dofs = self._node_list_to_dofs(I_nodes)
            G_dofs = self._node_list_to_dofs(G_nodes)

            self.sd_interior_dofs[i_sd] = I_dofs
            self.sd_interface_dofs[i_sd] = G_dofs

            self.sd_R_interior[i_sd] = self._make_restriction(I_dofs, ndof)
            self.sd_R_interface[i_sd] = self._make_restriction(G_dofs, ndof)

            self.sd_A_II[i_sd] = A[I_dofs][:, I_dofs].tobsr(
                blocksize=(self.dof_per_node, self.dof_per_node)
            )
            self.sd_A_IG[i_sd] = A[I_dofs][:, G_dofs].tobsr(
                blocksize=(self.dof_per_node, self.dof_per_node)
            )
            self.sd_A_GI[i_sd] = A[G_dofs][:, I_dofs].tobsr(
                blocksize=(self.dof_per_node, self.dof_per_node)
            )
            self.sd_A_GG[i_sd] = A[G_dofs][:, G_dofs].tobsr(
                blocksize=(self.dof_per_node, self.dof_per_node)
            )

        # # now make a unique list of the interface unknowns across all subdomains
        # self.global_interface_nodes = []
        # for i_subdomain in range(self.num_subdomains):


    # =================================================================
    # user-facing helpers
    # =================================================================

    def assemble_all(self, global_bcs: bool = True, subdomain_bcs: bool = True):
        """Convenience routine: assemble global + subdomain systems + interface blocks."""
        self._assemble_system(bcs=global_bcs)
        self._assemble_subdomain_systems(bcs=subdomain_bcs)
        self.build_subdomain_interface_blocks()

    def get_subdomain_global_nodes(self, i_sd: int) -> np.ndarray:
        """Return global node ids in local-node order for subdomain i_sd."""
        return np.array(
            [self.sd_node_inv_map[i_sd][lnode] for lnode in range(self.sd_nnodes[i_sd])],
            dtype=int
        )

    def get_subdomain_global_dofs(self, i_sd: int) -> np.ndarray:
        """Return global dof ids in local-dof order for subdomain i_sd."""
        dpn = self.dof_per_node
        gnodes = self.get_subdomain_global_nodes(i_sd)
        return np.array(
            [dpn * gnode + a for gnode in gnodes for a in range(dpn)],
            dtype=int
        )

    def get_subdomain_interface_global_nodes(self, i_sd: int) -> np.ndarray:
        """Return global interface node ids for subdomain i_sd."""
        return np.array(
            [self.sd_node_inv_map[i_sd][lnode] for lnode in self.sd_interface_nodes[i_sd]],
            dtype=int
        )

    def get_subdomain_interior_global_nodes(self, i_sd: int) -> np.ndarray:
        """Return global interior node ids for subdomain i_sd."""
        return np.array(
            [self.sd_node_inv_map[i_sd][lnode] for lnode in self.sd_interior_nodes[i_sd]],
            dtype=int
        )

    def get_xpts(self) -> np.ndarray:
        """
        Return global nodal coordinates as a flat (3*nnodes,) array:
            [x1, y1, z1,  x2, y2, z2, ...]
        """
        xyz = np.zeros(3 * self.nnodes, dtype=np.double)

        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx

            xyz[3 * inode + 0] = ix * self.dx
            xyz[3 * inode + 1] = iy * self.dy
            xyz[3 * inode + 2] = 0.0

        return xyz

    def direct_solve(self, assembly:bool=True):
        """Assemble and solve the global linear system directly."""
        if assembly: self._assemble_system(bcs=True)
        self.u = sp.linalg.spsolve(self.kmat.tocsc(), self.force)
        return self.u
    
    def _plot_disp_cylinder(self, disp_mag: float = 0.2, mode:str = 'normal', ax=None):
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
            
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))

        x = np.arange(self.nnx) * self.dx
        s = np.arange(self.nnx) * self.dy
        # s = np.linspace(-self.Ly, 0.0, self.nnx)
        phi = s / self.radius

        ix_vec = np.array([inode % self.nnx for inode in range(self.nnodes)])
        iy_vec = np.array([inode // self.nnx for inode in range(self.nnodes)])
        x_flat = ix_vec * self.dx
        s_flat = iy_vec * self.dy
        phi_flat = s_flat / self.radius

        # plot normal deflection v * yhat + w * zhat
        y_flat = -self.radius * np.cos(phi_flat)
        z_flat = self.radius * np.sin(phi_flat)
        yhat = y_flat / self.radius; zhat = z_flat / self.radius
        w = U[:, 1] * yhat + U[:,2] * zhat # true normal deflection
        W = w.reshape((self.nnx, self.nnx))

        X, Phi = np.meshgrid(x, phi, indexing="xy")
        Y = (self.radius + W) * -np.cos(Phi)
        Z = (self.radius + W) * np.sin(Phi)

        # print(f"{phi*180/np.pi=}")

        orig_mag = np.max(np.abs(w))
        W0 = W.copy()
        W = W0 * disp_mag / orig_mag

        # print(f"{W=}")

        X, Phi = np.meshgrid(x, phi, indexing="xy")
        Y = (self.radius + W) * -np.cos(Phi)
        Z = (self.radius + W) * np.sin(Phi)

        # ---- color by selected field ----
        C = np.abs(W0)
        C_face = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

        norm = mcolors.Normalize(vmin=float(C_face.min()), vmax=float(C_face.max()))
        cmap = cm.get_cmap("viridis")
        facecolors = cmap(norm(C_face))

        created_fig = ax is None
        if ax is None:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection="3d")
        # created_fig = True
        else:
            fig = ax.figure

        ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("radial")
        # ax.set_title(f"color={'w'}")
        ax.view_init(elev=25, azim=-135)

        ax.set_box_aspect((1,1,1))

        if created_fig:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=f"|w|")
            plt.tight_layout()
            plt.show()

    def _plot_disp_plate(self):
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))
        w = U[:, 0]

        W = w.reshape((self.nny, self.nnx))
        x = np.arange(self.nnx) * self.dx
        y = np.arange(self.nny) * self.dy
        X, Y = np.meshgrid(x, y, indexing="xy")

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, W, cmap="jet", linewidth=0, antialiased=True, shade=True)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("w")
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="w")
        plt.tight_layout()
        plt.show()

    def _plot_disp_sphere(self):
        print("WARNING plot disp sphere not written yet")
        return

    def plot_disp(self):
        if self.geometry == 'plate':
            self._plot_disp_plate()
        elif self.geometry == 'cylinder':
            self._plot_disp_cylinder()
        elif self.geometry == 'sphere':
            self._plot_disp_sphere()
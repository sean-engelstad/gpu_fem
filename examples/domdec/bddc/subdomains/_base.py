import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from collections import defaultdict


def plot_subdomain_mesh(
    xy,
    conn,
    subdomains,
    title="Quad mesh colored by subdomain",
    show_elem_ids=False,
    show_node_ids=False,
    edgecolor="k",
    linewidth=0.6,
    seed=0,
):
    """
    Plot a quad mesh with elements colored by subdomain.

    Parameters
    ----------
    xy : (nnodes, 2) ndarray
        Node coordinates.
    conn : (nelems, 4) ndarray of int
        Quad connectivity. Each row is [n1, n2, n3, n4].
    subdomains : (nelems,) ndarray of int
        Subdomain ID for each element.
    seed : int
        Random seed used to shuffle colors for better visual separation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    xy = np.asarray(xy, dtype=float)
    conn = np.asarray(conn, dtype=int)
    subdomains = np.asarray(subdomains, dtype=int)

    nelems = conn.shape[0]
    if subdomains.shape[0] != nelems:
        raise ValueError("subdomains must have length equal to number of elements")

    unique_subdomains = np.unique(subdomains)
    nsub = len(unique_subdomains)

    # Map arbitrary subdomain IDs to compact indices for coloring
    sdom_to_compact = {sd: i for i, sd in enumerate(unique_subdomains)}
    compact_colors = np.array([sdom_to_compact[s] for s in subdomains], dtype=int)

    polys = [xy[elem] for elem in conn]

    # Sample a continuous colormap, then shuffle colors so consecutive
    # subdomain IDs don't end up with similar colors
    base_colors = plt.get_cmap("turbo")(np.linspace(0.0, 1.0, max(nsub, 1)))
    rng = np.random.default_rng(seed)
    rng.shuffle(base_colors)

    cmap = ListedColormap(base_colors)
    norm = BoundaryNorm(np.arange(-0.5, nsub + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    coll = PolyCollection(
        polys,
        array=compact_colors,
        cmap=cmap,
        norm=norm,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    ax.add_collection(coll)

    if show_elem_ids:
        for e, elem in enumerate(conn):
            centroid = xy[elem].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(e),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    if show_node_ids:
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), fontsize=7, color="red")

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    cbar = fig.colorbar(coll, ax=ax, shrink=0.85, ticks=np.arange(nsub))
    cbar.set_ticklabels([str(sd) for sd in unique_subdomains])
    cbar.set_label("Subdomain ID")

    plt.tight_layout()
    plt.show()

# THIS IS WRONG DEFINITION OF SUBDOMAIN VERTICES (subdomain corner not same thing for unstructured mesh)
# def count_subdomain_corners(conn, subdomains):
#     """
#     A corner is a node attached to exactly one element in that subdomain.

#     Returns
#     -------
#     corners_per_subdomain : dict
#         corners_per_subdomain[sd] = ndarray of corner node IDs
#     corner_counts_per_subdomain : dict
#         corner_counts_per_subdomain[sd] = number of corners in subdomain sd
#     total_corners : int
#         Total over all subdomains
#     min_corners : int
#         Minimum number of corners over all subdomains
#     max_corners : int
#         Maximum number of corners over all subdomains
#     min_corner_subdomains : list[int]
#         Subdomains attaining the minimum
#     max_corner_subdomains : list[int]
#         Subdomains attaining the maximum
#     """
#     corners_per_subdomain = {}
#     corner_counts_per_subdomain = {}

#     for sd in np.unique(subdomains):
#         elem_ids = np.where(subdomains == sd)[0]
#         node_use_count = defaultdict(int)

#         for e in elem_ids:
#             for n in conn[e]:
#                 node_use_count[n] += 1

#         corner_nodes = np.array(
#             sorted([n for n, c in node_use_count.items() if c == 1]),
#             dtype=int,
#         )

#         corners_per_subdomain[int(sd)] = corner_nodes
#         corner_counts_per_subdomain[int(sd)] = len(corner_nodes)

#     total_corners = int(sum(corner_counts_per_subdomain.values()))

#     counts = np.array(list(corner_counts_per_subdomain.values()), dtype=int)
#     min_corners = int(np.min(counts))
#     max_corners = int(np.max(counts))

#     min_corner_subdomains = [
#         sd for sd, c in corner_counts_per_subdomain.items() if c == min_corners
#     ]
#     max_corner_subdomains = [
#         sd for sd, c in corner_counts_per_subdomain.items() if c == max_corners
#     ]

#     return (
#         corners_per_subdomain,
#         corner_counts_per_subdomain,
#         total_corners,
#         min_corners,
#         max_corners,
#         min_corner_subdomains,
#         max_corner_subdomains,
#     )

def count_subdomain_vertices(conn, subdomains):
    """
    Count BDDC-style vertex nodes, where a vertex node is defined as a node
    that belongs to 3 or more distinct subdomains.

    Returns
    -------
    vertices_per_subdomain : dict
        vertices_per_subdomain[sd] = ndarray of vertex node IDs belonging to subdomain sd
    vertex_counts_per_subdomain : dict
        vertex_counts_per_subdomain[sd] = number of vertex nodes in subdomain sd
    total_vertices : int
        Total number of unique global vertex nodes
    min_vertices : int
        Minimum number of vertex nodes over all subdomains
    max_vertices : int
        Maximum number of vertex nodes over all subdomains
    min_vertex_subdomains : list[int]
        Subdomains attaining the minimum
    max_vertex_subdomains : list[int]
        Subdomains attaining the maximum
    """
    conn = np.asarray(conn, dtype=int)
    subdomains = np.asarray(subdomains, dtype=int)

    unique_subdomains = np.unique(subdomains)

    # For each node, store the set of subdomains that touch it
    node_to_subdomains = defaultdict(set)

    for e, elem_nodes in enumerate(conn):
        sd = int(subdomains[e])
        for n in elem_nodes:
            node_to_subdomains[int(n)].add(sd)

    # Global vertex nodes: touched by 3 or more distinct subdomains
    global_vertex_nodes = {
        n for n, sds in node_to_subdomains.items() if len(sds) >= 3
    }

    # For each subdomain, collect the vertex nodes that belong to it
    vertices_per_subdomain = {}
    vertex_counts_per_subdomain = {}

    for sd in unique_subdomains:
        elem_ids = np.where(subdomains == sd)[0]

        sd_nodes = set()
        for e in elem_ids:
            sd_nodes.update(conn[e])

        sd_vertex_nodes = np.array(
            sorted(n for n in sd_nodes if n in global_vertex_nodes),
            dtype=int,
        )

        vertices_per_subdomain[int(sd)] = sd_vertex_nodes
        vertex_counts_per_subdomain[int(sd)] = len(sd_vertex_nodes)

    total_vertices = len(global_vertex_nodes)

    counts = np.array(list(vertex_counts_per_subdomain.values()), dtype=int)
    min_vertices = int(np.min(counts))
    max_vertices = int(np.max(counts))

    min_vertex_subdomains = [
        sd for sd, c in vertex_counts_per_subdomain.items() if c == min_vertices
    ]
    max_vertex_subdomains = [
        sd for sd, c in vertex_counts_per_subdomain.items() if c == max_vertices
    ]

    return (
        vertices_per_subdomain,
        vertex_counts_per_subdomain,
        total_vertices,
        min_vertices,
        max_vertices,
        min_vertex_subdomains,
        max_vertex_subdomains,
    )


def subdomain_element_stats(subdomains):
    """
    Compute number of elements per subdomain and min/max statistics.
    """
    subdomains = np.asarray(subdomains, dtype=int)
    unique_subdomains, counts = np.unique(subdomains, return_counts=True)

    counts_dict = {int(sd): int(c) for sd, c in zip(unique_subdomains, counts)}
    min_count = int(counts.min())
    max_count = int(counts.max())

    min_subdomains = [int(sd) for sd, c in counts_dict.items() if c == min_count]
    max_subdomains = [int(sd) for sd, c in counts_dict.items() if c == max_count]

    return counts_dict, min_count, max_count, min_subdomains, max_subdomains


def build_element_adjacency(conn):
    """
    Build element adjacency based on shared edges.

    Two quads are adjacent if they share an edge (2 common nodes).

    Returns
    -------
    neighbors : list of sets
        neighbors[e] = set of neighboring element indices
    """
    conn = np.asarray(conn, dtype=int)
    nelems = conn.shape[0]

    edge_to_elems = defaultdict(list)

    for e, elem in enumerate(conn):
        n0, n1, n2, n3 = elem
        edges = [
            tuple(sorted((n0, n1))),
            tuple(sorted((n1, n2))),
            tuple(sorted((n2, n3))),
            tuple(sorted((n3, n0))),
        ]
        for edge in edges:
            edge_to_elems[edge].append(e)

    neighbors = [set() for _ in range(nelems)]
    for elems_on_edge in edge_to_elems.values():
        if len(elems_on_edge) >= 2:
            for i in elems_on_edge:
                for j in elems_on_edge:
                    if i != j:
                        neighbors[i].add(j)

    return neighbors


def assign_subdomains_greedy(conn, nsub):
    """
    Simple graph-based greedy element partition for testing.

    This is NOT a sophisticated partitioner; it's just useful for
    generating a reasonable subdomain coloring for quick experiments.

    Strategy:
    - Build element adjacency graph
    - BFS through connected components
    - Assign elements to subdomains in roughly balanced round-robin chunks
    """
    conn = np.asarray(conn, dtype=int)
    nelems = conn.shape[0]

    if nsub <= 0:
        raise ValueError("nsub must be positive")

    neighbors = build_element_adjacency(conn)
    subdomains = -np.ones(nelems, dtype=int)

    visited = np.zeros(nelems, dtype=bool)

    # Collect BFS ordering over all connected components
    bfs_order = []
    for start in range(nelems):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        while queue:
            e = queue.pop(0)
            bfs_order.append(e)
            for nb in neighbors[e]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    # Balanced chunk assignment
    target = int(np.ceil(nelems / nsub))
    for k, e in enumerate(bfs_order):
        sd = min(k // target, nsub - 1)
        subdomains[e] = sd

    return subdomains


def make_structured_quad_mesh(nx, ny, lx=1.0, ly=1.0, jitter=0.0, seed=0):
    """
    Create a logically structured quad mesh, optionally jittered so the
    geometry looks more unstructured.

    Connectivity remains valid quads. This is convenient for testing
    subdomain logic before plugging in a true unstructured mesh.

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y.
    lx, ly : float
        Plate dimensions.
    jitter : float
        Fraction of cell size used to randomly perturb interior nodes.
        Recommended range: 0.0 to 0.35
    seed : int
        RNG seed.

    Returns
    -------
    xy : (nnodes, 2) ndarray
    conn : (nelems, 4) ndarray
    """
    rng = np.random.default_rng(seed)

    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    xy = np.column_stack([X.ravel(), Y.ravel()])

    dx = lx / nx
    dy = ly / ny

    # Jitter interior nodes only
    if jitter > 0.0:
        for j in range(1, ny):
            for i in range(1, nx):
                node = j * (nx + 1) + i
                xy[node, 0] += jitter * dx * (2.0 * rng.random() - 1.0)
                xy[node, 1] += jitter * dy * (2.0 * rng.random() - 1.0)

    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = (j + 1) * (nx + 1) + i
            n2 = n3 + 1
            conn.append([n0, n1, n2, n3])

    conn = np.array(conn, dtype=int)
    return xy, conn

import gmsh

def generate_structured_quad_mesh_gmesh(
    lx=1.0,
    ly=1.0,
    target_size=0.08,
    algorithm=8,
    recomb_algorithm=2,
):
    """
    Generate a genuinely unstructured quad mesh on a rectangular plate using Gmsh.

    Parameters
    ----------
    lx, ly : float
        Plate dimensions.
    target_size : float
        Target mesh size.
    algorithm : int
        2D mesh algorithm in Gmsh.
        Common choices:
          5 = Delaunay
          6 = Frontal-Delaunay
          8 = Frontal-Delaunay for Quads
    recomb_algorithm : int
        Recombination algorithm.
        Common choices:
          1 = blossom
          2 = simple full-quad style option

    Returns
    -------
    xy : (nnodes, 2) ndarray
        Node coordinates.
    conn : (nelems, 4) ndarray
        Quad connectivity, zero-based.
    """
    gmsh.initialize()
    gmsh.model.add("quad_plate")

    # Rectangle geometry
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, target_size)
    p2 = gmsh.model.geo.addPoint(lx, 0.0, 0.0, target_size)
    p3 = gmsh.model.geo.addPoint(lx, ly, 0.0, target_size)
    p4 = gmsh.model.geo.addPoint(0.0, ly, 0.0, target_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Ask for quad recombination
    gmsh.model.mesh.setRecombine(2, s)

    gmsh.option.setNumber("Mesh.Algorithm", algorithm)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recomb_algorithm)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)

    gmsh.model.mesh.generate(2)

    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=int)
    xyz = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    xy = xyz[:, :2]

    # Get all 2D elements and keep quads only
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)

    conn = []
    for etype, _, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
        name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

        # Keep only 4-node quads
        if num_nodes == 4 and "Quadrilateral" in name:
            arr = np.asarray(enodes, dtype=int).reshape(-1, 4)
            for elem in arr:
                conn.append([tag_to_idx[n] for n in elem])

    gmsh.finalize()

    conn = np.asarray(conn, dtype=int)

    if len(conn) == 0:
        raise RuntimeError(
            "No 4-node quads were generated. Try changing target_size or Gmsh settings."
        )

    return xy, conn

def generate_unstructured_quad_mesh(
    lx=1.0,
    ly=1.0,
    target_size=0.12,
    n_interior_pts=80,
    seed=1,
):
    """
    Generate a genuinely unstructured quad-dominant mesh on a rectangular plate.

    Strategy:
      1. Create rectangle
      2. Add random interior points to destroy regularity
      3. Generate unstructured triangular mesh
      4. Recombine triangles into quads

    Returns
    -------
    xy : (nnodes, 2) ndarray
        Node coordinates
    conn : (nelems, 4) ndarray
        4-node quad connectivity (zero-based)
    """
    rng = np.random.default_rng(seed)

    gmsh.initialize()
    gmsh.model.add("unstructured_quad_plate")

    # --- geometry ---
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, target_size)
    p2 = gmsh.model.geo.addPoint(lx, 0.0, 0.0, target_size)
    p3 = gmsh.model.geo.addPoint(lx, ly, 0.0, target_size)
    p4 = gmsh.model.geo.addPoint(0.0, ly, 0.0, target_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    # Add random interior points
    interior_pt_tags = []
    margin = 0.05 * min(lx, ly)
    for _ in range(n_interior_pts):
        x = rng.uniform(margin, lx - margin)
        y = rng.uniform(margin, ly - margin)
        pt = gmsh.model.geo.addPoint(x, y, 0.0, target_size)
        interior_pt_tags.append(pt)

    gmsh.model.geo.synchronize()

    # Embed points into the surface so they influence the mesh
    if interior_pt_tags:
        gmsh.model.mesh.embed(0, interior_pt_tags, 2, surf)

    # Ask Gmsh for an unstructured tri mesh first, then recombine
    gmsh.model.mesh.setRecombine(2, surf)

    # Delaunay / Frontal tend to be more irregular than "quad-friendly" algorithms
    gmsh.option.setNumber("Mesh.Algorithm", 5)              # 5 = Delaunay
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) # 1 = Blossom
    gmsh.option.setNumber("Mesh.RecombineAll", 1)

    # Optional smoothing helps quality a bit
    gmsh.option.setNumber("Mesh.Smoothing", 10)

    gmsh.model.mesh.generate(2)

    # --- extract nodes ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=int)
    xyz = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    xy = xyz[:, :2]

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

    # --- extract only 4-node quads ---
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)

    conn = []
    tri_count = 0
    quad_count = 0

    for etype, _, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
        name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

        arr = np.asarray(enodes, dtype=int)

        if num_nodes == 4 and "Quadrilateral" in name:
            arr = arr.reshape(-1, 4)
            for elem in arr:
                conn.append([tag_to_idx[n] for n in elem])
            quad_count += arr.shape[0]

        elif num_nodes == 3 and "Triangle" in name:
            arr = arr.reshape(-1, 3)
            tri_count += arr.shape[0]

    gmsh.finalize()

    conn = np.asarray(conn, dtype=int)

    if len(conn) == 0:
        raise RuntimeError("No quads were generated.")

    print(f"Generated {quad_count} quads and {tri_count} leftover triangles.")
    if tri_count > 0:
        print("Note: this is quad-dominant, not guaranteed all-quad.")

    return xy, conn

def print_subdomain_report(conn, subdomains, verbose=False):
    (
        vertices_per_subdomain,
        vertex_counts,
        total_vertices,
        min_vertices,
        max_vertices,
        min_vertex_subdomains,
        max_vertex_subdomains,
    ) = count_subdomain_vertices(conn, subdomains)

    elem_counts, min_count, max_count, min_sds, max_sds = subdomain_element_stats(subdomains)

    # total nodes in mesh
    total_nodes = int(np.max(conn) + 1)

    # ratio (handle divide-by-zero just in case)
    vertex_ratio = total_nodes / total_vertices if total_vertices > 0 else np.inf

    print("=" * 70)
    print("SUBDOMAIN REPORT")
    print("=" * 70)
    print(f"Number of quad elements          : {len(conn)}")
    print(f"Number of subdomains             : {len(np.unique(subdomains))}")
    print(f"Total nodes                      : {total_nodes}")
    print(f"Total global vertex nodes        : {total_vertices}")
    print(f"Node / vertex ratio              : {vertex_ratio:.2f}")
    print(f"Min elems / subdomain            : {min_count} (subdomains {min_sds})")
    print(f"Max elems / subdomain            : {max_count} (subdomains {max_sds})")
    print(f"Min vertex nodes / subdomain     : {min_vertices} (subdomains {min_vertex_subdomains})")
    print(f"Max vertex nodes / subdomain     : {max_vertices} (subdomains {max_vertex_subdomains})")
    print()

    print("Elements per subdomain:")
    for sd in sorted(elem_counts):
        print(f"  sd {sd:4d} : {elem_counts[sd]}")

    print()
    print("Vertex-node counts per subdomain:")
    for sd in sorted(vertex_counts):
        print(f"  sd {sd:4d} : {vertex_counts[sd]}")

    if verbose:
        print()
        print("Vertex node IDs per subdomain:")
        for sd in sorted(vertices_per_subdomain):
            print(f"  sd {sd:4d} : {vertices_per_subdomain[sd]}")

    print("=" * 70)

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # DEMO 1: Generate a test mesh automatically
    # ------------------------------------------------------------------
    nx, ny = 12, 8
    # xy, conn = make_structured_quad_mesh(
    #     nx=nx,
    #     ny=ny,
    #     lx=4.0,
    #     ly=2.5,
    #     jitter=0.22,   # makes geometry look less structured
    #     seed=4,
    # )

    # 1) Generate a genuinely unstructured quad mesh on a rectangular plate
    # xy, conn = generate_structured_quad_mesh_gmesh(
    #     lx=4.0,
    #     ly=2.0,
    #     target_size=0.18,
    #     algorithm=8,
    #     recomb_algorithm=2,
    # )

    xy, conn = generate_unstructured_quad_mesh(
        lx=4.0,
        ly=2.0,
        target_size=0.18,
        n_interior_pts=120,
        seed=7,
    )

    # Example partition for testing
    nsub = 9
    subdomains = assign_subdomains_greedy(conn, nsub=nsub)

    # Print stats
    report = print_subdomain_report(conn, subdomains, verbose=False)

    # Plot
    plot_subdomain_mesh(
        xy,
        conn,
        subdomains,
        title="Quad mesh with subdomain colors",
        show_elem_ids=False,
        show_node_ids=False,
    )
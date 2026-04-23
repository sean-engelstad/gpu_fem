from _base import build_element_adjacency
import numpy as np

def assign_bddc_subdomains(conn: list, target_sd_size: int = 4):
    """
    Build subdomains with a target size while minimizing subdomain corners for BDDC.

    - Each subdomain corner adds cost (requires a K_II^{-1} solve in S_VV)
    - Prefer compact subdomains (~4 corners) over jagged ones
    - Grow from a seed using adjacent elements (frontier)
    - Add elements that minimize resulting corner count
    """

    conn = np.asarray(conn, dtype=int)
    nelems = conn.shape[0]

    neighbors = build_element_adjacency(conn)
    subdomains = -np.ones(nelems, dtype=int)
    visited = np.zeros(nelems, dtype=bool)

    def _get_sd_corners(c_sd_elems):
        nodes = []
        for e in c_sd_elems:
            nodes += list(conn[e])
        nodes = np.unique(np.array(nodes, dtype=int))

        node_dict = {n: i for i, n in enumerate(nodes)}

        node_elem_cts = np.zeros(nodes.shape[0], dtype=np.int32)
        for e in c_sd_elems:
            for n in conn[e]:
                node_elem_cts[node_dict[n]] += 1

        num_corners = np.sum(node_elem_cts == 1)
        return int(num_corners)

    subdomain_ind = 0
    while not np.all(visited):

        # get next unvisited seed element
        elem = -1
        for _elem in range(nelems):
            if not visited[_elem]:
                elem = _elem
                break

        if elem == -1:
            break

        # initialize subdomain
        sd_elems = [elem]
        subdomains[elem] = subdomain_ind
        visited[elem] = True

        while len(sd_elems) < target_sd_size:

            # current frontier: all unvisited elems adjacent to current subdomain
            frontier = []
            for e in sd_elems:
                frontier += list(neighbors[e])

            frontier = list({
                e for e in frontier
                if (not visited[e]) and (e not in sd_elems)
            })

            if len(frontier) == 0:
                break

            # rank the whole frontier
            scores = []
            sd_set = set(sd_elems)
            for nb_elem in frontier:
                proposed_sd_elems = sd_elems + [nb_elem]
                num_corners = _get_sd_corners(proposed_sd_elems)

                # tie-breaker: prefer candidates touching more of current sd
                shared_adj = sum((nb in sd_set) for nb in neighbors[nb_elem])

                scores.append((num_corners, -shared_adj, nb_elem))

            scores.sort()

            # add as many frontier elems as possible this round
            added_any = False
            for _, _, best_elem in scores:
                if len(sd_elems) >= target_sd_size:
                    break
                if visited[best_elem]:
                    continue

                # optional safety: require it still touches current subdomain
                if not any((nb in set(sd_elems)) for nb in neighbors[best_elem]):
                    continue

                sd_elems.append(best_elem)
                subdomains[best_elem] = subdomain_ind
                visited[best_elem] = True
                added_any = True

            if not added_any:
                break

        subdomain_ind += 1

    return subdomains

from collections import defaultdict

def assign_bddc_subdomains_v2(conn: list, target_sd_size: int = 4, min_frac: float = 0.5):
    """
    Build subdomains with a target size while minimizing subdomain corners for BDDC.

    v2:
    - First build initial subdomains using the greedy frontier growth
    - Then post-process very small subdomains
    - For each small subdomain, test merging into each adjacent subdomain
    - Only accept a merge if it reduces the total number of global vertex nodes
      (where a vertex node belongs to 3 or more subdomains)

    Parameters
    ----------
    conn : list or ndarray
        Quad connectivity, shape (nelems, 4)
    target_sd_size : int
        Target number of elements per subdomain
    min_frac : float
        A subdomain is considered "small" if its size is less than
        min_frac * target_sd_size
    """
    conn = np.asarray(conn, dtype=int)
    nelems = conn.shape[0]

    neighbors = build_element_adjacency(conn)
    subdomains = -np.ones(nelems, dtype=int)
    visited = np.zeros(nelems, dtype=bool)

    def _get_sd_corners(c_sd_elems):
        nodes = []
        for e in c_sd_elems:
            nodes += list(conn[e])
        nodes = np.unique(np.array(nodes, dtype=int))

        node_dict = {n: i for i, n in enumerate(nodes)}

        node_elem_cts = np.zeros(nodes.shape[0], dtype=np.int32)
        for e in c_sd_elems:
            for n in conn[e]:
                node_elem_cts[node_dict[n]] += 1

        num_corners = np.sum(node_elem_cts == 1)
        return int(num_corners)

    def _count_total_vertices(c_subdomains):
        """
        Global BDDC-style vertex nodes:
        node belongs to 3 or more distinct subdomains
        """
        node_to_subdomains = defaultdict(set)

        for e, elem_nodes in enumerate(conn):
            sd = int(c_subdomains[e])
            for n in elem_nodes:
                node_to_subdomains[int(n)].add(sd)

        total_vertices = sum(1 for sds in node_to_subdomains.values() if len(sds) >= 3)
        return int(total_vertices)

    def _compress_subdomain_ids(c_subdomains):
        """
        Re-label subdomains so IDs are contiguous: 0,1,2,...
        """
        unique_sds = np.unique(c_subdomains)
        mapping = {old_sd: new_sd for new_sd, old_sd in enumerate(unique_sds)}
        return np.array([mapping[sd] for sd in c_subdomains], dtype=int)

    def _get_adjacent_subdomains(c_subdomains, sd):
        """
        Return subdomain IDs adjacent to subdomain sd.
        """
        sd_elems = np.where(c_subdomains == sd)[0]
        adj_sds = set()

        for e in sd_elems:
            for nb in neighbors[e]:
                nb_sd = int(c_subdomains[nb])
                if nb_sd != sd:
                    adj_sds.add(nb_sd)

        return sorted(adj_sds)

    # ------------------------------------------------------------------
    # Phase 1: initial greedy construction
    # ------------------------------------------------------------------
    subdomain_ind = 0
    while not np.all(visited):

        # get next unvisited seed element
        elem = -1
        for _elem in range(nelems):
            if not visited[_elem]:
                elem = _elem
                break

        if elem == -1:
            break

        # initialize subdomain
        sd_elems = [elem]
        subdomains[elem] = subdomain_ind
        visited[elem] = True

        while len(sd_elems) < target_sd_size:

            # current frontier: all unvisited elems adjacent to current subdomain
            frontier = []
            for e in sd_elems:
                frontier += list(neighbors[e])

            frontier = list({
                e for e in frontier
                if (not visited[e]) and (e not in sd_elems)
            })

            if len(frontier) == 0:
                break

            # rank the whole frontier
            scores = []
            sd_set = set(sd_elems)
            for nb_elem in frontier:
                proposed_sd_elems = sd_elems + [nb_elem]
                num_corners = _get_sd_corners(proposed_sd_elems)

                # tie-breaker: prefer candidates touching more of current sd
                shared_adj = sum((nb in sd_set) for nb in neighbors[nb_elem])

                scores.append((num_corners, -shared_adj, nb_elem))

            scores.sort()

            # add as many frontier elems as possible this round
            added_any = False
            sd_elem_set = set(sd_elems)
            for _, _, best_elem in scores:
                if len(sd_elems) >= target_sd_size:
                    break
                if visited[best_elem]:
                    continue

                # still must touch current subdomain
                if not any((nb in sd_elem_set) for nb in neighbors[best_elem]):
                    continue

                sd_elems.append(best_elem)
                sd_elem_set.add(best_elem)
                subdomains[best_elem] = subdomain_ind
                visited[best_elem] = True
                added_any = True

            if not added_any:
                break

        subdomain_ind += 1

    subdomains = _compress_subdomain_ids(subdomains)

    # ------------------------------------------------------------------
    # Phase 2: merge very small subdomains if it reduces total vertices
    # ------------------------------------------------------------------
    min_sd_size = max(1, int(np.floor(min_frac * target_sd_size)))

    improved = True
    while improved:
        improved = False

        unique_sds, counts = np.unique(subdomains, return_counts=True)
        sd_sizes = {int(sd): int(ct) for sd, ct in zip(unique_sds, counts)}

        # smallest first
        small_sds = [sd for sd in unique_sds if sd_sizes[int(sd)] < min_sd_size]
        small_sds = sorted(small_sds, key=lambda sd: sd_sizes[int(sd)])

        if len(small_sds) == 0:
            break

        current_total_vertices = _count_total_vertices(subdomains)

        for small_sd in small_sds:
            # it may have disappeared after an earlier merge in this pass
            if small_sd not in np.unique(subdomains):
                continue

            adj_sds = _get_adjacent_subdomains(subdomains, int(small_sd))
            if len(adj_sds) == 0:
                continue

            # baseline = do nothing
            best_subdomains = subdomains.copy()
            best_total_vertices = current_total_vertices
            best_target_sd = None

            # test merge into each adjacent subdomain
            for adj_sd in adj_sds:
                trial = subdomains.copy()
                trial[trial == small_sd] = adj_sd
                trial = _compress_subdomain_ids(trial)

                trial_total_vertices = _count_total_vertices(trial)

                if trial_total_vertices < best_total_vertices:
                    best_total_vertices = trial_total_vertices
                    best_subdomains = trial
                    best_target_sd = adj_sd

            # accept only if strictly better
            if best_target_sd is not None:
                subdomains = best_subdomains
                current_total_vertices = best_total_vertices
                improved = True
                break  # restart scan since IDs changed

    return subdomains


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # DEMO 1: Generate a test mesh automatically
    # ------------------------------------------------------------------
    from _base import generate_structured_quad_mesh_gmesh, assign_subdomains_greedy, print_subdomain_report, plot_subdomain_mesh
    from _base import make_structured_quad_mesh

    # 1) Generate a genuinely unstructured quad mesh on a rectangular plate
    # xy, conn = generate_structured_quad_mesh_gmesh(
    #     lx=4.0,
    #     ly=2.0,
    #     target_size=0.18,
    #     algorithm=8,
    #     recomb_algorithm=2,
    # )

    nx, ny = 12, 8
    xy, conn = make_structured_quad_mesh(
        nx=nx,
        ny=ny,
        lx=4.0,
        ly=2.5,
        jitter=0.22,   # makes geometry look less structured
        seed=4,
    )

    # Example partition for testing
    # nsub = 9
    # subdomains = assign_subdomains_greedy(conn, nsub=nsub)

    subdomains = assign_bddc_subdomains(conn, target_sd_size=4)

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
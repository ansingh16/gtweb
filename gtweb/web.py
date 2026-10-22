"""
Step 3 of the IC-seeded ground-truth pipeline: the closed-form power-diagram web.

The web is the dual of the regular (weighted Delaunay) triangulation of the nuclei:
  * nodes     = weighted circumcentres (power centres) of the tetrahedra,
  * filaments = straight segments joining adjacent power centres (dual of internal
                Delaunay triangles),
so it is exact and finder-independent -- no DisPerSE run on the truth, no circularity.

The regular triangulation is the lower convex hull of the lifted points (scipy.spatial.ConvexHull);
periodicity is handled by padding the nuclei with their periodic images.

NB: a power-diagram node is one tetrahedron of the regular triangulation, so every node
has exactly 4 filaments (degree-4 connectivity pinning) -- an inherent property of the
construction, documented in the paper.
"""
import numpy as np
from scipy.spatial import ConvexHull


def regular_tetra(pts, wts):
    """Lower convex hull of (q, |q|^2 - w) -> regular-triangulation tetrahedra."""
    lift = (pts ** 2).sum(1) - wts
    P = np.column_stack([pts, lift])
    hull = ConvexHull(P, qhull_options="Qt")
    low = hull.equations[:, 3] < -1e-9                  # outward normal points down in lift axis = lower hull
    return hull, low


def power_centres(pts, wts, simplices):
    """Weighted circumcentre y of each tetra: 2 y.(q_b - q_0) = (|q_b|^2 - w_b) - (|q_0|^2 - w_0)."""
    q = pts[simplices]                                   # (T,4,3)
    w = wts[simplices]                                   # (T,4)
    lift = (q ** 2).sum(2) - w                           # (T,4)
    A = 2.0 * (q[:, 1:, :] - q[:, :1, :])                # (T,3,3)
    rhs = lift[:, 1:] - lift[:, :1]                      # (T,3)
    y = np.full((len(simplices), 3), np.nan)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-12
    y[ok] = np.linalg.solve(A[ok], rhs[ok][:, :, None])[:, :, 0]
    return y


def padded_nuclei(c, w, box, margin):
    """Periodic images of the (irregular) nucleus set within a margin shell.
    Returns pts, wts, base_id and the integer image shift of each point."""
    n = len(c)
    base0 = np.arange(n)
    pts, wts, base, shift = [c], [w], [base0], [np.zeros((n, 3), int)]
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if sx == sy == sz == 0:
                    continue
                ci = c + np.array([sx, sy, sz]) * box
                keep = np.all((ci > -margin) & (ci < box + margin), axis=1)
                if keep.any():
                    pts.append(ci[keep]); wts.append(w[keep]); base.append(base0[keep])
                    shift.append(np.tile([sx, sy, sz], (int(keep.sum()), 1)))
    return np.vstack(pts), np.concatenate(wts), np.concatenate(base), np.vstack(shift)


def _tetra_key(verts, base_id, shift, cell):
    """Translation-invariant identity of a tetrahedron: its four (nucleus, image) pairs, expressed
    relative to the periodic copy `cell` that holds its power centre."""
    return tuple(sorted((int(base_id[v]),) + tuple(int(x) for x in shift[v] - cell) for v in verts))


def required_margin(pts, wts, simplices, centres, facets, box):
    """Padding needed for the power vertices `facets` to be exact: a nucleus can invalidate the
    vertex y only if |y - q|^2 - w_q < R^2, i.e. if it lies within sqrt(R^2 + max w) of y."""
    if len(facets) == 0:
        return 0.0
    y = centres[facets]
    q0 = pts[simplices[facets, 0]]
    R2 = ((y - q0) ** 2).sum(1) - wts[simplices[facets, 0]]
    reach = np.sqrt(np.maximum(R2 + wts.max(), 0.0))[:, None]
    yw = y - box * np.floor(y / box)
    return float(np.maximum(reach - yw, reach - (box - yw)).max().clip(0))


def build_web(hull, low, pts, wts, centres, box, resample, max_len, base_id, shift):
    """Nodes and filament spines of the periodic power diagram.

    Nodes are the lower-hull tetrahedra whose power centre lies in [0, box)^3.  Every tetrahedron is
    identified exactly by its four (nucleus, periodic image) pairs, so a neighbour that lies outside
    the box is matched to its in-box copy without any positional tolerance, and each edge is stored
    once.  Spines are stored WRAPPED (%box); feed them to a periodic KDTree as point clouds (never
    diff() a stored spine without a min-image step)."""
    simp = hull.simplices                                # (F,4) point indices
    nbr = hull.neighbors                                 # (F,4) neighbour facet per opposite-vertex ridge
    with np.errstate(invalid='ignore'):                 # non-lower hull facets have undefined centres
        cell = np.floor(centres / box)
        inbox = low & np.all(cell == 0, axis=1)
    node_facets = np.where(inbox)[0]
    key_to_node = {_tetra_key(simp[f], base_id, shift, np.zeros(3, int)): i for i, f in enumerate(node_facets)}
    nodes = centres[node_facets]
    q = pts[simp[node_facets]]
    vol = np.abs(np.einsum('ti,ti->t', q[:, 1] - q[:, 0],
                           np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]))) / 6.0

    seen = set()
    edges, spines, fil_len, fil_area, fil_rf = [], [], [], [], []
    n_unmatched = 0
    for i, f in enumerate(node_facets):
        cf = centres[f]
        for r in range(4):
            g = nbr[f, r]
            if g < 0 or not low[g]:
                continue
            cg_cell = cell[g].astype(int)
            j = key_to_node.get(_tetra_key(simp[g], base_id, shift, cg_cell))
            if j is None:                                # padding too thin: caller enlarges it
                n_unmatched += 1
                continue
            ek = (i, j, tuple(cg_cell)) if i <= j else (j, i, tuple(-cg_cell))
            if ek in seen:
                continue
            seen.add(ek)
            cg = centres[g]
            L = np.linalg.norm(cg - cf)
            if L < 1e-9 or (max_len is not None and L > max_len):
                continue
            shared = simp[f][np.arange(4) != r]
            tri = pts[shared]
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            ts = np.linspace(0, 1, resample)
            sp = (cf[None, :] + ts[:, None] * (cg - cf)[None, :]) % box
            edges.append((i, j)); spines.append(sp.astype(np.float32)); fil_len.append(L)
            fil_area.append(area)
            fil_rf.append(np.clip(0.12 * L, 0.3, 1.2))
    return dict(nodes=nodes.astype(np.float32), node_vol=vol, spines=spines, edges=np.array(edges, int).reshape(-1, 2),
                fil_len=np.array(fil_len), fil_area=np.array(fil_area), fil_rf=np.array(fil_rf),
                n_unmatched=n_unmatched, node_facets=node_facets)


def build_web_from_geometry(nuclei, weights, box, margin=None, max_len=None, resample=6):
    """Convenience wrapper: (nuclei, weights) -> dict with nodes, spines (list of (resample,3) polylines),
    edges (node-index pairs), fil_len, node_degree.

    `margin` is the starting width of the periodic-image shell (default 0.32 box); it is widened until
    every node is provably exact.  `max_len` optionally drops edges longer than it (default: keep all)."""
    c = np.asarray(nuclei, float) % box
    w = np.asarray(weights, float)
    margin = 0.32 * box if margin is None else float(margin)
    for _ in range(6):
        pts, wts, base, shift = padded_nuclei(c, w, box, margin)
        hull, low = regular_tetra(pts, wts)
        centres = power_centres(pts, wts, hull.simplices)
        web = build_web(hull, low, pts, wts, centres, box, resample, max_len, base, shift)
        need = required_margin(pts, wts, hull.simplices, centres, web['node_facets'], box)
        if web['n_unmatched'] == 0 and need <= margin:
            break
        if margin >= box:
            raise RuntimeError("periodic power diagram did not converge with a full-box image shell")
        margin = min(box, max(1.25 * need, 1.5 * margin))
    else:
        raise RuntimeError("periodic power diagram did not converge")

    nodes = web['nodes']
    deg = np.bincount(web['edges'].ravel(), minlength=len(nodes)) if len(web['edges']) else np.zeros(len(nodes), int)
    web.pop('node_facets'); web.pop('n_unmatched')
    web.update(fil_len=web['fil_len'].astype(np.float32), node_degree=deg, margin=np.float64(margin),
               box_size=np.float64(box), n_filaments=np.int32(len(web['spines'])))
    return web

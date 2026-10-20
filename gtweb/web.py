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
from scipy.spatial import ConvexHull, cKDTree


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
    """Periodic images of the (irregular) nucleus set within a margin shell; returns pts, wts, base_id."""
    n = len(c)
    base0 = np.arange(n)
    pts, wts, base = [c], [w], [base0]
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if sx == sy == sz == 0:
                    continue
                ci = c + np.array([sx, sy, sz]) * box
                keep = np.all((ci > -margin) & (ci < box + margin), axis=1)
                if keep.any():
                    pts.append(ci[keep]); wts.append(w[keep]); base.append(base0[keep])
    return np.vstack(pts), np.concatenate(wts), np.concatenate(base)


def build_web(hull, low, pts, wts, centres, box, resample, max_len, base_id):
    """Nodes and filament spines from the lower-hull adjacency, restricted (periodically) to the box.
    Spines are stored WRAPPED (%box); feed them to a periodic KDTree as point clouds (never diff()
    a stored spine without a min-image step)."""
    def minimg(d):
        return d - box * np.round(d / box)

    simp = hull.simplices                                # (F,4) point indices
    nbr = hull.neighbors                                 # (F,4) neighbour facet per opposite-vertex ridge
    inbox = np.all((centres >= 0) & (centres < box), axis=1) & low
    with np.errstate(invalid='ignore'):                 # non-lower hull facets have undefined centres
        cen_w = centres % box

    node_facets = np.where(inbox)[0]
    nodes = cen_w[node_facets]
    q = pts[simp[node_facets]]
    vol = np.abs(np.einsum('ti,ti->t', q[:, 1] - q[:, 0],
                           np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]))) / 6.0

    seen = set()
    spines, fil_len, fil_area, fil_rf = [], [], [], []
    for f in node_facets:
        cf = centres[f]
        for r in range(4):
            g = nbr[f, r]
            if g < 0 or not low[g] or g == f:
                continue
            cg = cf + minimg(centres[g] - cf)
            ea = tuple(np.round((cf % box) * 2).astype(int))
            eb = tuple(np.round((cg % box) * 2).astype(int))
            key = (ea, eb) if ea <= eb else (eb, ea)
            if key in seen:
                continue
            seen.add(key)
            shared = np.array([v for v in simp[f] if v != simp[f][r]])
            tri = pts[shared]
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            L = np.linalg.norm(cg - cf)
            if L < 1e-6 or L > max_len:                 # drop far-image artifacts (length > void size)
                continue
            ts = np.linspace(0, 1, resample)
            sp = (cf[None, :] + ts[:, None] * (cg - cf)[None, :]) % box
            spines.append(sp.astype(np.float32)); fil_len.append(L)
            fil_area.append(area)
            fil_rf.append(np.clip(0.12 * L, 0.3, 1.2))
    return dict(nodes=nodes.astype(np.float32), node_vol=vol, spines=spines,
                fil_len=np.array(fil_len), fil_area=np.array(fil_area), fil_rf=np.array(fil_rf))


def build_web_from_geometry(nuclei, weights, box, margin=None, max_len=None, resample=6):
    """Convenience wrapper: (nuclei, weights) -> dict with nodes, spines (list of (resample,3) polylines),
    fil_len, node_degree.  `margin` and `max_len` default to box-scaled values."""
    c = np.asarray(nuclei) % box
    w = np.asarray(weights)
    if margin is None:
        margin = 0.32 * box
    if max_len is None:
        max_len = 0.45 * box
    pts, wts, base = padded_nuclei(c, w, box, margin)
    hull, low = regular_tetra(pts, wts)
    centres = power_centres(pts, wts, hull.simplices)
    web = build_web(hull, low, pts, wts, centres, box, resample, max_len, base)

    spines = web['spines']; fl = web['fil_len']
    keep = fl > 1e-3
    if not keep.all():
        spines = [s for s, kf in zip(spines, keep) if kf]
        fl = fl[keep]
        web['fil_area'] = web['fil_area'][keep]; web['fil_rf'] = web['fil_rf'][keep]
    nodes = web['nodes']
    deg = np.zeros(len(nodes), int)
    if len(spines):
        nt = cKDTree(nodes % box, boxsize=box)
        ends = np.vstack([np.vstack([s[0], s[-1]]) for s in spines]) % box
        _, idx = nt.query(ends, k=1)
        for i in idx:
            deg[i] += 1
    web.update(spines=spines, fil_len=fl.astype(np.float32), node_degree=deg,
               box_size=np.float64(box), n_filaments=np.int32(len(spines)))
    return web

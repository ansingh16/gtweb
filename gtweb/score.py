"""
Score any filament finder against the analytic ground-truth spine network.

The scoring is symmetric and geometric, at a spatial tolerance `tol` (Mpc/h), on the
periodic box:

  completeness = fraction of GROUND-TRUTH spine sample points within `tol` of the finder skeleton
  purity       = fraction of FINDER skeleton sample points within `tol` of a ground-truth spine

Both the GT spines and the finder skeleton are first densified to a fixed step so the
metrics do not depend on how coarsely either was sampled.

A finder skeleton may be supplied as EITHER
  * a list of (M_k, 3) polylines (its filament segments), or
  * an (N, 3) point cloud of skeleton samples.
Ground-truth spines are the list of (6, 3) polylines in `ground_truth_spines.npz`
(load them with gtweb.io.load_ground_truth).
"""
import numpy as np
from scipy.spatial import cKDTree


def _densify_polylines(polylines, box, step):
    """Resample each polyline to <= `step` spacing (min-image on the periodic box); return (P,3) points."""
    out = []
    for s in polylines:
        p = np.asarray(s, float)
        if len(p) < 2:
            out.append(p % box); continue
        d = np.diff(p, axis=0)
        d -= box * np.round(d / box)                       # min-image steps (spines may be stored wrapped)
        seglen = np.linalg.norm(d, axis=1)
        acc = [p[0:1] % box]
        cur = p[0].copy()
        for k in range(len(d)):
            n = max(1, int(np.ceil(seglen[k] / step)))
            ts = np.arange(1, n + 1) / n
            pts = cur[None, :] + ts[:, None] * d[k][None, :]
            acc.append(pts % box)
            cur = cur + d[k]
        out.append(np.vstack(acc))
    return np.vstack(out) if out else np.zeros((0, 3))


def _as_points(skeleton, box, step):
    """Coerce a finder skeleton (list of polylines OR (N,3) point cloud) to (N,3) sample points."""
    if isinstance(skeleton, np.ndarray) and skeleton.ndim == 2 and skeleton.shape[1] == 3:
        return skeleton.astype(float) % box
    # otherwise treat as a list/sequence of polylines
    return _densify_polylines(list(skeleton), box, step)


def score_finder(finder_skeleton, gt_spines, box, tol=1.0, step=0.3):
    """Completeness / purity / C-A of a finder skeleton vs the ground-truth spines.

    Parameters
    ----------
    finder_skeleton : list of (M,3) polylines, or (N,3) array of skeleton points.
    gt_spines       : list of (6,3) ground-truth spine polylines.
    box             : box size (Mpc/h), for periodic wrapping.
    tol             : matching tolerance (Mpc/h).
    step            : densification spacing (Mpc/h).

    Returns
    -------
    dict with completeness, purity, C_minus_A (= completeness - purity-complement proxy),
    n_gt_samples, n_finder_samples, tol.
    """
    gt_pts = _densify_polylines(list(gt_spines), box, step)
    fd_pts = _as_points(finder_skeleton, box, step)
    if len(gt_pts) == 0 or len(fd_pts) == 0:
        return dict(completeness=0.0, purity=0.0, C_minus_A=0.0,
                    n_gt_samples=len(gt_pts), n_finder_samples=len(fd_pts), tol=float(tol))

    fd_tree = cKDTree(fd_pts, boxsize=box)
    gt_tree = cKDTree(gt_pts, boxsize=box)
    d_gt, _ = fd_tree.query(gt_pts, k=1, workers=-1)       # GT -> nearest finder
    d_fd, _ = gt_tree.query(fd_pts, k=1, workers=-1)       # finder -> nearest GT
    completeness = float(np.mean(d_gt <= tol))
    purity = float(np.mean(d_fd <= tol))
    return dict(completeness=completeness, purity=purity,
                C_minus_A=float(completeness - (1.0 - purity)),
                n_gt_samples=int(len(gt_pts)), n_finder_samples=int(len(fd_pts)), tol=float(tol))


def score_per_filament(finder_skeleton, gt_spines, box, tol=1.0, step=0.3):
    """Per-filament completeness: for each ground-truth spine, the fraction of its (densified) length
    within `tol` of the finder skeleton. Returns an array of shape (len(gt_spines),) in [0,1].

    Pair it with the `filament_target_contrast` (delta_vol) or `filament_lengths` dial from
    load_ground_truth() to ask WHICH filaments a finder recovers (e.g. are faint ones missed?)."""
    fd_pts = _as_points(finder_skeleton, box, step)
    n = len(gt_spines)
    if len(fd_pts) == 0:
        return np.zeros(n)
    pts_list, ids = [], []
    for i, s in enumerate(gt_spines):
        p = _densify_polylines([s], box, step)
        pts_list.append(p); ids.append(np.full(len(p), i))
    pts = np.vstack(pts_list); ids = np.concatenate(ids)
    d, _ = cKDTree(fd_pts, boxsize=box).query(pts, k=1, workers=-1)
    hit = (d <= tol).astype(float)
    comp = np.zeros(n); cnt = np.zeros(n)
    np.add.at(comp, ids, hit); np.add.at(cnt, ids, 1.0)
    return comp / np.maximum(cnt, 1.0)

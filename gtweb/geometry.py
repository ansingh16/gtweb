"""
Step 1-2 of the IC-seeded ground-truth pipeline: turn an initial-condition
particle field into the void nuclei + Laguerre weights that define the web.

`box` and the grids are explicit arguments, so it runs on an IC of any box size and
resolution.  Pure numpy/scipy.

Pipeline:
  1. CIC the (already growth-scaled to z=0) IC particles -> 1+delta.
  2. Grow spherical underdensities about the density minima to rho_v -> void
     centres + radii (broad by construction; Jennings+2013 rho_v = 0.2).
  3. Void-in-void rejection ("largest crossing wins") + shrink-to-tangency.
  4. Inverse-Laguerre fit: choose weights so each power cell has volume V_i propto R_i^3
     (semi-discrete optimal-transport dual; gradient ascent, Lautensack & Zuyev 2008).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.spatial import cKDTree

try:                                    # optional numba acceleration for the CIC loop
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                       # pragma: no cover - numba is optional
    _HAVE_NUMBA = False


# --------------------------------------------------------------------------- CIC density
def _cic_numpy(positions, grid_size, box_size):
    """Vectorised cloud-in-cell (periodic) via scatter-add; the numba-free fallback."""
    cell = box_size / grid_size
    f = (positions % box_size) / cell
    i0 = np.floor(f).astype(np.int64)
    d = f - i0
    out = np.zeros(grid_size ** 3, np.float64)
    for cx in (0, 1):
        wx = d[:, 0] if cx else 1.0 - d[:, 0]; ix = (i0[:, 0] + cx) % grid_size
        for cy in (0, 1):
            wy = d[:, 1] if cy else 1.0 - d[:, 1]; iy = (i0[:, 1] + cy) % grid_size
            for cz in (0, 1):
                wz = d[:, 2] if cz else 1.0 - d[:, 2]; iz = (i0[:, 2] + cz) % grid_size
                lin = (ix * grid_size + iy) * grid_size + iz
                out += np.bincount(lin, weights=wx * wy * wz, minlength=grid_size ** 3)
    return out.reshape(grid_size, grid_size, grid_size)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _cic_numba(positions, grid_size, box_size):        # pragma: no cover - exercised when numba present
        n = positions.shape[0]
        cell = box_size / grid_size
        grid = np.zeros((grid_size, grid_size, grid_size))
        for p in range(n):
            x = (positions[p, 0] % box_size) / cell
            y = (positions[p, 1] % box_size) / cell
            z = (positions[p, 2] % box_size) / cell
            ix = int(np.floor(x)); iy = int(np.floor(y)); iz = int(np.floor(z))
            dx = x - ix; dy = y - iy; dz = z - iz
            for ii in range(2):
                wx = dx if ii == 1 else (1.0 - dx); gx = (ix + ii) % grid_size
                for jj in range(2):
                    wy = dy if jj == 1 else (1.0 - dy); gy = (iy + jj) % grid_size
                    for kk in range(2):
                        wz = dz if kk == 1 else (1.0 - dz); gz = (iz + kk) % grid_size
                        grid[gx, gy, gz] += wx * wy * wz
        return grid


def cic_density(positions, grid_size, box_size):
    """Periodic CIC density grid.  Uses numba if available, else a numpy scatter-add."""
    pos = np.ascontiguousarray(positions.astype(np.float64))
    if _HAVE_NUMBA:
        return _cic_numba(pos, int(grid_size), float(box_size))
    return _cic_numpy(pos, int(grid_size), float(box_size))


def ic_one_plus_delta(ic_positions, grid, box):
    """1 + delta of the (growth-scaled) IC particle field on a `grid`^3 CIC mesh."""
    rho = cic_density(np.asarray(ic_positions) % box, grid, box)
    return rho / rho.mean()


# --------------------------------------------------------------------------- void geometry
def spherical_voids(one_plus_delta, box, rho_v=0.2, smooth=1.5, r_min=1.5, r_max=22.0):
    """Spherical-underdensity voids: grow a sphere about each density minimum until the enclosed
    mean density reaches rho_v, then void-in-void reject (largest first, drop a candidate whose
    centre lies inside an accepted larger void).  Returns (centres[N,3], radii[N])."""
    grid = one_plus_delta.shape[0]; cell = box / grid
    sm = gaussian_filter(one_plus_delta, smooth / cell, mode='wrap')
    minima = sm == minimum_filter(sm, size=3, mode='wrap')
    ci = np.argwhere(minima); ci = ci[np.argsort(sm[minima])]      # deepest first
    Rc = int(np.ceil(r_max / cell)); ax = np.arange(-Rc, Rc + 1)
    OX, OY, OZ = np.meshgrid(ax, ax, ax, indexing='ij')
    dist = np.sqrt(OX ** 2 + OY ** 2 + OZ ** 2) * cell
    m = dist <= r_max
    offs = np.stack([OX[m], OY[m], OZ[m]], 1); dphys = dist[m]
    o = np.argsort(dphys); offs = offs[o]; dphys = dphys[o]
    radii, cent = [], []
    for c in ci:
        idx = (c + offs) % grid
        vals = one_plus_delta[idx[:, 0], idx[:, 1], idx[:, 2]]
        cummean = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        below = np.where(cummean <= rho_v)[0]
        if len(below) == 0:
            continue
        r = dphys[below.max()]
        if r >= r_min:
            radii.append(r); cent.append(c.astype(np.float64) * cell)
    radii = np.array(radii); cent = np.array(cent)
    order = np.argsort(-radii)
    acc_c = np.empty((0, 3)); acc_r = np.empty(0)
    for j in order:
        c, r = cent[j], radii[j]
        if len(acc_r):
            d = c - acc_c; d -= box * np.round(d / box)
            if np.any(np.linalg.norm(d, axis=1) < acc_r):     # centre inside a larger accepted void -> nested
                continue
        acc_c = np.vstack([acc_c, c]); acc_r = np.append(acc_r, r)
    return acc_c, acc_r


def shrink_to_tangency(c, r, box, r_floor):
    """Trim radii (largest first) so no two spheres overlap: R_i <- min(R_i, min_j(d_ij - R_j)); drop
    any that fall below r_floor.  Removes burial and puts every pair at/beyond contact."""
    order = np.argsort(-r)
    cc = c[order].copy(); rr = r[order].copy()
    keep = np.ones(len(rr), bool)
    for i in range(1, len(rr)):
        d = cc[:i] - cc[i]; d -= box * np.round(d / box)
        dd = np.linalg.norm(d, axis=1)
        lim = np.min(dd - rr[:i])
        rr[i] = min(rr[i], lim)
        if rr[i] < r_floor:
            keep[i] = False
    return cc[keep], rr[keep]


# --------------------------------------------------------------------------- Laguerre power cells / fit
def power_cells(nuc, w, grid, box, k):
    """R_eff, count, owner of each grid point under power distance |x-c|^2 - w (k Euclidean-nearest
    then power-argmin).  Same estimator as 01_laguerre_geometry_check.py."""
    cell = box / grid
    ax = (np.arange(grid) + 0.5) * cell
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    t = cKDTree(nuc % box, boxsize=box)
    kk = min(k, len(nuc))
    d, idx = t.query(pts, k=kk, workers=-1)
    if kk == 1:
        idx = idx[:, None]; d = d[:, None]
    own = idx[np.arange(len(pts)), np.argmin(d ** 2 - w[idx], axis=1)]
    cnt = np.bincount(own, minlength=len(nuc)).astype(float)
    V = cnt * cell ** 3
    return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0), cnt, own


def laguerre_fit(c, void_radii, box, grid=112, k=64, iters=260, eta=0.6, decay=60.0, verbose=False):
    """Inverse-Laguerre weights so power-cell volumes hit V_i propto R_i^3.  Gradient ascent on the
    concave semi-discrete OT dual (grad_i = V_i^target - V_i(w)); per-cell length^2 step scale s_i =
    Vt_i^{2/3} + 1/(1+it/decay) annealing.  Returns weights w[N]."""
    Vt = box ** 3 * void_radii ** 3 / (void_radii ** 3).sum()     # prescribed volumes propto R^3
    w = np.zeros(len(c))
    s_i = Vt ** (2.0 / 3.0)
    best = (np.inf, w.copy())
    for it in range(iters):
        _, cnt, _ = power_cells(c, w, grid, box, k)
        V = cnt * (box / grid) ** 3
        rel = (Vt - V) / Vt
        lr = eta / (1.0 + it / decay)
        w = w + lr * s_i * rel
        w -= w.mean()                                             # gauge freedom
        mre = float(np.mean(np.abs(rel)))
        n_empty = int((cnt == 0).sum())
        if mre < best[0] and n_empty == 0:
            best = (mre, w.copy())
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"    it {it:4d}  mean|dV|/V {mre:6.3f}  empty {n_empty:4d}  lr {lr:.3f}", flush=True)
    return best[1] if np.isfinite(best[0]) else w


def build_geometry(ic_positions, box, grid=128, rho_v=0.2, smooth=1.5, r_min=1.5, r_max=22.0,
                   shrink=True, fit_grid=112, fit_k=64, fit_iters=260, verbose=False):
    """Full step 1-2: IC particles -> (nuclei, weights, void_radii).  `r_max` should be scaled with the
    box; the default (22 Mpc/h) is tuned for a 100 Mpc/h box."""
    opd = ic_one_plus_delta(ic_positions, grid, box)
    c, r = spherical_voids(opd, box, rho_v=rho_v, smooth=smooth, r_min=r_min, r_max=r_max)
    if len(c) < 5:
        raise ValueError(
            f"only {len(c)} void nuclei grown -- the IC field has too little structure for this "
            f"configuration. Provide a genuine growth-scaled IC (with real underdensities), or relax "
            f"rho_v (currently {rho_v}) / r_min ({r_min}) / increase the CIC grid ({grid}).")
    if shrink:
        c, r = shrink_to_tangency(c, r, box, r_min)
    w = laguerre_fit(c, r, box, grid=fit_grid, k=fit_k, iters=fit_iters, verbose=verbose)
    return dict(nuclei=c.astype(np.float64), nuclei_weights=w.astype(np.float64),
                void_radii=r.astype(np.float64), box_size=np.float64(box))

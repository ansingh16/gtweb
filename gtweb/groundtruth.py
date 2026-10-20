"""
Step 5 of the IC-seeded ground-truth pipeline: package the analytic web into the
ground-truth schema and measure the per-filament / per-node physical dials.

Dials (measured on the particle field so a user can bin filaments by strength):
  * per-node peak     = particle mass within CALIB_APERTURE (1.5 Mpc/h) of each node.
  * per-edge delta_vol = volumetric overdensity rho_core/rho_bar sampled along the spine
                         (256^3 grid smoothed 0.5 Mpc/h).  PRIMARY dial (filament_target_contrast).
  * per-edge lambda   = linear mass density in an R_cyl = 1 Mpc/h cylinder.  SECONDARY (legacy) dial.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, map_coordinates

from . import cosmology as cosmo

CALIB_APERTURE = 1.5
R_CYL = 1.0
VOL_NGRID, VOL_SMOOTH = 256, 0.5


def spine_overdensity(spines, pos, m_p, box, ngrid=VOL_NGRID, smooth=VOL_SMOOTH):
    """delta_vol = rho_core/rho_bar averaged ALONG each spine."""
    rho_bar = cosmo.Om * cosmo.RHO_CRIT
    cell = box / ngrid
    e = np.linspace(0, box, ngrid + 1)
    H, _ = np.histogramdd(pos % box, bins=(e, e, e))
    rho = gaussian_filter(H, smooth / cell, mode='wrap') * m_p / cell ** 3
    pts, fid = [], []
    for i, s in enumerate(spines):
        pts.append(np.asarray(s)[:, :3] % box); fid.append(np.full(len(s), i))
    pts = np.vstack(pts); fid = np.concatenate(fid)
    vals = map_coordinates(rho, (pts.T / cell), order=1, mode='grid-wrap') / rho_bar
    o = np.zeros(len(spines)); np.add.at(o, fid, vals)
    o /= np.maximum(np.bincount(fid, minlength=len(spines)), 1)
    return o.astype(np.float32)


def filament_linear_density(seg3, pos, m_p, box, r_cyl=R_CYL, step=0.3):
    """lambda_i = mass within r_cyl of straight edge i / length_i (nearest-edge-sample assignment)."""
    samp, eid = [], []
    for e, (x0, y0, z0, x1, y1, z1) in enumerate(seg3):
        p0 = np.array([x0, y0, z0]); p1 = np.array([x1, y1, z1]); L = np.linalg.norm(p1 - p0)
        ns = max(2, int(L / step))
        samp.append(p0 + np.linspace(0, 1, ns)[:, None] * (p1 - p0)); eid.append(np.full(ns, e))
    samp = np.vstack(samp); eid = np.concatenate(eid)
    d, idx = cKDTree(samp).query(pos, k=1, workers=-1)
    hit = d < r_cyl
    counts = np.bincount(eid[idx[hit]], minlength=len(seg3))
    length = np.linalg.norm(seg3[:, 3:] - seg3[:, :3], axis=1)
    lam = counts * m_p / np.maximum(length, 1e-6)
    return lam, np.maximum(length, 1e-6)


def assemble_ground_truth(web, nuclei, weights, pos, box):
    """Build the GT dict-of-arrays from an analytic `web` (from web.build_web_from_geometry)
    and the particle field.  Returns (gt, nodes_cat, walls_cat, web_geom) dicts ready to np.savez."""
    spines = web['spines']
    K = len(spines)
    nodes = np.asarray(web['nodes'], float) % box
    fl = np.asarray(web['fil_len'], np.float32)
    pos = np.asarray(pos).astype(np.float64) % box
    m_p = cosmo.Om * cosmo.RHO_CRIT * box ** 3 / len(pos)
    lambda0 = cosmo.Om * cosmo.RHO_CRIT * np.pi * R_CYL ** 2

    # per-node peak (mass within the calibration aperture)
    dn, inode = cKDTree(nodes, boxsize=box).query(pos, k=1, distance_upper_bound=CALIB_APERTURE, workers=-1)
    hitn = np.isfinite(dn)
    peaks = np.bincount(inode[hitn], minlength=len(nodes)).astype(np.float64) * m_p

    # per-edge straight-segment endpoints (min-image; spines are stored wrapped)
    seg3 = np.empty((K, 6))
    for i, s in enumerate(spines):
        p = np.asarray(s, float)
        dd = np.diff(p, axis=0); dd -= box * np.round(dd / box)
        a = p[0] % box
        seg3[i, :3] = a
        seg3[i, 3:] = a + dd.sum(0)
    lam, _ = filament_linear_density(seg3, pos, m_p, box, r_cyl=R_CYL)
    contrast = (lam / lambda0).astype(np.float32)

    # PRIMARY dial: volumetric overdensity along the spine
    odens = spine_overdensity(spines, pos, m_p, box)

    r_f = np.full(K, 1.0, np.float32)
    gt = {f'spine_{i}': np.asarray(s, np.float32) for i, s in enumerate(spines)}
    gt.update(n_filaments=np.int32(K), filament_lengths=fl,
              filament_sigmas=r_f, filament_radii=r_f,
              filament_target_contrast=odens, filament_contrasts=odens, filament_target_overdensity=odens,
              filament_linear_contrast=contrast, filament_lambda=lam.astype(np.float32),
              target_dial=np.array('volumetric_overdensity'),
              vol_ngrid=np.int32(VOL_NGRID), vol_smooth=np.float32(VOL_SMOOTH),
              filament_labels=np.array([f"fil{i}" for i in range(K)]),
              node_positions=nodes.astype(np.float32),
              pl3_gamma=np.float32(2.0), pl3_beta_out=np.float32(3.0),
              filament_rcore=(0.3 * r_f).astype(np.float32), filament_rout=(3.0 * r_f).astype(np.float32))

    nodes_cat = dict(positions=nodes, radii=np.full(len(nodes), CALIB_APERTURE),
                     r200=np.full(len(nodes), 1.0), peaks=peaks, n_hubs=np.int32(len(nodes)),
                     box_size=np.float64(box))
    walls_cat = dict(triangles=np.zeros((0, 3, 3), np.float32), tri_contrast=np.zeros(0, np.float32),
                     normals=np.zeros((0, 3), np.float32), sigma=np.float64(0.3), box_size=np.float64(box))
    web_geom = dict(nuclei=np.asarray(nuclei), nuclei_weights=np.asarray(weights),
                    vertices=nodes, box_size=np.float64(box))
    return gt, nodes_cat, walls_cat, web_geom

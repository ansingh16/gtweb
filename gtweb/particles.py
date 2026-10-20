"""
Step 4 of the IC-seeded ground-truth pipeline: paint particles onto the web.

Two stages (pure numpy/scipy):

  cascade_displace
    Displaces each IC particle radially out of its Laguerre (power) cell and projects it
    onto the cell's nearest face / edge / vertex -> lands it in a void / wall / filament /
    node (stage 0/1/2/3).  Big voids evacuate harder (per-void expansion T_i).

  halo_model_population
    Re-clumps the collapsed (filament+node) mass into NFW halos whose abundance follows the
    Sheth-Tormen (1999) mass function; a `diffuse_frac` of filament mass is kept smooth so
    the inter-halo bridges survive.

KPOW must be large (64) when the Laguerre weights span a broad range: with only 8
candidates the power-nearest cell is mis-assigned for ~12 per cent of points.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter

from . import cosmology as cosmo

# cascade constants
KPOW = 64                # Euclidean-nearest candidates refined to the power-nearest cell (broad weights need >8)
KNB_CAP = 64             # neighbour half-spaces bounding each cell
PART_CHUNK = 250_000          # particles per cascade chunk (memory ~ PART_CHUNK * KNB_CAP * 24 bytes per array)
EPS = 1e-9


def _argmax_excl(g, *excl):
    """Per-row argmax of g (m,K) with the given local-index columns masked out; returns (idx, value)."""
    gg = g.copy()
    rows = np.arange(len(g))
    for e in excl:
        gg[rows, e] = -np.inf
    j = np.argmax(gg, axis=1)
    return j, gg[rows, j]


def cascade_displace(part, nuclei, nbr_off, T, box, rng, scatter, nuc_w, nbr_w, Tvec, workers=-1):
    """Project the radially-scaled desired position onto the power (Laguerre) cell, landing on
    void/wall/filament/node.  Returns (pos[m,3], stage[m] in {0,1,2,3})."""
    tree = cKDTree(nuclei, boxsize=box)
    pos = np.empty_like(part)
    stage = np.zeros(len(part), np.uint8)
    ar = np.arange
    for lo in range(0, len(part), PART_CHUNK):
        hi = min(lo + PART_CHUNK, len(part)); m = hi - lo
        p = part[lo:hi]
        dk, idxk = tree.query(p, k=min(KPOW, len(nuclei)), workers=workers)   # power cell = argmin |x-c_k|^2 - w_k
        idx = idxk[ar(m), np.argmin(dk ** 2 - nuc_w[idxk], axis=1)]
        c = nuclei[idx]
        v0 = p - c; v0 -= box * np.round(v0 / box)
        W = nbr_off[idx]                                          # (m,K,3) neighbour offsets a_k = c_k - c_i
        wn2 = (W ** 2).sum(-1)
        half = 0.5 * wn2 + 0.5 * (nuc_w[idx][:, None] - nbr_w[idx])  # radical-plane offsets
        anorm = np.sqrt(wn2)
        Tp = Tvec[idx]
        vdes = (1.0 + np.asarray(Tp))[..., None] * v0

        v = vdes.copy()
        st = np.zeros(m, np.uint8)

        # stage 1: project x_des onto the most-violated face -> WALL
        g0 = np.einsum('mkj,mj->mk', W, vdes) - half
        jloc, gj_perp = _argmax_excl(g0 / anorm)
        viol1 = gj_perp > EPS
        a_j = W[ar(m), jloc]; wn2_j = wn2[ar(m), jloc]
        gj = g0[ar(m), jloc]
        v1 = vdes - (gj / wn2_j)[:, None] * a_j
        v = np.where(viol1[:, None], v1, v)
        st[viol1] = 1

        # stage 2: project onto the EDGE (planes j,k2) -> FILAMENT
        P = np.where(viol1)[0]
        if len(P):
            Wp = W[P]; halfp = half[P]; anp = anorm[P]
            g1 = np.einsum('mkj,mj->mk', Wp, v1[P]) - halfp
            jP = jloc[P]
            k2, g2_perp = _argmax_excl(g1 / anp, jP)
            viol2 = g2_perp > EPS
            Q = P[viol2]
            if len(Q):
                aj = W[Q, jloc[Q]]; ak = W[Q, k2[viol2]]
                G = np.empty((len(Q), 2, 2))
                G[:, 0, 0] = (aj * aj).sum(1); G[:, 0, 1] = (aj * ak).sum(1)
                G[:, 1, 0] = G[:, 0, 1];        G[:, 1, 1] = (ak * ak).sum(1)
                resid = np.stack([(aj * v1[Q]).sum(1) - half[Q, jloc[Q]],
                                  (ak * v1[Q]).sum(1) - half[Q, k2[viol2]]], axis=1)[:, :, None]
                lam = np.linalg.solve(G, resid)[:, :, 0]
                v2 = v1[Q] - lam[:, 0:1] * aj - lam[:, 1:2] * ak
                v[Q] = v2
                st[Q] = 2

                # stage 3: snap to the VERTEX (planes j,k2,k3) -> NODE
                Wq = W[Q]; halfq = half[Q]; anq = anorm[Q]
                g2 = np.einsum('mkj,mj->mk', Wq, v2) - halfq
                k3, g3_perp = _argmax_excl(g2 / anq, jloc[Q], k2[viol2])
                viol3 = g3_perp > EPS
                Rr = Q[viol3]
                if len(Rr):
                    A = np.stack([W[Rr, jloc[Rr]], W[Rr, k2[viol2][viol3]], W[Rr, k3[viol3]]], axis=1)
                    b = np.stack([half[Rr, jloc[Rr]], half[Rr, k2[viol2][viol3]], half[Rr, k3[viol3]]], axis=1)
                    det = np.linalg.det(A)
                    good = np.abs(det) > 1e-12
                    v3 = v2[viol3].copy()
                    if good.any():
                        v3[good] = np.linalg.solve(A[good], b[good][:, :, None])[:, :, 0]
                    v[Rr] = v3
                    st[Rr] = 3

        out = c + v
        if scatter > 0:
            out = out + rng.normal(0.0, scatter, out.shape)
        pos[lo:hi] = out % box
        stage[lo:hi] = st
    return pos, stage


def cascade_onto_geometry(ic_positions, nuclei, weights, void_radii, box, seed=0,
                          T_base=0.98, T_exp=0.5, scatter=0.10, workers=-1, verbose=False):
    """Wrapper: cascade the IC particles onto the Laguerre geometry.  Returns (positions, stage)."""
    nuc = np.asarray(nuclei) % box
    w = np.asarray(weights)
    R = np.asarray(void_radii)
    N = len(nuc)
    Tvec = T_base * (R / np.median(R)) ** T_exp
    KNB = min(N - 1, KNB_CAP)
    _, nidx = cKDTree(nuc, boxsize=box).query(nuc, k=KNB + 1, workers=workers)
    nbr_idx = nidx[:, 1:]
    nbr_off = nuc[nbr_idx] - nuc[:, None, :]; nbr_off -= box * np.round(nbr_off / box)
    nbr_w = w[nbr_idx]
    part = np.asarray(ic_positions).astype(np.float64) % box
    rng = np.random.default_rng(1000 + seed)
    pos, stage = cascade_displace(part, nuc, nbr_off, T_base, box, rng, scatter,
                                  nuc_w=w, nbr_w=nbr_w, Tvec=Tvec, workers=workers)
    if verbose:
        cnt = np.bincount(stage, minlength=4)
        names = ['void', 'wall', 'filament', 'node']
        for s in range(4):
            print(f"    {names[s]:>9}: {cnt[s]:>12,} ({100 * cnt[s] / len(part):5.1f}%)")
    return pos, stage


# --------------------------------------------------------------------------- halo model
def R_of_M(M):
    return (3.0 * M / (4.0 * np.pi * cosmo.RHO_M)) ** (1.0 / 3.0)


def rvir_of_M(M):
    return (3.0 * M / (4.0 * np.pi * cosmo.DELTA_VIR * cosmo.RHO_M)) ** (1.0 / 3.0)


_LNK = np.linspace(np.log(1e-4), np.log(50.0), 4096)
_KK = np.exp(_LNK)
_D3 = _KK ** 3 * cosmo.linear_Pk(_KK) / (2.0 * np.pi ** 2)


def _tophat_W(x):
    return np.where(x < 1e-3, 1.0 - x ** 2 / 10.0, 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3)


def _tophat_dWdx(x):
    return np.where(x < 1e-3, -x / 5.0,
                    3.0 * np.sin(x) / x ** 2 - 9.0 * (np.sin(x) - x * np.cos(x)) / x ** 4)


def sigma_and_dlninvsig(M):
    R = R_of_M(np.atleast_1d(M).astype(float))[:, None]
    x = _KK[None, :] * R
    W = _tophat_W(x); dW = _tophat_dWdx(x)
    s2 = np.trapezoid(_D3[None, :] * W ** 2, _LNK, axis=1)
    ds2_dlnR = np.trapezoid(_D3[None, :] * 2.0 * W * dW * x, _LNK, axis=1)
    return np.sqrt(s2), -(1.0 / 6.0) * ds2_dlnR / s2


def nfw_particles(centers, n_i, rvir, conc, rng, box):
    """Synthesise sum(n_i) NFW particles: per halo, inverse-CDF radii (self-similar in x=r/r_s) +
    isotropic directions about the centre."""
    N = int(n_i.sum())
    cen = np.repeat(centers, n_i, axis=0)
    rv = np.repeat(rvir, n_i); cc = np.repeat(conc, n_i)
    u = rng.random(N)
    cmax = float(cc.max())
    xs = np.linspace(0.0, cmax, 8192)
    mcum = np.log1p(xs) - xs / (1.0 + xs)
    m_at_c = np.log1p(cc) - cc / (1.0 + cc)
    x = np.interp(u * m_at_c, mcum, xs)
    r = x * (rv / cc)
    cost = rng.uniform(-1.0, 1.0, N); sint = np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    off = np.column_stack([r * sint * np.cos(phi), r * sint * np.sin(phi), r * cost])
    return (cen + off) % box


def _halo_centres(pos, stage, box, rng, grid=512, peak_smooth=0.3, m_min_part=20.0, diffuse_frac=0.4):
    """Halo centres + emergent particle counts + the diffuse (kept-in-place) mask.

    Centres are the density peaks of the collapsed (filament-non-diffuse + node) particles;
    each collapsed particle is assigned to its nearest peak and halos with >= `m_min_part`
    particles are kept (orphans folded into the nearest kept halo).  Consumes EXACTLY one
    `rng.random(N)` draw (the fil_diffuse split), so a caller that then synthesises NFW
    particles from the same `rng` gets an unchanged random stream.

    Returns (centers[H,3], n_i[H] particle counts, diffuse_mask[N])."""
    N = len(pos)
    keep_mask = np.isin(stage, [0, 1])
    is_fil = stage == 2; is_node = stage == 3
    f = float(np.clip(diffuse_frac, 0.0, 1.0))
    fil_diffuse = is_fil & (rng.random(N) < f)
    diffuse_mask = keep_mask | fil_diffuse
    C = pos[(is_fil & ~fil_diffuse) | is_node]
    N_C = len(C)

    g = grid; cell = box / g
    ix = np.minimum((C / cell).astype(np.int64), g - 1)
    flat = (ix[:, 0] * g + ix[:, 1]) * g + ix[:, 2]
    dens = np.bincount(flat, minlength=g ** 3).astype(np.float32).reshape(g, g, g)
    dens = gaussian_filter(dens, peak_smooth / cell, mode='wrap')
    peak = (dens == maximum_filter(dens, size=3, mode='wrap')) & (dens > 0)
    centers_all = (np.argwhere(peak).astype(np.float64) + 0.5) * cell

    _, lab = cKDTree(centers_all, boxsize=box).query(C, k=1, workers=-1)
    n_all = np.bincount(lab, minlength=len(centers_all))
    keep_h = n_all >= m_min_part
    centers = centers_all[keep_h]; n_i = n_all[keep_h].astype(np.int64)
    if (N_C - int(n_i.sum())) > 0 and len(centers):
        _, l2 = cKDTree(centers, boxsize=box).query(C[np.isin(lab, np.where(~keep_h)[0])], k=1, workers=-1)
        n_i += np.bincount(l2, minlength=len(centers))
    return centers, n_i, diffuse_mask


def _catalogue_dict(centers, n_i, box, m_p, m_min_part):
    return dict(positions=centers.astype(np.float32), mass=(n_i * m_p).astype(np.float32),
                n_particles=n_i.astype(np.int64), box_size=float(box),
                m_particle=float(m_p), min_npart=float(m_min_part))


def halo_catalogue(pos, stage, box, seed=0, grid=512, peak_smooth=0.3, m_min_part=20.0, diffuse_frac=0.4):
    """The mock's Sheth-Tormen/NFW halo CATALOGUE for a collapsed (pos, stage) field: the same
    halos `halo_model_population` re-clumps, returned as centres + emergent masses rather than
    NFW particles.  Returns dict(positions, mass, n_particles, box_size, m_particle, min_npart).
    Deterministic (seed).  Note: apply the same per-stage thickness to `pos` first if the field
    was built with it (see generate.generate_from_ic)."""
    pos = np.asarray(pos).astype(np.float64) % box
    stage = np.asarray(stage)
    m_p = cosmo.RHO_M * box ** 3 / len(pos)
    rng = np.random.default_rng(5000 + seed)
    centers, n_i, _ = _halo_centres(pos, stage, box, rng, grid=grid, peak_smooth=peak_smooth,
                                    m_min_part=m_min_part, diffuse_frac=diffuse_frac)
    return _catalogue_dict(centers, n_i, box, m_p, m_min_part)


def halo_model_population(pos, stage, box, seed=0, grid=512, peak_smooth=0.3,
                          m_min_part=20.0, conc=6.0, diffuse_frac=0.4, verbose=False,
                          return_catalogue=False):
    """Re-clump collapsed (filament+node) mass into NFW halos (masses emerge from the peaks).  Void+wall and a
    `diffuse_frac` of filaments are kept in place.  Returns (out_pos, out_stage) with len == len(pos).
    With `return_catalogue=True` also returns the halo catalogue dict (see `halo_catalogue`) as a third
    element -- the SAME centres/masses, at no extra cost."""
    pos = np.asarray(pos).astype(np.float64) % box
    stage = np.asarray(stage)
    N = len(pos)
    m_p = cosmo.RHO_M * box ** 3 / N
    rng = np.random.default_rng(5000 + seed)

    centers, n_i, diffuse_mask = _halo_centres(pos, stage, box, rng, grid=grid, peak_smooth=peak_smooth,
                                               m_min_part=m_min_part, diffuse_frac=diffuse_frac)
    rvir = rvir_of_M(n_i * m_p); conc_arr = np.full(len(centers), conc)
    halo_pos = nfw_particles(centers, n_i, rvir, conc_arr, rng, box)

    out_pos = np.vstack([pos[diffuse_mask], halo_pos])
    out_stage = np.concatenate([stage[diffuse_mask], np.full(len(halo_pos), 3, np.uint8)]).astype(np.uint8)
    if verbose:
        print(f"    halos {len(centers):,}; kept-in-place {int(diffuse_mask.sum()):,}; "
              f"reclumped {N - int(diffuse_mask.sum()):,}; N_out {len(out_pos):,} (== N {N})")
    result = (out_pos.astype(np.float32), out_stage)
    if return_catalogue:
        result = result + (_catalogue_dict(centers, n_i, box, m_p, m_min_part),)
    return result

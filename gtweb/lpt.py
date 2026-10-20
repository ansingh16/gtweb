"""
Truncated second-order Lagrangian perturbation theory (2LPT) for the diffuse component,
and its velocity field.

The mock fills the voids of the collapsed cascade with a fraction `f_diffuse` of a 2LPT
realisation of the same initial conditions, so the in-void matter carries uncollapsed,
second-order structure.  The only input is the first-order displacement Psi1 on the
Lagrangian grid (from gtweb.ic.gaussian_psi1, or from your own initial conditions).

Algebra (Scoccimarro 1998; Crocce, Pueblas & Scoccimarro 2006), normalised to D1(z=0)=1:

    x   = q - grad phi1 - (3/7) Om^(-1/143) grad phi2
    lap phi1 = delta1        (Psi1 = -grad phi1, i.e. what we bake)
    lap phi2 = sum_{i<j} [ phi1_,ii phi1_,jj - (phi1_,ij)^2 ]

Truncated LPT (Coles, Melott & Shandarin 1993): Gaussian-filter phi1 at scale Rs before displacing,
which maximises the cross-correlation with N-body.  Rs=0 is plain 2LPT (the production value).

The 2LPT peculiar velocity of the growing mode (z=0, a=1) is

    v = H0 * ( f1 * Psi1_s + f2 * Psi2 )      km/s      (Psi in Mpc/h, H0 = 100 h km/s/Mpc)

with f1 = Om^0.55, f2 = 2 Om^(6/11) (Bouchet et al. 1995).  We deposit it (momentum + mass CIC,
smoothed) onto an Eulerian cube -- exactly what velocity-based web classifiers ingest -- with the
same schema and cell-centre convention as gtweb.velocity, so gtweb.sample_velocity reads it directly.

Pure numpy/scipy; no hardcoded paths.
"""
import numpy as np
from scipy.fft import rfftn, irfftn
from scipy.ndimage import gaussian_filter

from . import cosmology as cosmo


def load_psi1(path):
    """Load a baked Psi1 grid -> dict with psi1 (3,n,n,n), box_size, ngrid, growth, rms_psi1."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['box_size'] = float(out['box_size'])
    if 'ngrid' in out:
        out['ngrid'] = int(out['ngrid'])
    return out


def truncated_2lpt(psi1, box, Rs=0.0, workers=-1):
    """Given Psi1[3,n,n,n] at z=0 linear amplitude, return (Psi1_s, Psi2, rms1, rms2) for
    truncation scale Rs (Gaussian on phi1; Rs=0 = plain 2LPT)."""
    P = np.asarray(psi1, np.float64)
    n = P.shape[1]
    kf = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kr = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    KX = kf[:, None, None]; KY = kf[None, :, None]; KZ = kr[None, None, :]
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0

    # phi1_k = i (k . Psi1_k) / k^2      [Psi1 = -grad phi1]
    phi = 1j * KX * rfftn(P[0], workers=workers)
    phi += 1j * KY * rfftn(P[1], workers=workers)
    phi += 1j * KZ * rfftn(P[2], workers=workers)
    phi /= K2
    phi[0, 0, 0] = 0.0
    if Rs > 0:
        phi *= np.exp(-0.5 * K2 * Rs ** 2)         # truncated LPT: filter the potential (both orders follow)

    K = (KX, KY, KZ)

    def dd(i, j):
        return irfftn(-K[i] * K[j] * phi, s=(n, n, n), workers=workers)

    d00, d11, d22 = dd(0, 0), dd(1, 1), dd(2, 2)
    S = d00 * d11 + d00 * d22 + d11 * d22
    del d00, d11, d22
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        t = dd(i, j)
        S -= t * t
        del t

    Sk = rfftn(S, workers=workers); del S
    Sk /= -K2                                       # phi2_k = -S_k / k^2
    Sk[0, 0, 0] = 0.0

    fac = -(3.0 / 7.0) * cosmo.Om ** (-1.0 / 143.0)  # Psi2 = D2 grad phi2, D2 = -(3/7) D1^2 Om^(-1/143)
    P2 = np.empty_like(P)
    for j in range(3):
        P2[j] = fac * irfftn(1j * K[j] * Sk, s=(n, n, n), workers=workers)
    del Sk

    P1s = np.empty_like(P)                           # first order consistent with the (truncated) phi1
    for j in range(3):
        P1s[j] = irfftn(-1j * K[j] * phi, s=(n, n, n), workers=workers)
    del phi

    r1 = float(np.sqrt(np.mean(np.sum(P1s ** 2, axis=0))))
    r2 = float(np.sqrt(np.mean(np.sum(P2 ** 2, axis=0))))
    return P1s, P2, r1, r2


def linear_density_from_psi1(psi1, box, workers=-1):
    """Linear overdensity delta_lin = -div Psi1 on the n^3 grid, at the same (z=0, D=1) amplitude as Psi1.

    Since Psi1 = -grad phi1 and lap phi1 = delta1, the divergence of the first-order displacement is
    exactly minus the linear density contrast.  This is the code-agnostic 'phase' object an IC generator
    (2LPTic / L-PICOLA / FastPM / N-GenIC) needs to reproduce this realisation: the field carries the
    exact Fourier phases, and the generator applies its own growth D(z_init) and transfer normalisation.

    Returns (delta[n,n,n] float32, rms).  <delta> ~ 0 by construction.
    """
    P = np.asarray(psi1, np.float64)
    n = P.shape[1]
    kf = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kr = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    KX = kf[:, None, None]; KY = kf[None, :, None]; KZ = kr[None, None, :]
    # delta_k = i k . Psi1_k   (real-space divergence; delta = -div(Psi1) with Psi1 = x - q => sign below)
    dk = 1j * KX * rfftn(P[0], workers=workers)
    dk += 1j * KY * rfftn(P[1], workers=workers)
    dk += 1j * KZ * rfftn(P[2], workers=workers)
    delta = -irfftn(dk, s=(n, n, n), workers=workers)     # delta_lin = -div Psi1
    return delta.astype(np.float32), float(delta.std())


def _lagrangian_grid(n, box):
    g = (np.arange(n) + 0.5) * (box / n)
    return np.stack(np.meshgrid(g, g, g, indexing='ij'), axis=0)


def lpt_positions(psi1s, psi2, box):
    """2LPT Eulerian positions q + Psi1_s + Psi2 (grid-order, wrapped).  Returns (n^3, 3)."""
    P1 = np.asarray(psi1s); P2 = np.asarray(psi2); n = P1.shape[1]
    q = _lagrangian_grid(n, box)
    return ((q + P1 + P2).reshape(3, -1).T % box)


def lpt_velocity_field(psi1s, psi2, box, ngrid=128, smooth=1.0, f1=None, f2=None, H0=100.0, z=0.0):
    """2LPT peculiar-velocity field v = H0 (f1 Psi1_s + f2 Psi2), deposited (momentum+mass CIC,
    Gaussian-smoothed) onto an ngrid^3 Eulerian cube.  Returns a dict with the same schema as
    gtweb.velocity.linear_velocity_field (so gtweb.sample_velocity reads it), tagged order='2lpt'.
    """
    P1 = np.asarray(psi1s, np.float64); P2 = np.asarray(psi2, np.float64)
    n = P1.shape[1]; box = float(box)
    if f1 is None:
        f1 = cosmo.growth_rate_f1()
    if f2 is None:
        f2 = cosmo.growth_rate_f2()

    q = _lagrangian_grid(n, box)
    pos = ((q + P1 + P2).reshape(3, -1).T % box)              # 2LPT particle positions
    vsite = (H0 * (f1 * P1 + f2 * P2)).reshape(3, -1).T       # per-site velocity (km/s)
    del P1, P2, q

    cell = box / ngrid
    e = np.linspace(0.0, box, ngrid + 1)
    mass, _ = np.histogramdd(pos, bins=(e, e, e))             # counts per Eulerian cell
    out = {}
    for j, name in enumerate(('vx', 'vy', 'vz')):
        mom, _ = np.histogramdd(pos, bins=(e, e, e), weights=vsite[:, j])
        mom = gaussian_filter(mom, smooth / cell, mode='wrap')
        m = gaussian_filter(mass, smooth / cell, mode='wrap')
        out[name] = (mom / np.maximum(m, 1e-12)).astype(np.float32)

    out.update(ngrid=int(ngrid), box_size=box, smooth=float(smooth),
               f=float(f1), f2=float(f2), H0=float(H0), z=float(z),
               units='km/s', order='2lpt')
    return out

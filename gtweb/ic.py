"""
Initial conditions for a mock web: a Gaussian random field and its Zel'dovich displacement.

The mock in the paper was grown from the initial conditions of a particular N-body run.  To make
your own mock you only need a first-order (Zel'dovich) displacement field Psi1 on a Lagrangian
grid at z=0 linear amplitude.  `gaussian_psi1` draws one from the linear LCDM power spectrum of
gtweb.cosmology (Eisenstein & Hu 1998 no-wiggle, sigma8-normalised); `za_positions` turns it into
the growth-scaled particle load the void search uses.

Conventions (shared with gtweb.lpt):
    q        = Lagrangian grid at cell centres, (i + 1/2) * box / n
    delta    = -div Psi1                      (linear density contrast at z=0)
    Psi1(k)  = i k / k^2 * delta(k)
For numpy's unnormalised DFT, delta(k) = W(k) * sqrt(P(k) n^3 / V), with W the DFT of unit-variance
white noise, gives <|delta(k)|^2> = n^6 P(k) / V, i.e. a field whose power spectrum is P(k).

Bring your own initial conditions instead: any Psi1 array of shape (3, n, n, n) in Mpc/h works.
"""
import numpy as np
from scipy.fft import rfftn, irfftn

from . import cosmology as cosmo


def _kgrid(n, box):
    kf = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kr = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    return kf[:, None, None], kf[None, :, None], kr[None, None, :]


def gaussian_psi1(box=100.0, ngrid=128, seed=0, pk=None, workers=-1):
    """Draw a Gaussian random field at z=0 linear amplitude and return its Zel'dovich displacement.

    Parameters
    ----------
    box   : side of the periodic box [Mpc/h]
    ngrid : Lagrangian grid per side (the mock has ngrid^3 particles)
    seed  : random seed (sets the phases)
    pk    : callable P(k) [(Mpc/h)^3], k in h/Mpc; defaults to gtweb.cosmology.linear_Pk

    Returns a dict in the format of gtweb.lpt.load_psi1:
        psi1 (3, n, n, n) float32 [Mpc/h], box_size, ngrid, seed, rms_psi1
    """
    if pk is None:
        pk = cosmo.linear_Pk
    n = int(ngrid); box = float(box)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n, n))
    wk = rfftn(white, workers=workers)
    KX, KY, KZ = _kgrid(n, box)
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    k = np.sqrt(K2)
    amp = np.zeros_like(k)
    nz = k > 0
    amp[nz] = np.sqrt(pk(k[nz]) * n ** 3 / box ** 3)
    dk = wk * amp
    dk[0, 0, 0] = 0.0
    K2[0, 0, 0] = 1.0
    psi = np.empty((3, n, n, n), np.float32)
    for i, Ki in enumerate((KX, KY, KZ)):
        psi[i] = irfftn(1j * Ki / K2 * dk, s=(n, n, n), workers=workers)
    rms = float(np.sqrt((psi.astype(np.float64) ** 2).sum(axis=0).mean()))
    return dict(psi1=psi, box_size=box, ngrid=n, seed=int(seed), rms_psi1=rms)


def za_positions(psi1, box):
    """Zel'dovich particle positions q + Psi1 (grid order, wrapped into the box).  Returns (n^3, 3)."""
    P = np.asarray(psi1)
    n = P.shape[1]
    g = (np.arange(n) + 0.5) * (box / n)
    q = np.stack(np.meshgrid(g, g, g, indexing='ij'), axis=0)
    return ((q + P).reshape(3, -1).T % box)


def measure_pk(delta, box, nbins=30):
    """Spherically averaged power spectrum of a density-contrast grid -> (k [h/Mpc], P(k) [(Mpc/h)^3])."""
    n = delta.shape[0]
    dk = rfftn(delta)
    KX, KY, KZ = _kgrid(n, box)
    k = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).ravel()
    p = (np.abs(dk) ** 2).ravel() * box ** 3 / n ** 6
    # the rfft half-plane double-counts nothing but drops the conjugate half; weights cancel in the mean
    kmin, kmax = 2 * np.pi / box, np.pi * n / box
    edges = np.geomspace(kmin, kmax, nbins + 1)
    idx = np.digitize(k, edges)
    kc, pc = [], []
    for b in range(1, nbins + 1):
        m = idx == b
        if m.any():
            kc.append(k[m].mean()); pc.append(p[m].mean())
    return np.array(kc), np.array(pc)

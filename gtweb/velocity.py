"""
The ground-truth velocity field.

No velocity is stored per particle (the mock is built by displacing initial-condition
particles onto the analytic web, but the initial-condition Lagrangian identities are not
kept, so a per-particle displacement velocity is not recoverable).  Instead we provide the
standard cosmological velocity field of the growing mode, derived from the density itself:

    linear (Zel'dovich) peculiar velocity     v(k) = f * a * H0 * i k / k^2 * delta(k)

which follows from linear continuity, div v = - a H f delta.  This needs NO particle
identities, works directly on any mock particle field, and physically converges onto
the cosmic web -- it is the large-scale accretion flow that funnels matter along the
filaments and into the nodes.

Units: km/s (positions in Mpc/h, H0 = 100 h km/s/Mpc, a = 1 at z = 0), on the periodic box.
This is deliberately the LINEAR field: it captures the bulk accretion flow and omits the
nonlinear virial motions inside collapsed halos.

Everything here is pure numpy/scipy with no hardcoded paths.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import cosmology as cosmo


def linear_velocity_field(pos, box, ngrid=128, smooth=1.0, f=None, H0=100.0, z=0.0):
    """Linear-theory peculiar-velocity field of a particle distribution.

    Deposits `pos` on an `ngrid`^3 mesh, smooths by `smooth` (Mpc/h), forms the overdensity
    delta, and solves v(k) = f H0 i k / k^2 delta(k) in Fourier space (growing mode, a=1).

    Parameters
    ----------
    pos    : (N,3) particle positions (Mpc/h).
    box    : box size (Mpc/h), periodic.
    ngrid  : velocity-cube resolution per side.
    smooth : Gaussian smoothing of the density before differentiation (Mpc/h).
    f      : linear growth rate; defaults to Omega_m^0.55 (z=0).
    H0     : Hubble constant in km/s per (Mpc/h)  (=100, so v comes out in km/s).
    z      : redshift label stored in the returned dict (the solve assumes a=1 / z=0).

    Returns
    -------
    dict with vx, vy, vz (ngrid^3 float32, km/s), ngrid, box_size, smooth, f, H0, z, units.
    """
    pos = np.asarray(pos, float)
    box = float(box)
    if f is None:
        f = float(cosmo.Om) ** 0.55
    cell = box / ngrid

    # overdensity on the mesh
    e = np.linspace(0.0, box, ngrid + 1)
    rho, _ = np.histogramdd(pos % box, bins=(e, e, e))
    if smooth > 0:
        rho = gaussian_filter(rho, smooth / cell, mode='wrap')
    delta = rho / rho.mean() - 1.0

    # v(k) = f H0 i k / k^2 delta(k)
    dk = np.fft.rfftn(delta)
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=cell)
    kr = 2.0 * np.pi * np.fft.rfftfreq(ngrid, d=cell)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing='ij')
    K2 = KX * KX + KY * KY + KZ * KZ
    K2[0, 0, 0] = 1.0                                   # avoid /0; the k=0 (bulk) mode is unconstrained
    pref = f * H0
    axes = (0, 1, 2)
    vx = np.fft.irfftn(pref * 1j * KX / K2 * dk, s=delta.shape, axes=axes)
    vy = np.fft.irfftn(pref * 1j * KY / K2 * dk, s=delta.shape, axes=axes)
    vz = np.fft.irfftn(pref * 1j * KZ / K2 * dk, s=delta.shape, axes=axes)

    return dict(vx=vx.astype(np.float32), vy=vy.astype(np.float32), vz=vz.astype(np.float32),
                ngrid=int(ngrid), box_size=box, smooth=float(smooth),
                f=float(f), H0=float(H0), z=float(z), units='km/s')


def sample_velocity(field, points, box=None):
    """Trilinearly interpolate a velocity cube (from linear_velocity_field / load_velocity_field)
    at arbitrary `points` (M,3).  Returns (M,3) velocities in km/s.

    Use it to attach velocities to any positions -- a particle subsample, or your finder's
    node/skeleton points.  Periodic (mode='grid-wrap')."""
    if box is None:
        box = float(field['box_size'])
    ng = int(field['ngrid'])
    cell = box / ng
    # cube node m is the density cell centred at (m+0.5)*cell, so index = x/cell - 0.5
    coords = (np.asarray(points, float) % box).T / cell - 0.5
    v = [map_coordinates(np.asarray(field[c]), coords, order=1, mode='grid-wrap')
         for c in ('vx', 'vy', 'vz')]
    return np.stack(v, axis=1)


def velocity_field_from_ic(ic_path, out_path=None, ngrid=128, **kw):
    """Convenience: build a linear velocity field straight from a particle .npz (IC seed or a
    mock field with `positions` + `box_size`).  If `out_path` is given, also writes it there
    via io.write_velocity_field.  Returns the field dict."""
    d = np.load(ic_path, allow_pickle=True)
    pos = d['positions'].astype(np.float64)
    box = float(d['box_size']) if 'box_size' in d.files else 100.0
    field = linear_velocity_field(pos, box, ngrid=ngrid, **kw)
    if out_path is not None:
        from .io import write_velocity_field
        write_velocity_field(out_path, field)
    return field

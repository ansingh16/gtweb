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
`psi1_from_gadget` builds it from GADGET-1 initial-condition files (e.g. the paper's, written by
Ginnungagap), and `read_gadget` reads one such file.
"""
import re
import struct
from pathlib import Path

import numpy as np
from scipy.fft import rfftn, irfftn
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter

from . import cosmology as cosmo
from .periodic import wrap


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
    return wrap((q + P).reshape(3, -1).T, box)


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


# ----------------------------------------------------------------------------- GADGET-1 initial conditions
_HEADER_BYTES = 256
# The paper's 384^3 initial conditions at z = 120 leave 0.18 per cent of the cells empty; far more means the
# displacements are not small compared with a cell (or particles are missing), and the binning is not valid.
MAX_EMPTY_FRACTION = 0.05


def _block(f, path, what):
    """One Fortran-style block <int32 n><n bytes><int32 n>; None at the end of the file."""
    raw = f.read(4)
    if len(raw) == 0:
        return None
    if len(raw) < 4:
        raise ValueError(f"{path}: truncated {what} block marker")
    n = struct.unpack('<I', raw)[0]
    data = f.read(n)
    tail = f.read(4)
    if len(data) != n or len(tail) < 4:
        raise ValueError(f"{path}: {what} block truncated")
    if struct.unpack('<I', tail)[0] != n:
        raise ValueError(f"{path}: {what} block size markers differ")
    return data


def read_gadget(path):
    """Read one GADGET-1 (format 1, little-endian) file: header and particle positions.

    Returns (pos, header): pos is a float32 (N, 3) array in the file's length unit, N = sum(npart);
    header is a dict with npart, massarr, a, redshift, npart_total, num_files, box (BoxSize, file
    units), omega_m (Omega0), omega_l, h.  Velocities and IDs are not read.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        hdr = _block(f, path, 'header')
        if hdr is None or len(hdr) != _HEADER_BYTES:
            raise ValueError(f"{path}: header block is not {_HEADER_BYTES} bytes")
        npart = struct.unpack('<6i', hdr[0:24])
        massarr = struct.unpack('<6d', hdr[24:72])
        a, redshift = struct.unpack('<2d', hdr[72:88])
        total_lo = struct.unpack('<6I', hdr[96:120])
        num_files = struct.unpack('<i', hdr[124:128])[0]
        box, om, ol, h = struct.unpack('<4d', hdr[128:160])
        # bytes 160 and 164 are flag_stellarage and flag_metals; 168-192 the high words of the totals (> 2^32)
        total_hi = struct.unpack('<6I', hdr[168:192])
        n_tot = sum(npart)
        posb = _block(f, path, 'position')
        if posb is None or len(posb) != 12 * n_tot:
            raise ValueError(f"{path}: position block holds {0 if posb is None else len(posb)} bytes, "
                             f"expected 12 * {n_tot}")
    pos = np.frombuffer(posb, dtype='<f4').reshape(n_tot, 3).astype(np.float32)
    header = dict(npart=list(npart), massarr=list(massarr), a=a, redshift=redshift,
                  npart_total=[int(lo) + (int(hi) << 32) for lo, hi in zip(total_lo, total_hi)],
                  num_files=num_files, box=box, omega_m=om, omega_l=ol, h=h)
    return pos, header


def _ic_files(files):
    """A directory -> its files <stem>.<integer>, in integer order (one stem only); else the given paths."""
    if isinstance(files, (str, Path)):
        d = Path(files)
        if not d.is_dir():
            return [d]
        groups = {}
        for p in d.iterdir():
            m = re.fullmatch(r'(.+)\.(\d+)', p.name)
            if m and p.is_file():
                groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
        if not groups:
            raise ValueError(f"{d}: no files named <stem>.<integer>")
        if len(groups) > 1:
            raise ValueError(f"{d}: several file sets ({', '.join(sorted(groups))}); pass the files as a list")
        return [p for _, p in sorted(next(iter(groups.values())))]
    return [Path(p) for p in files]


def _growth_unnorm(a, om):
    """Unnormalised flat-LCDM growth D(a) ~ H(a) int_0^a da' / (a' H(a'))^3 for matter density om."""
    ol = 1.0 - om

    def hnorm(x):
        return np.sqrt(om / x ** 3 + ol)
    integ, _ = quad(lambda x: 1.0 / (x * hnorm(x)) ** 3, 1e-8, a)
    return hnorm(a) * integ


def psi1_from_gadget(files, out=None, omega_m=None, length_unit=1e-3, verbose=False):
    """Zel'dovich displacement Psi1 at z=0 linear amplitude from GADGET-1 initial-condition files.

    files       : directory holding <stem>.0 ... <stem>.N-1 (read in integer order), or a list of files
    out         : optional .npz path; gtweb.lpt.load_psi1 reads it and generate_from_ic takes it as psi1_path
    omega_m     : matter density for the growth factor; default the header's Omega0
    length_unit : Mpc/h per file length unit (1e-3 for files in kpc/h, 1.0 for Mpc/h)

    Each particle is assigned to the Lagrangian cell (centre (i + 1/2) * box / n) that contains its
    position, so the initial displacements must be smaller than half a cell (true at the high starting
    redshift of an N-body run).  A cell holds the mean displacement of its particles; the few cells that
    receive none take the mean of their filled neighbours, weighted by a periodic Gaussian of one cell.
    The result is scaled by D(1) / D(a_ic).

    Returns a dict: psi1 (3, n, n, n) float32 [Mpc/h], box_size, ngrid, growth, rms_psi1, omega_m, a_ic,
    n_empty (cells filled from their neighbours) and, if `out` was given, out (the file written; '.npz' is
    appended when missing).

    Raises ValueError if the files are inconsistent (scale factor, box, particle totals, num_files), a file
    is given twice, particles of another type are present, the total is not a cube, the box comes out
    below 1 Mpc/h (wrong length_unit), or more than MAX_EMPTY_FRACTION of the cells receive no particle.
    """
    paths = _ic_files(files)
    if not paths:
        raise ValueError("no initial-condition files given")
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("the same file is given more than once")
    per_unit = 1.0 / length_unit                    # file units per Mpc/h (divide by it, as the paper did)
    chunks, h0 = [], None
    for p in paths:
        pos, h = read_gadget(p)
        if h0 is None:
            h0 = h
        else:
            for key in ('a', 'box', 'npart_total', 'num_files'):
                if h[key] != h0[key]:
                    raise ValueError(f"{p}: header {key} = {h[key]} differs from {paths[0]} ({h0[key]})")
        if any(h['npart'][k] for k in range(6) if k != 1):
            raise ValueError(f"{p}: particles of a type other than 1 (npart = {h['npart']}); "
                             "only single-species dark-matter initial conditions are supported")
        chunks.append(pos)
    if h0['num_files'] != len(paths):
        raise ValueError(f"header num_files = {h0['num_files']}, but {len(paths)} files were given")
    if any(h0['npart_total'][k] for k in range(6) if k != 1):
        raise ValueError(f"npart_total = {h0['npart_total']}: particles of a type other than 1")
    n_part = sum(len(c) for c in chunks)
    if n_part != h0['npart_total'][1]:
        raise ValueError(f"the files hold {n_part} particles, header npart_total = {h0['npart_total'][1]}")
    n = int(round(n_part ** (1.0 / 3.0)))
    n = next((m for m in (n - 1, n, n + 1) if m > 0 and m ** 3 == n_part), None)
    if n is None:
        raise ValueError(f"{n_part} particles is not a cube n^3: not a Lagrangian grid")

    box = h0['box'] / per_unit
    if box < 1.0:
        raise ValueError(f"box of {box:g} Mpc/h: the length unit is probably wrong (length_unit = {length_unit:g} "
                         "Mpc/h per file unit; use 1.0 for files in Mpc/h)")
    cell = box / n
    om = float(h0['omega_m'] if omega_m is None else omega_m)
    a_ic = float(h0['a'])

    x = wrap(np.concatenate(chunks).astype(np.float64) / per_unit, box)
    del chunks
    ix = np.minimum((x / cell).astype(np.int64), n - 1)
    psi = x - (ix + 0.5) * cell
    psi -= box * np.round(psi / box)                # minimum image
    del x
    flat = (ix[:, 0] * n + ix[:, 1]) * n + ix[:, 2]
    del ix

    cnt = np.bincount(flat, minlength=n ** 3).astype(np.float64)
    if cnt.sum() != n_part:
        raise RuntimeError("particles lost in the Lagrangian binning")
    P = np.empty((3, n, n, n))
    for j in range(3):
        P[j] = np.bincount(flat, weights=psi[:, j], minlength=n ** 3).reshape(n, n, n)
    del psi, flat
    cnt = cnt.reshape(n, n, n)
    empty = cnt == 0
    if empty.mean() > MAX_EMPTY_FRACTION:
        raise ValueError(f"{empty.mean():.1%} of the Lagrangian cells received no particle (at most "
                         f"{MAX_EMPTY_FRACTION:.0%} allowed): displacements larger than half a cell, or files missing")
    den = np.where(empty, 1.0, cnt)
    for j in range(3):
        P[j] /= den
    if empty.any():
        num = gaussian_filter(np.where(empty, 0.0, 1.0), 1.0, mode='wrap')
        for j in range(3):
            fill = gaussian_filter(np.where(empty, 0.0, P[j]), 1.0, mode='wrap') / np.maximum(num, 1e-12)
            P[j][empty] = fill[empty]

    growth = _growth_unnorm(1.0, om) / _growth_unnorm(a_ic, om)
    P *= growth
    psi1 = P.astype(np.float32)
    del P
    rms = float(np.sqrt((psi1.astype(np.float64) ** 2).sum(axis=0).mean()))
    res = dict(psi1=psi1, box_size=float(box), ngrid=int(n), growth=float(growth), rms_psi1=rms,
               omega_m=om, a_ic=a_ic, n_empty=int(empty.sum()))
    if verbose:
        print(f"read {len(paths)} files, {n_part} particles = {n}^3, box {box:g} Mpc/h, z_ic = {1 / a_ic - 1:.4g}, "
              f"growth D(1)/D(a_ic) = {growth:.6g} (Omega_m = {om:g}); {res['n_empty']} empty cells "
              f"({res['n_empty'] / n ** 3:.3%}), max occupancy {int(cnt.max())}; rms|Psi1| = {rms:.4g} Mpc/h",
              flush=True)
    if out is not None:
        out = Path(out)
        if out.suffix != '.npz':                    # np.savez would append it silently
            out = out.with_name(out.name + '.npz')
        res['out'] = str(out)
        np.savez(out, psi1=psi1, box_size=np.float64(box), ngrid=np.int32(n), growth=np.float64(growth),
                 rms_psi1=np.float64(rms), omega_m=np.float64(om), a_ic=np.float64(a_ic))
    return res

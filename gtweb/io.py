"""
Readers / writers for the ground-truth benchmark, plus a DisPerSE skeleton reader
so DisPerSE users can score their output directly.

All coordinates are in Mpc/h on a periodic box of side `box_size`.
Nothing here has a hardcoded path.
"""
import json
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------- loaders
def load_ground_truth(path):
    """Load ground_truth_spines.npz -> dict with:
        spines            : list of (6,3) polylines (the true filament network)
        box_size          : float (from the file, or from the sibling nodes/web files)
        node_positions    : (Nnode,3)
        filament_lengths, filament_target_contrast (delta_vol), filament_lambda, ... (per-filament arrays)
    """
    d = np.load(path, allow_pickle=True)
    K = int(d['n_filaments'])
    out = {k: d[k] for k in d.files if not k.startswith('spine_')}
    out['spines'] = [d[f'spine_{i}'] for i in range(K)]
    out['n_filaments'] = K
    if 'box_size' not in out:                       # files written by older versions: use the siblings
        here = Path(path).parent
        for sib in ("nodes_catalogue.npz", "web_geometry.npz"):
            if (here / sib).exists():
                s = np.load(here / sib, allow_pickle=True)
                if 'box_size' in s.files:
                    out['box_size'] = np.float64(s['box_size']); break
        else:
            raise ValueError(f"{path} has no box_size and no sibling file gives one")
    return out


def load_nodes(path):
    """Load nodes_catalogue.npz -> dict (positions, radii, r200, peaks, box_size)."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_particles(path):
    """Load a particle field npz -> dict with positions (N,3), box_size, and any metadata present."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['positions'] = out['positions'].astype(np.float32)
    return out


def load_halos(path):
    """Load halos_catalogue.npz -> dict (positions (H,3), mass (H,), n_particles (H,), box_size,
    m_particle, min_npart).  These are the mock's Sheth-Tormen/NFW halos (a cosmic-web tracer)."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------- writers
def write_halos(path, cat):
    """Write a halo catalogue dict (from particles.halo_catalogue) to `path`."""
    np.savez(path, **cat)
def write_ground_truth(out_dir, gt, nodes_cat, walls_cat, web_geom, metadata=None):
    """Write the four GT npz files (+ optional metadata.json) into out_dir."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gt = dict(gt)
    if 'box_size' not in gt:
        gt['box_size'] = np.float64(nodes_cat['box_size'] if 'box_size' in nodes_cat else web_geom['box_size'])
    np.savez(out_dir / "ground_truth_spines.npz", **gt)
    np.savez(out_dir / "nodes_catalogue.npz", **nodes_cat)
    np.savez(out_dir / "walls_catalogue.npz", **walls_cat)
    np.savez(out_dir / "web_geometry.npz", **web_geom)
    if metadata is not None:
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def write_velocity_field(path, field):
    """Write a velocity-field dict (from velocity.linear_velocity_field or lpt.lpt_velocity_field)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if 'order' in field:
        extra['order'] = np.array(field['order'])          # 'linear' or '2lpt'
    if 'f2' in field:
        extra['f2'] = np.float64(field['f2'])              # 2LPT second-order growth rate
    np.savez(path,
             vx=np.asarray(field['vx'], np.float32), vy=np.asarray(field['vy'], np.float32),
             vz=np.asarray(field['vz'], np.float32),
             ngrid=np.int32(field['ngrid']), box_size=np.float64(field['box_size']),
             smooth=np.float64(field['smooth']), f=np.float64(field['f']),
             H0=np.float64(field['H0']), z=np.float64(field['z']),
             units=np.array(field.get('units', 'km/s')), **extra)


def load_velocity_field(path):
    """Load velocity_field.npz -> dict with vx, vy, vz (ngrid^3, km/s) + ngrid, box_size, f, H0, z.

    Pair it with gtweb.sample_velocity(field, points) to read the flow at any positions."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['ngrid'] = int(out['ngrid'])
    out['box_size'] = float(out['box_size'])
    return out


def write_disperse_pos(positions, filepath, box):
    """Write particle positions as a DisPerSE ANDFIELD COORDS text file (delaunay_3D input)."""
    positions = np.asarray(positions)
    with open(filepath, 'w') as f:
        f.write("ANDFIELD COORDS\n")
        f.write(f"[3,{positions.shape[0]}]\n")
        np.savetxt(f, positions, fmt="%.6f")
        f.write(f"BBOX [0,0,0] [{box},{box},{box}]\n")


# --------------------------------------------------------------------------- DisPerSE skeleton reader
def read_disperse_skeleton(path):
    """Parse a DisPerSE ASCII skeleton (`.NDskl` written by `skelconv ... -to NDskl_ascii`, often named
    `*.a.NDskl`) and return its filaments as a list of (nsamp, 3) polylines, ready for gtweb.score.

    The ASCII format has a `[FILAMENTS]` block: a count line, then per filament a header
    `cp1 cp2 nsamp` followed by `nsamp` coordinate lines.  Works for 2D or 3D (uses the first 3
    columns; pads a z=0 for 2D)."""
    text = Path(path).read_text().splitlines()
    # locate the [FILAMENTS] section
    i = 0
    while i < len(text) and not text[i].strip().upper().startswith('[FILAMENTS]'):
        i += 1
    if i >= len(text):
        raise ValueError(f"no [FILAMENTS] section found in {path}")
    i += 1
    nfil = int(text[i].split()[0]); i += 1
    fils = []
    for _ in range(nfil):
        # skip blank lines
        while i < len(text) and not text[i].strip():
            i += 1
        hdr = text[i].split(); i += 1
        nsamp = int(hdr[2])
        pts = []
        got = 0
        while got < nsamp and i < len(text):
            toks = text[i].split(); i += 1
            if not toks:
                continue
            vals = [float(t) for t in toks]
            if len(vals) < 2:
                continue
            x, y = vals[0], vals[1]
            z = vals[2] if len(vals) >= 3 else 0.0
            pts.append((x, y, z)); got += 1
        if pts:
            fils.append(np.asarray(pts, dtype=float))
    return fils


def read_disperse_critical_points(path):
    """Parse the [CRITICAL POINTS] block of a DisPerSE ASCII skeleton.  Returns a dict with
    positions (N,3), type (N,) and value (N,).  In 3D, type 3 = density maximum, 0 = minimum,
    1 and 2 = saddles, 4 = bifurcation point (added by skelconv -breakdown)."""
    text = Path(path).read_text().splitlines()
    i = 0
    while i < len(text) and text[i].strip().upper() != '[CRITICAL POINTS]':
        i += 1
    if i >= len(text):
        raise ValueError(f"no [CRITICAL POINTS] section found in {path}")
    i += 1
    ncp = int(text[i].split()[0]); i += 1
    pos, typ, val = [], [], []
    for _ in range(ncp):
        toks = text[i].split(); i += 1
        typ.append(int(toks[0]))
        xyz = [float(t) for t in toks[1:-3]]
        pos.append((xyz + [0.0, 0.0, 0.0])[:3])
        val.append(float(toks[-3]))
        nfil = int(text[i].split()[0]); i += 1 + nfil
    return dict(positions=np.asarray(pos, float), type=np.asarray(typ, int), value=np.asarray(val, float))

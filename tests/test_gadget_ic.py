"""GADGET-1 initial conditions -> Zel'dovich Psi1 on the Lagrangian grid (gtweb.read_gadget, gtweb.psi1_from_gadget).

Written from the specification only (the implementation was not read).  Spec property numbers (P1..P11) refer to the
task specification of read_gadget / psi1_from_gadget.

Conventions under test: Lagrangian grid at cell centres q_i = (i + 1/2) * cell, cell = box / n, axis order (x, y, z) ->
array index [ix, iy, iz]; positions wrapped into [0, box); psi = minimum-image (x - q_cell); empty cells filled with the
periodic Gaussian-weighted (sigma = 1 cell) mean of filled cells; result multiplied by growth = D(1) / D(a_ic).

The tiny GADGET-1 files are written by `write_gadget` below (format 1, little-endian, Fortran record markers).
The real-data test reads its paths from the environment variables GTWEB_TEST_IC_DIR and GTWEB_TEST_PSI1.
"""
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter

import gtweb
from gtweb import lpt


# ---------------------------------------------------------------------------------------------- GADGET-1 writer
def _block(payload, lead=None, trail=None):
    """One Fortran record: <int32 nbytes><payload><int32 nbytes> (markers overridable to make corrupt files)."""
    nb = len(payload)
    lead = nb if lead is None else lead
    trail = nb if trail is None else trail
    return struct.pack("<i", lead) + payload + struct.pack("<i", trail)


def make_header(npart, *, a=1.0, box=10.0, npart_total=None, num_files=1, omega0=0.307, omega_l=None,
                h=0.6777, massarr=None, size=256):
    """256-byte GADGET-1 header with the byte offsets of the specification."""
    npart = list(npart)
    npart_total = list(npart) if npart_total is None else list(npart_total)
    massarr = [0.0] * 6 if massarr is None else list(massarr)
    omega_l = 1.0 - omega0 if omega_l is None else omega_l
    hdr = bytearray(size)
    struct.pack_into("<6i", hdr, 0, *npart)
    struct.pack_into("<6d", hdr, 24, *massarr)
    struct.pack_into("<d", hdr, 72, a)
    struct.pack_into("<d", hdr, 80, 1.0 / a - 1.0)
    struct.pack_into("<i", hdr, 88, 0)
    struct.pack_into("<i", hdr, 92, 0)
    struct.pack_into("<6I", hdr, 96, *npart_total)
    struct.pack_into("<i", hdr, 120, 0)
    struct.pack_into("<i", hdr, 124, num_files)
    struct.pack_into("<d", hdr, 128, box)
    struct.pack_into("<d", hdr, 136, omega0)
    struct.pack_into("<d", hdr, 144, omega_l)
    struct.pack_into("<d", hdr, 152, h)
    return bytes(hdr)


def write_gadget(path, pos, *, npart=None, velocities=True, ids="u4", id_offset=0, header_size=256,
                 pos_payload=None, bad_marker=False, **hdr_kw):
    """Write one GADGET-1 file holding `pos` (N, 3) as type-1 particles (unless `npart` is given).

    header_size / pos_payload / bad_marker build deliberately broken files for the error tests."""
    pos = np.ascontiguousarray(pos, dtype="<f4").reshape(-1, 3)
    N = len(pos)
    npart = [0, N, 0, 0, 0, 0] if npart is None else list(npart)
    hdr = make_header(npart, size=header_size, **hdr_kw)
    out = _block(hdr)
    payload = pos.tobytes() if pos_payload is None else pos_payload
    if bad_marker:
        out += _block(payload, trail=len(payload) + 4)
    else:
        out += _block(payload)
    if velocities:
        out += _block(np.zeros((N, 3), "<f4").tobytes())
        if ids is not None:
            out += _block((np.arange(N) + 1 + id_offset).astype("<" + ids).tobytes())
    Path(path).write_bytes(out)
    return path


def write_split(dirpath, pos, nfiles, *, stem="GADGET", shuffle_seed=None, **hdr_kw):
    """Split `pos` into nfiles uneven chunks GADGET.0 .. GADGET.<nfiles-1>; return the paths in numeric order."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(pos, np.float32).reshape(-1, 3)
    N = len(pos)
    if nfiles == 1:
        cuts = [0, N]
    else:   # uneven chunk sizes, every chunk non-empty
        w = np.arange(1, nfiles + 1, dtype=float)
        cuts = np.concatenate([[0], np.round(np.cumsum(w) / w.sum() * N).astype(int)])
        cuts[-1] = N
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    paths = []
    for i in range(nfiles):
        chunk = pos[cuts[i]:cuts[i + 1]]
        if rng is not None:
            chunk = chunk[rng.permutation(len(chunk))]
        p = dirpath / f"{stem}.{i}"
        write_gadget(p, chunk, npart_total=[0, N, 0, 0, 0, 0], num_files=nfiles, id_offset=cuts[i], **hdr_kw)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------------------------- lattice helpers
def lattice_q(n, box):
    """Cell-centre Lagrangian grid, shape (3, n, n, n), q[d][ix, iy, iz] = (i_d + 1/2) * cell."""
    cell = box / n
    g = (np.arange(n) + 0.5) * cell
    return np.array(np.meshgrid(g, g, g, indexing="ij"))


def particles(q_plus_psi):
    """(3, n, n, n) positions -> (n^3, 3) particle array (grid order)."""
    return q_plus_psi.reshape(3, -1).T.copy()


def growth_ref(a, om):
    """Reference D(1)/D(a), D(a) = E(a) int_0^a da' / (a' E(a'))^3, E^2 = Om/a^3 + 1 - Om (flat LCDM)."""
    E = lambda x: np.sqrt(om / x ** 3 + (1.0 - om))
    D = lambda x: E(x) * quad(lambda y: 1.0 / (y * E(y)) ** 3, 0.0, x, epsabs=0, epsrel=1e-12, limit=400)[0]
    return D(1.0) / D(a)


N8, BOX = 8, 10.0          # default tiny configuration: 8^3 particles in a 10 Mpc/h box (cell 1.25 Mpc/h)
CELL = BOX / N8
KPC = 1000.0               # file units kpc/h: positions and BoxSize are written x 1000, length_unit = 1e-3


def run_kpc(tmp_path, pos_mpc, *, name="ic", nfiles=1, a=1.0, omega0=0.307, box=BOX, **kw):
    """Write Mpc/h positions as kpc/h GADGET files and convert them with the default length_unit (1e-3)."""
    paths = write_split(tmp_path / name, np.asarray(pos_mpc) * KPC, nfiles, a=a, box=box * KPC, omega0=omega0)
    return gtweb.psi1_from_gadget([str(p) for p in paths], **kw)


# ============================================================================================ read_gadget
def test_read_gadget_header_and_positions(tmp_path):
    """read_gadget returns the positions (float32, N x 3, file units) and every header field at its byte offset.

    Hand values: npart = [0, 27, 0, 0, 0, 0]; a = 0.25 -> redshift 3; BoxSize 5000; Omega0 0.3, OmegaLambda 0.7,
    h 0.7; num_files 1; massarr[1] = 1.5."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 5000.0, (27, 3)).astype(np.float32)
    p = write_gadget(tmp_path / "GADGET.0", pos, a=0.25, box=5000.0, omega0=0.3, omega_l=0.7, h=0.7,
                     massarr=[0, 1.5, 0, 0, 0, 0])
    got, hdr = gtweb.read_gadget(str(p))
    assert got.dtype == np.float32
    assert got.shape == (27, 3)
    assert np.array_equal(got, pos)
    assert list(hdr["npart"]) == [0, 27, 0, 0, 0, 0]
    assert list(hdr["npart_total"]) == [0, 27, 0, 0, 0, 0]
    assert np.allclose(hdr["massarr"], [0, 1.5, 0, 0, 0, 0])
    assert hdr["a"] == 0.25
    assert hdr["redshift"] == pytest.approx(3.0)
    assert hdr["num_files"] == 1
    assert hdr["box"] == 5000.0
    assert hdr["omega_m"] == 0.3
    assert hdr["omega_l"] == 0.7
    assert hdr["h"] == 0.7


@pytest.mark.parametrize("velocities,ids", [(False, None), (True, None), (True, "u4"), (True, "u8")])
def test_read_gadget_optional_blocks(tmp_path, velocities, ids):
    """'Only header and positions are needed; velocities and IDs may be absent'; IDs may be uint32 or uint64."""
    pos = np.arange(30, dtype=np.float32).reshape(10, 3)
    p = write_gadget(tmp_path / "f.0", pos, velocities=velocities, ids=ids)
    got, hdr = gtweb.read_gadget(str(p))
    assert np.array_equal(got, pos)
    assert sum(hdr["npart"]) == 10


def test_read_gadget_axis_order(tmp_path):
    """Columns are (x, y, z) as written: a particle at (1, 2, 3) is read back as row [1, 2, 3] (no z,y,x swap)."""
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]], np.float32)
    got, _ = gtweb.read_gadget(str(write_gadget(tmp_path / "g.0", pos)))
    assert np.array_equal(got, pos)


def test_read_gadget_bad_marker(tmp_path):
    """'Raises ValueError: a block's leading and trailing size markers differ.'"""
    pos = np.zeros((8, 3), np.float32)
    p = write_gadget(tmp_path / "bad.0", pos, bad_marker=True)
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(p))


def test_read_gadget_header_not_256(tmp_path):
    """'Raises ValueError: the header block is not 256 bytes.'"""
    pos = np.zeros((8, 3), np.float32)
    p = write_gadget(tmp_path / "bad.0", pos, header_size=252)
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(p))


def test_read_gadget_position_block_wrong_size(tmp_path):
    """'Raises ValueError: the position block size != 12 * sum(npart).' (npart says 8, block holds 7 particles)."""
    pos = np.zeros((8, 3), np.float32)
    p = write_gadget(tmp_path / "bad.0", pos, pos_payload=np.zeros((7, 3), "<f4").tobytes(), velocities=False)
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(p))


# ============================================================================================ psi1_from_gadget
def test_exported_from_top_level():
    """Both functions are 'also exported from the top-level package as gtweb.read_gadget and gtweb.psi1_from_gadget'."""
    from gtweb import ic
    assert gtweb.read_gadget is ic.read_gadget
    assert gtweb.psi1_from_gadget is ic.psi1_from_gadget


def test_unperturbed_lattice_gives_zero(tmp_path):
    """P1: particles exactly at the cell centres -> psi1 == 0 everywhere, n_empty == 0; shapes and metadata.

    Hand computation: x = q_cell, so psi = x - q_cell = 0 (up to float32 rounding of kpc/h positions, ~1e-6 Mpc/h)."""
    q = lattice_q(N8, BOX)
    d = run_kpc(tmp_path, particles(q))
    assert d["psi1"].shape == (3, N8, N8, N8)
    assert d["psi1"].dtype == np.float32
    assert d["ngrid"] == N8
    assert d["box_size"] == pytest.approx(BOX)
    assert d["n_empty"] == 0
    assert np.allclose(d["psi1"], 0.0, atol=1e-5)
    assert d["growth"] == pytest.approx(1.0, rel=1e-9)   # a_ic = 1
    assert d["a_ic"] == 1.0


@pytest.mark.parametrize("s", [(0.3, -0.2, 0.1), (-0.45, 0.45, -0.05)])
def test_uniform_shift_a1(tmp_path, s):
    """P2: all particles at q + s with |s| < cell/2 and a_ic = 1 -> psi1 == s in every cell, per component.

    s is in units of the cell (1.25 Mpc/h); asymmetric components so an axis swap shows."""
    s = np.array(s) * CELL
    q = lattice_q(N8, BOX)
    d = run_kpc(tmp_path, particles(q + s[:, None, None, None]))
    for k in range(3):
        assert np.allclose(d["psi1"][k], s[k], atol=1e-5), k
    assert d["n_empty"] == 0


def test_uniform_shift_eds_growth(tmp_path):
    """P2: a_ic = 0.25, Om = 1 (EdS, D = a) -> growth = 1/a_ic = 4 and psi1 == s / a_ic = 4 s."""
    a = 0.25
    s = np.array([0.3, -0.2, 0.1]) * CELL
    q = lattice_q(N8, BOX)
    d = run_kpc(tmp_path, particles(q + s[:, None, None, None]), a=a, omega0=1.0)
    assert d["growth"] == pytest.approx(1.0 / a, rel=1e-6)
    assert d["omega_m"] == 1.0
    assert d["a_ic"] == pytest.approx(a)
    for k in range(3):
        assert np.allclose(d["psi1"][k], s[k] / a, atol=1e-4), k


def test_smooth_field_axis_order(tmp_path):
    """P3: psi_x = A sin(2 pi q_y / box), psi_z = B cos(2 pi q_x / box), psi_y = 0 with A, B < cell/2.

    Expected: psi1[0][ix, iy, iz] depends only on iy, psi1[2] only on ix, psi1[1] == 0 (axis order x,y,z -> index 0,1,2).
    Box 12 Mpc/h, n = 12 (cell 1 Mpc/h), A = 0.4, B = 0.3."""
    n, box = 12, 12.0
    q = lattice_q(n, box)
    psi = np.zeros_like(q)
    psi[0] = 0.4 * np.sin(2 * np.pi * q[1] / box)
    psi[2] = 0.3 * np.cos(2 * np.pi * q[0] / box)
    d = run_kpc(tmp_path, particles(q + psi), box=box)
    P = d["psi1"]
    assert np.allclose(P, psi, atol=1e-5)
    # the x-displacement varies along index 1 only; the z-displacement along index 0 only
    assert np.ptp(P[0], axis=1).max() > 0.5 and np.ptp(P[0], axis=0).max() < 1e-5
    assert np.ptp(P[2], axis=0).max() > 0.4 and np.ptp(P[2], axis=1).max() < 1e-5
    assert np.allclose(P[1], 0.0, atol=1e-5)


def test_zeldovich_round_trip_gaussian_field(tmp_path):
    """Round trip: a Gaussian Psi1 from gtweb.gaussian_psi1 (z=0 amplitude), placed at q + a_ic Psi1 in an EdS file
    (D = a, so growth = 1/a_ic), is recovered as psi1 == Psi1 at every cell.  a_ic is chosen so max |a Psi1| = 0.4 cell.
    Also checks the grid convention against gtweb.za_positions (q + Psi1 in grid order)."""
    n, box = 16, 40.0
    cell = box / n
    P = gtweb.gaussian_psi1(box, n, seed=3)["psi1"].astype(np.float64)
    a = 0.4 * cell / np.abs(P).max()
    q = lattice_q(n, box)
    x = q + a * P
    # same layout as the package's own Zel'dovich positions (grid order, cell centres)
    za = np.asarray(gtweb.za_positions(P.astype(np.float32), box), np.float64)
    dza = za - particles(q + P)
    dza -= box * np.round(dza / box)
    assert np.abs(dza).max() < 1e-4
    d = run_kpc(tmp_path, particles(x), a=a, omega0=1.0, box=box)
    assert d["n_empty"] == 0
    assert np.allclose(d["psi1"], P, atol=1e-5 / a)
    rms = np.sqrt(np.mean(np.sum(P ** 2, axis=0)))
    assert d["rms_psi1"] == pytest.approx(rms, rel=1e-4)


def test_cell_mean_of_two_particles(tmp_path):
    """'Each particle is assigned to the grid cell that contains its position ... The value of a cell is the mean psi
    of the particles assigned to it.'  Unperturbed lattice except the particle of cell (0, 3, 5) moved +0.6 cell in x:
    it lands in cell (1, 3, 5) at x = 1.1 cell, q = 1.5 cell -> psi_x = -0.4 cell; that cell's own particle has
    psi = 0 -> mean psi_x = -0.2 cell.  Cell (0, 3, 5) is empty (n_empty == 1) and is filled with
    G*(P filled)/G*(filled) (sigma 1 cell, wrap), computed here with scipy."""
    q = lattice_q(N8, BOX)
    x = q.copy()
    x[0, 0, 3, 5] += 0.6 * CELL
    d = run_kpc(tmp_path, particles(x))
    assert d["n_empty"] == 1
    P = d["psi1"]
    assert P[0, 1, 3, 5] == pytest.approx(-0.2 * CELL, abs=1e-5)
    assert np.allclose(P[1:, 1, 3, 5], 0.0, atol=1e-5)
    filled = np.ones((N8,) * 3)
    filled[0, 3, 5] = 0.0
    Pf = np.zeros((3,) + (N8,) * 3)
    Pf[0, 1, 3, 5] = -0.2 * CELL
    den = gaussian_filter(filled, sigma=1.0, mode="wrap")
    for k in range(3):
        exp = gaussian_filter(Pf[k] * filled, sigma=1.0, mode="wrap")[0, 3, 5] / den[0, 3, 5]
        assert P[k, 0, 3, 5] == pytest.approx(exp, abs=1e-6), k
    # filled cells are untouched: everything except (0,3,5) and (1,3,5) is 0
    mask = np.ones((N8,) * 3, bool)
    mask[0, 3, 5] = mask[1, 3, 5] = False
    assert np.allclose(P[:, mask], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------------------------- periodicity (P4)
def test_unwrapped_coordinates_same_as_wrapped(tmp_path):
    """P4: shift s = (+0.3, -0.3, +0.2) cell.  Particles of the last x-cell sit at box - 0.2 cell and those of the first
    y-cell at 0.2 cell.  Writing them unwrapped (x - box -> slightly below 0; y + box -> at/above box) gives the same
    psi1 as the wrapped file, and both equal s."""
    s = np.array([0.3, -0.3, 0.2]) * CELL
    q = lattice_q(N8, BOX)
    x = q + s[:, None, None, None]
    xw = x % BOX
    xu = xw.copy()
    xu[0, -1] -= BOX          # last x-layer written as -0.2 cell
    xu[1, :, 0] += BOX        # first y-layer written as box + 0.2 cell
    assert (xu[0] < 0).any() and (xu[1] >= BOX).any()
    dw = run_kpc(tmp_path, particles(xw), name="w")
    du = run_kpc(tmp_path, particles(xu), name="u")
    assert np.allclose(du["psi1"], dw["psi1"], atol=1e-6 * CELL * 10)
    for k in range(3):
        assert np.allclose(du["psi1"][k], s[k], atol=1e-5)
    assert du["n_empty"] == 0


def test_negative_shift_first_cell_is_minimum_image(tmp_path):
    """P4: particles of the first x-cell shifted by -0.4 cell (x = 0.1 cell) give psi_x = -0.4 cell, not ~ +box;
    particles of the last x-cell shifted +0.4 cell (x = box - 0.1 cell) give +0.4 cell.  All |psi| <= cell/2."""
    q = lattice_q(N8, BOX)
    x = q.copy()
    x[0, 0] -= 0.4 * CELL
    x[0, -1] += 0.4 * CELL
    d = run_kpc(tmp_path, particles(x))
    P = d["psi1"]
    assert np.allclose(P[0, 0], -0.4 * CELL, atol=1e-5)
    assert np.allclose(P[0, -1], 0.4 * CELL, atol=1e-5)
    assert np.abs(P).max() <= CELL / 2 + 1e-5


def test_coordinate_exactly_at_box_wraps_to_first_cell(tmp_path):
    """P4 ('at/above box'): the particle of cell (0, 2, 6) written at x = BoxSize exactly wraps to x = 0, i.e. into
    cell 0 (n_empty == 0, no cell n), with |psi_x| = cell/2 (sign ambiguous at exactly half a cell).  Mpc/h file."""
    q = lattice_q(N8, BOX)
    x = q.copy()
    x[0, 0, 2, 6] = BOX
    paths = write_split(tmp_path / "edge", particles(x), 1, box=BOX)
    d = gtweb.psi1_from_gadget([str(p) for p in paths], length_unit=1.0)
    assert d["n_empty"] == 0
    assert abs(d["psi1"][0, 0, 2, 6]) == pytest.approx(CELL / 2, abs=1e-5)
    mask = np.ones((N8,) * 3, bool)
    mask[0, 2, 6] = False
    assert np.allclose(d["psi1"][:, mask], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------------------------- invariance (P5)
def _smooth_config(n=N8, box=BOX):
    q = lattice_q(n, box)
    psi = np.zeros_like(q)
    cell = box / n
    psi[0] = 0.35 * cell * np.sin(2 * np.pi * q[1] / box + 0.3)
    psi[1] = 0.25 * cell * np.cos(2 * np.pi * q[2] / box)
    psi[2] = 0.2 * cell * np.sin(2 * np.pi * (q[0] + q[1]) / box)
    return q + psi, psi


def test_split_and_shuffle_invariance(tmp_path):
    """P5: the same particles in 1 file, in 4 files, and in 4 files shuffled within each file (and the particle list
    globally permuted) give the same psi1 (atol 1e-6 cell)."""
    x, psi = _smooth_config()
    pts = particles(x)
    d1 = run_kpc(tmp_path, pts, name="one", nfiles=1)
    d4 = run_kpc(tmp_path, pts, name="four", nfiles=4)
    perm = np.random.default_rng(1).permutation(len(pts))
    paths = write_split(tmp_path / "shuf", pts[perm] * KPC, 4, shuffle_seed=2, box=BOX * KPC)
    ds = gtweb.psi1_from_gadget([str(p) for p in paths])
    assert np.allclose(d4["psi1"], d1["psi1"], rtol=0, atol=1e-6 * CELL)
    assert np.allclose(ds["psi1"], d1["psi1"], rtol=0, atol=1e-6 * CELL)
    assert np.allclose(d1["psi1"], psi, atol=1e-5)


def test_explicit_list_any_order_same_result(tmp_path):
    """P5: an explicit list is 'used in the given order'; the result does not depend on that order (reversed list)."""
    x, _ = _smooth_config()
    paths = [str(p) for p in write_split(tmp_path / "ic", particles(x) * KPC, 4, box=BOX * KPC)]
    a = gtweb.psi1_from_gadget(paths)
    b = gtweb.psi1_from_gadget(paths[::-1])
    assert np.allclose(a["psi1"], b["psi1"], rtol=0, atol=1e-6 * CELL)


def test_directory_input_reads_all_files_numerically(tmp_path):
    """P5: a directory with GADGET.0 .. GADGET.11 (12 files, so GADGET.10/11 sort after GADGET.2 only numerically)
    reads all 12 files (num_files = 12 passes, n_empty = 0) and equals the explicit-list input in numeric order."""
    x, psi = _smooth_config()
    paths = write_split(tmp_path / "icdir", particles(x) * KPC, 12, box=BOX * KPC)
    dd = gtweb.psi1_from_gadget(str(tmp_path / "icdir"))
    dl = gtweb.psi1_from_gadget([str(p) for p in paths])
    assert dd["n_empty"] == 0
    assert np.allclose(dd["psi1"], dl["psi1"], rtol=0, atol=1e-6 * CELL)
    assert np.allclose(dd["psi1"], psi, atol=1e-5)


def test_directory_input_ignores_other_files(tmp_path):
    """P5: 'every file in it named <stem>.<integer> with the same stem is used': a README and an .npz in the same
    directory are not read as GADGET files (the result equals the explicit list)."""
    x, _ = _smooth_config()
    paths = write_split(tmp_path / "icdir", particles(x) * KPC, 3, box=BOX * KPC)
    (tmp_path / "icdir" / "README").write_text("not a gadget file\n")
    np.savez(tmp_path / "icdir" / "psi1.npz", a=np.zeros(3))
    dd = gtweb.psi1_from_gadget(str(tmp_path / "icdir"))
    dl = gtweb.psi1_from_gadget([str(p) for p in paths])
    assert np.allclose(dd["psi1"], dl["psi1"], rtol=0, atol=1e-6 * CELL)


def test_directory_with_two_file_sets_raises(tmp_path):
    """P5: a directory holding two sets <stemA>.<int> and <stemB>.<int> is ambiguous: ValueError naming both, rather
    than silently mixing or picking one (the explicit-list input still works)."""
    x, _ = _smooth_config()
    paths = write_split(tmp_path / "icdir", particles(x) * KPC, 2, box=BOX * KPC)
    write_split(tmp_path / "icdir", particles(x) * KPC, 2, stem="other", box=BOX * KPC)
    with pytest.raises(ValueError, match="GADGET.*other|other.*GADGET"):
        gtweb.psi1_from_gadget(str(tmp_path / "icdir"))
    gtweb.psi1_from_gadget([str(p) for p in paths])


# ---------------------------------------------------------------------------------------------- empty cells (P6)
# Removing a particle would make N = n^3 - 1, which P9 rejects (not a cube).  A cell is emptied instead by moving its
# particle exactly onto another particle's position (a "donor" in a different cell): the donor cell then holds two
# identical displacements, so its mean is unchanged, and N stays n^3.
def empty_cells(x, cells, donors):
    """Grid-order (3, n, n, n) positions -> (n^3, 3) particles with each cell in `cells` emptied onto `donors`."""
    x = x.copy()
    for c, dnr in zip(cells, donors):
        x[(slice(None),) + tuple(c)] = x[(slice(None),) + tuple(dnr)]
    return particles(x)


def test_one_empty_cell_constant_field(tmp_path):
    """P6: uniform shift s, the particle of cell (3, 1, 6) moved onto the particle of cell (6, 6, 2) -> n_empty == 1
    and the empty cell's value == s (the Gaussian-weighted mean of a constant field is the constant)."""
    s = np.array([0.3, -0.2, 0.1]) * CELL
    q = lattice_q(N8, BOX)
    pts = empty_cells(q + s[:, None, None, None], [(3, 1, 6)], [(6, 6, 2)])
    d = run_kpc(tmp_path, pts)
    assert d["n_empty"] == 1
    for k in range(3):
        assert d["psi1"][k, 3, 1, 6] == pytest.approx(s[k], abs=1e-5), k
        assert np.allclose(d["psi1"][k], s[k], atol=1e-5)


def test_several_empty_cells_counted(tmp_path):
    """P6: 5 cells emptied (incl. an adjacent pair and both box corners) -> n_empty == 5; every cell finite and == s."""
    s = np.array([-0.1, 0.4, 0.25]) * CELL
    q = lattice_q(N8, BOX)
    cells = [(0, 0, 0), (0, 0, 1), (1, 1, 5), (4, 5, 4), (7, 7, 7)]
    donors = [(3, 3, 3), (3, 3, 3), (2, 6, 0), (6, 1, 2), (5, 2, 6)]
    d = run_kpc(tmp_path, empty_cells(q + s[:, None, None, None], cells, donors))
    assert d["n_empty"] == 5
    assert np.isfinite(d["psi1"]).all()
    for k in range(3):
        assert np.allclose(d["psi1"][k], s[k], atol=1e-5), k


def test_empty_cell_fill_rule_nonconstant_field(tmp_path):
    """P6 (fill rule): smooth non-constant field, cells (2, 5, 7) and (6, 0, 0) emptied.  Each empty cell equals
    G*(P filled)/G*(filled) at that cell (scipy gaussian_filter, sigma=1, mode='wrap', P = true psi on filled cells);
    filled cells keep their exact value."""
    n, box = 12, 12.0
    x, psi = _smooth_config(n, box)
    cells, donors = [(2, 5, 7), (6, 0, 0)], [(9, 9, 1), (3, 10, 4)]
    filled = np.ones((n,) * 3)
    for c in cells:
        filled[c] = 0.0
    d = run_kpc(tmp_path, empty_cells(x, cells, donors), box=box)
    assert d["n_empty"] == 2
    den = gaussian_filter(filled, sigma=1.0, mode="wrap")
    for k in range(3):
        fill = gaussian_filter(psi[k] * filled, sigma=1.0, mode="wrap") / den
        for c in cells:
            assert d["psi1"][k][c] == pytest.approx(fill[c], abs=1e-5), (k, c)
        assert np.allclose(d["psi1"][k][filled > 0], psi[k][filled > 0], atol=1e-5)


def test_removed_particle_is_not_a_cube(tmp_path):
    """P6 vs P9: literally removing one particle from an 8^3 lattice leaves N = 511, which is not a cube ->
    ValueError (the cube check takes precedence; empty cells arise only with N = n^3)."""
    q = lattice_q(N8, BOX)
    with pytest.raises(ValueError):
        run_kpc(tmp_path, particles(q)[1:])


# ---------------------------------------------------------------------------------------------- units (P7)
def test_kpc_and_mpc_units_agree(tmp_path):
    """P7: the same configuration written in kpc/h (BoxSize 10000, length_unit=1e-3) and in Mpc/h (BoxSize 10,
    length_unit=1.0) gives the same psi1 and box_size = 10."""
    x, psi = _smooth_config()
    pts = particles(x)
    dk = run_kpc(tmp_path, pts, name="kpc")                    # default length_unit = 1e-3
    pm = write_split(tmp_path / "mpc", pts, 1, box=BOX)
    dm = gtweb.psi1_from_gadget([str(p) for p in pm], length_unit=1.0)
    assert dk["box_size"] == pytest.approx(BOX)
    assert dm["box_size"] == pytest.approx(BOX)
    assert np.allclose(dk["psi1"], dm["psi1"], atol=1e-5)
    assert np.allclose(dm["psi1"], psi, atol=1e-5)


def test_default_length_unit_is_kpc(tmp_path):
    """P7: length_unit default 1e-3 -> box_size = BoxSize * 1e-3 (BoxSize 10000 kpc/h -> 10 Mpc/h)."""
    q = lattice_q(N8, BOX)
    d = run_kpc(tmp_path, particles(q))
    assert d["box_size"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------- cosmology (P8)
def test_growth_default_om_from_header(tmp_path):
    """P8: omega_m=None -> Om = header Omega0 (0.307); growth = D(1)/D(a_ic) for flat LCDM (reference by quad);
    growth > 1 for a_ic = 1/121 (z = 120); psi1 = s * growth for a uniform shift s."""
    a = 1.0 / 121.0
    s = np.array([0.3, -0.2, 0.1]) * CELL
    q = lattice_q(N8, BOX)
    d = run_kpc(tmp_path, particles(q + s[:, None, None, None]), a=a, omega0=0.307)
    g = growth_ref(a, 0.307)
    assert d["omega_m"] == 0.307
    assert d["growth"] > 1.0
    assert d["growth"] == pytest.approx(g, rel=1e-6)
    for k in range(3):
        assert np.allclose(d["psi1"][k], s[k] * g, rtol=1e-4, atol=1e-5 * g), k


def test_omega_m_argument_overrides_header(tmp_path):
    """P8: omega_m=0.25 overrides the header's Omega0 = 0.307: returned omega_m == 0.25, growth == D(1)/D(a) for
    Om = 0.25 (which differs from the Om = 0.307 value), psi1 scales with it."""
    a = 0.1
    s = np.array([0.3, -0.2, 0.1]) * CELL
    q = lattice_q(N8, BOX)
    pts = particles(q + s[:, None, None, None])
    d0 = run_kpc(tmp_path, pts, name="hdr", a=a, omega0=0.307)
    d1 = run_kpc(tmp_path, pts, name="arg", a=a, omega0=0.307, omega_m=0.25)
    assert d1["omega_m"] == 0.25
    assert d0["growth"] == pytest.approx(growth_ref(a, 0.307), rel=1e-6)
    assert d1["growth"] == pytest.approx(growth_ref(a, 0.25), rel=1e-6)
    assert abs(d1["growth"] - d0["growth"]) > 1e-3
    assert np.allclose(d1["psi1"] / d1["growth"], d0["psi1"] / d0["growth"], atol=1e-5)


def test_growth_eds_and_a1():
    """Sanity of the reference growth used above: EdS -> 1/a; a = 1 -> 1 for any Om."""
    assert growth_ref(0.2, 1.0) == pytest.approx(5.0, rel=1e-9)
    assert growth_ref(1.0, 0.3) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------------------------- errors (P9)
def _lattice_files(tmp_path, nfiles=2, **kw):
    q = lattice_q(N8, BOX)
    return [str(p) for p in write_split(tmp_path / "ic", particles(q) * KPC, nfiles, box=BOX * KPC, **kw)]


def test_error_num_files_mismatch(tmp_path):
    """P9: header num_files (3) != number of files given (2) -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    N = len(q)
    p0 = write_gadget(tmp_path / "GADGET.0", q[:200], npart_total=[0, N, 0, 0, 0, 0], num_files=3, box=BOX * KPC)
    p1 = write_gadget(tmp_path / "GADGET.1", q[200:], npart_total=[0, N, 0, 0, 0, 0], num_files=3, box=BOX * KPC)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p0), str(p1)])


def test_error_subset_of_files(tmp_path):
    """P9: passing only 1 of 2 files -> num_files mismatch (and sum npart != npart_total) -> ValueError."""
    paths = _lattice_files(tmp_path, nfiles=2)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget(paths[:1])


@pytest.mark.parametrize("field", ["a", "box"])
def test_error_files_disagree(tmp_path, field):
    """P9: files disagree on a (1.0 vs 0.5) or BoxSize (10000 vs 10001) -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    N = len(q)
    kw0 = dict(npart_total=[0, N, 0, 0, 0, 0], num_files=2, box=BOX * KPC, a=1.0)
    kw1 = dict(kw0)
    kw1["a" if field == "a" else "box"] = 0.5 if field == "a" else BOX * KPC + 1.0
    p0 = write_gadget(tmp_path / "GADGET.0", q[:100], **kw0)
    p1 = write_gadget(tmp_path / "GADGET.1", q[100:], **kw1)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p0), str(p1)])


def test_error_files_disagree_npart_total(tmp_path):
    """P9: files disagree on npart_total -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    N = len(q)
    p0 = write_gadget(tmp_path / "GADGET.0", q[:100], npart_total=[0, N, 0, 0, 0, 0], num_files=2, box=BOX * KPC)
    p1 = write_gadget(tmp_path / "GADGET.1", q[100:], npart_total=[0, N + 1, 0, 0, 0, 0], num_files=2,
                      box=BOX * KPC)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p0), str(p1)])


def test_error_sum_npart_not_total(tmp_path):
    """P9: one file, npart = 512 but npart_total = 520 -> sum npart != npart_total -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    p = write_gadget(tmp_path / "GADGET.0", q, npart_total=[0, 520, 0, 0, 0, 0], box=BOX * KPC)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p)])


def test_error_not_a_cube(tmp_path):
    """P9: N = 511 particles (not n^3) -> ValueError."""
    q = particles(lattice_q(N8, BOX))[:-1] * KPC
    p = write_gadget(tmp_path / "GADGET.0", q, box=BOX * KPC)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p)])


def test_error_other_particle_type(tmp_path):
    """P9: npart = [8, 504, 0, 0, 0, 0] (type 0 present, total 512 = 8^3 so only the type rule fails) -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    p = write_gadget(tmp_path / "GADGET.0", q, npart=[8, 504, 0, 0, 0, 0], npart_total=[8, 504, 0, 0, 0, 0],
                     box=BOX * KPC)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p)])


def test_error_corrupt_marker_propagates(tmp_path):
    """P9: a corrupt block marker in one of the files -> ValueError from psi1_from_gadget."""
    q = particles(lattice_q(N8, BOX)) * KPC
    p = write_gadget(tmp_path / "GADGET.0", q, box=BOX * KPC, bad_marker=True)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p)])


def test_error_wrong_position_block_propagates(tmp_path):
    """P9: position block of the wrong size (511 particles' bytes for npart 512) -> ValueError."""
    q = particles(lattice_q(N8, BOX)) * KPC
    p = write_gadget(tmp_path / "GADGET.0", q, box=BOX * KPC, velocities=False,
                     pos_payload=q[:-1].astype("<f4").tobytes())
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p)])


# ---------------------------------------------------------------------------------------------- output (P10)
def test_output_npz_round_trip(tmp_path):
    """P10: out=... writes an .npz that gtweb.lpt.load_psi1 reads back with the same psi1 (exactly), box_size, ngrid,
    growth and rms_psi1; rms_psi1 = sqrt(mean over cells of |psi1|^2) of the returned array."""
    x, _ = _smooth_config()
    out = tmp_path / "psi1_out.npz"
    d = run_kpc(tmp_path, particles(x), a=0.5, omega0=0.307, out=str(out))
    assert out.exists()
    r = lpt.load_psi1(str(out))
    assert np.array_equal(r["psi1"], d["psi1"])
    assert r["psi1"].dtype == np.float32
    assert r["box_size"] == pytest.approx(d["box_size"], rel=1e-12)
    assert r["ngrid"] == d["ngrid"] == N8
    assert float(r["growth"]) == pytest.approx(d["growth"], rel=1e-12)
    assert float(r["rms_psi1"]) == pytest.approx(d["rms_psi1"], rel=1e-12)
    P = d["psi1"].astype(np.float64)
    assert d["rms_psi1"] == pytest.approx(np.sqrt(np.mean(np.sum(P ** 2, axis=0))), rel=1e-6)


def test_output_accepted_by_generate_from_ic(tmp_path):
    """P10: the written npz 'gtweb.generate_from_ic accepts as psi1_path' (ic_path=None, 40 Mpc/h, 16^3, a tiny
    run with fit_iters=20).  Psi1 is a Gaussian field placed in an EdS file at a_ic chosen so max displacement
    is 0.4 cell."""
    n, box = 16, 40.0
    P = gtweb.gaussian_psi1(box, n, seed=1)["psi1"].astype(np.float64)
    a = 0.4 * (box / n) / np.abs(P).max()
    x = lattice_q(n, box) + a * P
    out = tmp_path / "psi1.npz"
    d = run_kpc(tmp_path, particles(x), a=a, omega0=1.0, box=box, out=str(out))
    assert np.allclose(d["psi1"], P, atol=1e-5 / a)
    meta = gtweb.generate_from_ic(None, tmp_path / "mock", psi1_path=str(out), fit_iters=20, verbose=False)
    assert isinstance(meta, dict)


# ---------------------------------------------------------------------------------------------- real data (P11)
_IC_DIR = os.environ.get("GTWEB_TEST_IC_DIR")
_PSI1 = os.environ.get("GTWEB_TEST_PSI1")


@pytest.mark.skipif(not (_IC_DIR and _PSI1), reason="set GTWEB_TEST_IC_DIR and GTWEB_TEST_PSI1 to run")
def test_reproduces_paper_psi1_bit_for_bit():
    """P11: psi1_from_gadget(<the paper's GADGET.* directory>, omega_m=0.307) equals the paper's stored Psi1 exactly
    (np.array_equal, float32); box_size, ngrid equal and growth to 1e-12 relative."""
    ref = lpt.load_psi1(_PSI1)
    d = gtweb.psi1_from_gadget(_IC_DIR, omega_m=0.307)
    assert d["psi1"].dtype == np.float32
    assert d["psi1"].shape == ref["psi1"].shape
    assert np.array_equal(d["psi1"], ref["psi1"])
    assert d["box_size"] == ref["box_size"]
    assert d["ngrid"] == int(ref["ngrid"])
    assert d["growth"] == pytest.approx(float(ref["growth"]), rel=1e-12)


# ============================================================================================ REVIEWER TESTS
# Added by the code review of 1.2.0 (not from the spec-tester).  The first five FAIL on the reviewed code; the last
# three pass and pin that unsupported layouts fail loudly.
def _patch_header(path, offset, fmt, *vals):
    """Overwrite header bytes at `offset` (header-relative; +4 skips the leading record marker)."""
    b = bytearray(Path(path).read_bytes())
    struct.pack_into(fmt, b, 4 + offset, *vals)
    Path(path).write_bytes(bytes(b))


def test_reviewer_nallhw_high_word_at_offset_168(tmp_path):
    """GADGET-2 / Ginnungagap header: flag_stellarage@160, flag_metal@164, npartTotalHighWord[6]@168
    (ginnungagap src/libutil/gadgetHeader.c, local_actualReadHeader).  nall[1] = 512 with nallhighw[1] = 1 (byte 172)
    means 2^32 + 512 type-1 particles."""
    p = write_gadget(tmp_path / "GADGET.0", particles(lattice_q(N8, BOX)))
    _patch_header(p, 172, "<I", 1)
    _, hdr = gtweb.read_gadget(str(p))
    assert hdr["npart_total"] == [0, 2 ** 32 + 512, 0, 0, 0, 0]


def test_reviewer_flag_stellarage_is_not_a_particle_count(tmp_path):
    """A valid 8^3 type-1 IC whose header has flag_stellarage = 1 (byte 160) is accepted."""
    paths = write_split(tmp_path / "ic", particles(lattice_q(N8, BOX)) * KPC, 1, box=BOX * KPC)
    _patch_header(paths[0], 160, "<i", 1)
    d = gtweb.psi1_from_gadget([str(paths[0])])
    assert d["n_empty"] == 0


def test_reviewer_flag_metal_is_not_a_particle_count(tmp_path):
    """flag_metal = 1 (byte 164) does not change the type-1 total."""
    paths = write_split(tmp_path / "ic", particles(lattice_q(N8, BOX)) * KPC, 1, box=BOX * KPC)
    _patch_header(paths[0], 164, "<i", 1)
    _, hdr = gtweb.read_gadget(str(paths[0]))
    assert hdr["npart_total"][1] == 512


def test_reviewer_same_file_twice_is_rejected(tmp_path):
    """[GADGET.0, GADGET.0] for a 2-file set of equal halves passes every header check (num_files 2, sum npart =
    npart_total, cube), so half the particles are used twice and half never (256 of 512 cells filled by smoothing).
    The real 384^3 set has 64 equal files, so [GADGET.0] * 64 passes the same way."""
    pts = particles(lattice_q(N8, BOX)) * KPC
    N = len(pts)
    kw = dict(npart_total=[0, N, 0, 0, 0, 0], num_files=2, box=BOX * KPC)
    p0 = write_gadget(tmp_path / "GADGET.0", pts[: N // 2], **kw)
    write_gadget(tmp_path / "GADGET.1", pts[N // 2:], **kw)
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([str(p0), str(p0)])


def test_reviewer_empty_list_is_valueerror():
    """No files -> ValueError (the reviewed code raises TypeError: 'NoneType' object is not subscriptable)."""
    with pytest.raises(ValueError):
        gtweb.psi1_from_gadget([])


def test_reviewer_big_endian_fails_loudly(tmp_path):
    pos = particles(lattice_q(N8, BOX)).astype(">f4")
    hdr = bytearray(256)
    struct.pack_into(">6i", hdr, 0, 0, 512, 0, 0, 0, 0)
    struct.pack_into(">d", hdr, 72, 1.0)
    struct.pack_into(">6I", hdr, 96, 0, 512, 0, 0, 0, 0)
    struct.pack_into(">i", hdr, 124, 1)
    struct.pack_into(">d", hdr, 128, BOX)
    b = struct.pack(">i", 256) + bytes(hdr) + struct.pack(">i", 256)
    b += struct.pack(">i", pos.nbytes) + pos.tobytes() + struct.pack(">i", pos.nbytes)
    p = tmp_path / "be.0"
    p.write_bytes(b)
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(p))


def test_reviewer_format2_fails_loudly(tmp_path):
    p = write_gadget(tmp_path / "f1.0", particles(lattice_q(N8, BOX)))
    lab = struct.pack("<i", 8) + b"HEAD" + struct.pack("<i", 264) + struct.pack("<i", 8)
    q = tmp_path / "f2.0"
    q.write_bytes(lab + p.read_bytes())
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(q))


def test_reviewer_double_precision_positions_fail_loudly(tmp_path):
    """Ginnungagap can write flagdoubleprecision = 1 files (24 bytes per particle): rejected, not misread."""
    pts = particles(lattice_q(N8, BOX))
    p = write_gadget(tmp_path / "d.0", pts, pos_payload=pts.astype("<f8").tobytes(), velocities=False)
    with pytest.raises(ValueError):
        gtweb.read_gadget(str(p))


def test_reviewer_num_files_must_agree_between_files(tmp_path):
    """num_files is checked in every file, not only the first: file 0 says 2, file 1 says 3 -> ValueError."""
    pts = particles(lattice_q(N8, BOX)) * KPC
    N = len(pts)
    kw = dict(npart_total=[0, N, 0, 0, 0, 0], box=BOX * KPC)
    p0 = write_gadget(tmp_path / "GADGET.0", pts[: N // 2], num_files=2, **kw)
    p1 = write_gadget(tmp_path / "GADGET.1", pts[N // 2:], num_files=3, **kw)
    with pytest.raises(ValueError, match="num_files"):
        gtweb.psi1_from_gadget([str(p0), str(p1)])


def test_reviewer_out_without_suffix_gets_npz(tmp_path):
    """out='psi1' writes psi1.npz, the returned 'out' names it, and load_psi1 reads it."""
    paths = write_split(tmp_path / "ic", particles(lattice_q(N8, BOX)) * KPC, 1, box=BOX * KPC)
    d = gtweb.psi1_from_gadget([str(paths[0])], out=str(tmp_path / "psi1"))
    assert d["out"] == str(tmp_path / "psi1.npz")
    assert np.array_equal(lpt.load_psi1(d["out"])["psi1"], d["psi1"])


def test_reviewer_wrong_length_unit_is_rejected(tmp_path):
    """A file in Mpc/h read with the default length_unit (kpc/h) gives a 0.01 Mpc/h box: ValueError, not a Psi1
    1000 times too small."""
    paths = write_split(tmp_path / "ic", particles(lattice_q(N8, BOX)), 1, box=BOX)
    with pytest.raises(ValueError, match="length unit"):
        gtweb.psi1_from_gadget([str(paths[0])])
    assert gtweb.psi1_from_gadget([str(paths[0])], length_unit=1.0)["box_size"] == BOX


def test_reviewer_too_many_empty_cells_is_rejected(tmp_path):
    """Displacements over half a cell: particles of even-x cells moved +0.6 cell land in odd-x cells, leaving half the
    cells empty -> ValueError (the binning is valid only for displacements under half a cell)."""
    x = lattice_q(N8, BOX)
    ix = np.arange(N8)[:, None, None] * np.ones((1, N8, N8))
    x[0] = x[0] + np.where(ix % 2 == 0, 0.6 * CELL, 0.0)
    paths = write_split(tmp_path / "ic", particles(x) * KPC, 1, box=BOX * KPC)
    with pytest.raises(ValueError, match="no particle"):
        gtweb.psi1_from_gadget([str(paths[0])])

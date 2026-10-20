"""A generated mock is complete, self-consistent and deterministic."""
import json

import numpy as np
from scipy.spatial import cKDTree

import gtweb
from conftest import BOX, NGRID, SEED


def _spine_points(spines, box, step=0.25):
    pts = []
    for s in spines:
        p = np.asarray(s, float); d = np.diff(p, axis=0); d -= box * np.round(d / box)
        cur = p[0].copy(); pts.append(cur[None] % box)
        for seg in d:
            ns = max(1, int(np.linalg.norm(seg) / step)); t = np.arange(1, ns + 1)[:, None] / ns
            pts.append((cur + t * seg) % box); cur = cur + seg
    return np.vstack(pts)


def test_outputs_and_metadata(mock):
    out, meta = mock
    for f in ("ground_truth_spines.npz", "nodes_catalogue.npz", "web_geometry.npz", "halos_catalogue.npz",
              f"mock_particles_s{SEED}.npz", "velocity_field.npz", "metadata.json"):
        assert (out / f).exists(), f
    assert json.loads((out / "metadata.json").read_text())["n_filaments"] == meta["n_filaments"]
    assert meta["n_voids"] > 10 and meta["n_filaments"] > 50


def test_ground_truth_geometry(mock):
    out, meta = mock
    gt = gtweb.load_ground_truth(out / "ground_truth_spines.npz")
    assert len(gt["spines"]) == meta["n_filaments"]
    for s in gt["spines"]:
        s = np.asarray(s)
        assert (s >= 0).all() and (s < BOX + 1e-3).all()
    deg = np.load(out / "web_geometry.npz")
    assert len(gtweb.load_nodes(out / "nodes_catalogue.npz")["positions"]) == meta["n_nodes"]


def test_particles(mock):
    out, _ = mock
    p = gtweb.load_particles(out / f"mock_particles_s{SEED}.npz")
    pos = np.asarray(p["positions"])
    assert pos.shape == (NGRID ** 3, 3)
    assert (pos >= 0).all() and (pos < BOX).all()
    assert set(np.unique(p["stage"])) <= {0, 1, 2, 3}
    v = np.asarray(p["velocities"])
    assert v.shape == pos.shape and np.isfinite(v).all()


def test_mass_sits_on_the_web(mock):
    out, _ = mock
    gt = gtweb.load_ground_truth(out / "ground_truth_spines.npz")
    tree = cKDTree(_spine_points(gt["spines"], BOX), boxsize=BOX)
    pos = np.asarray(gtweb.load_particles(out / f"mock_particles_s{SEED}.npz")["positions"], float)
    rng = np.random.default_rng(0)
    d_part = tree.query(pos[rng.choice(len(pos), 20000, replace=False)])[0]
    d_rand = tree.query(rng.uniform(0, BOX, (20000, 3)))[0]
    assert np.median(d_part) < 0.5 * np.median(d_rand), (np.median(d_part), np.median(d_rand))


def test_deterministic(mock, tmp_path):
    out, meta = mock
    meta2 = gtweb.generate_mock(tmp_path, box=BOX, ngrid=NGRID, seed=SEED, fit_iters=120, verbose=False)
    assert meta2["n_filaments"] == meta["n_filaments"]
    a = np.load(out / "ground_truth_spines.npz"); b = np.load(tmp_path / "ground_truth_spines.npz")
    assert np.array_equal(a["node_positions"], b["node_positions"])
    pa = np.load(out / f"mock_particles_s{SEED}.npz")["positions"]
    pb = np.load(tmp_path / f"mock_particles_s{SEED}.npz")["positions"]
    assert np.array_equal(pa, pb)


def test_generate_from_own_ic(tmp_path):
    d = gtweb.gaussian_psi1(40.0, 32, seed=3)
    np.savez(tmp_path / "psi1.npz", **d)
    meta = gtweb.generate_from_ic(None, tmp_path / "mock", psi1_path=tmp_path / "psi1.npz",
                                  fit_iters=60, verbose=False)
    assert meta["n_particles"] == 32 ** 3 and meta["n_filaments"] > 20


def test_box_size_stored(mock):
    out, _ = mock
    assert float(np.load(out / "ground_truth_spines.npz")["box_size"]) == BOX
    assert float(gtweb.load_ground_truth(out / "ground_truth_spines.npz")["box_size"]) == BOX

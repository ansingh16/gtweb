"""The analytic skeleton must be the exact periodic power diagram of the nuclei."""
import numpy as np
from scipy.spatial import cKDTree

from gtweb import web


def _power_sorted(x, c, w, box, k=12):
    d = x[:, None, :] - c[None, :, :]
    d -= box * np.round(d / box)
    p = (d ** 2).sum(-1) - w[None, :]
    return np.sort(p, axis=1)[:, :k]


def test_power_diagram_is_exact():
    box = 40.0
    rng = np.random.default_rng(3)
    c = rng.uniform(0, box, (60, 3))
    w = rng.normal(0, 4.0, 60)                       # unequal cells
    wb = web.build_web_from_geometry(c, w, box)
    nodes = np.asarray(wb["nodes"], float)
    E = wb["edges"]

    # every node is a vertex: four nuclei tied at the smallest power distance
    p = _power_sorted(nodes, c, w, box)
    assert np.all(p[:, 3] - p[:, 0] < 1e-3)               # nodes are stored as float32

    # a generic 3D power diagram is 4-regular, so E = 2V, each edge stored once
    assert np.all(wb["node_degree"] == 4)
    assert len(E) == 2 * len(nodes) == int(wb["n_filaments"])
    assert len({(a, b) if a < b else (b, a) for a, b in E}) == len(E)

    # every filament lies on a three-cell junction and ends on its nodes
    tree = cKDTree(nodes, boxsize=box)
    for (i, j), s in zip(E, wb["spines"]):
        s = np.asarray(s, float)
        a = s[0]
        b = a + (s[-1] - a) - box * np.round((s[-1] - a) / box)
        mid = _power_sorted(((a + b) / 2)[None] % box, c, w, box)[0]
        assert mid[2] - mid[0] < 1e-3 and mid[3] - mid[2] > 1e-3
        assert set(tree.query(np.array([a % box, b % box]))[1]) == {i, j}


def test_small_box_needs_wider_padding():
    # few, large cells: the default 0.32-box image shell is too thin and must be widened
    box = 50.0
    rng = np.random.default_rng(0)
    wb = web.build_web_from_geometry(rng.uniform(0, box, (12, 3)), np.zeros(12), box)
    assert float(wb["margin"]) > 0.32 * box
    assert np.all(wb["node_degree"] == 4)

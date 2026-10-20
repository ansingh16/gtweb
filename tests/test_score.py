"""The scorer gives the right answers for known finders."""
import numpy as np

import gtweb
from conftest import BOX


def _spines(mock):
    out, _ = mock
    return gtweb.load_ground_truth(out / "ground_truth_spines.npz")


def test_truth_scores_perfectly(mock):
    sp = _spines(mock)["spines"]
    s = gtweb.score_finder(sp, sp, BOX, tol=1.0)
    assert s["completeness"] > 0.99 and s["purity"] > 0.99, s


def test_random_points_are_impure(mock):
    sp = _spines(mock)["spines"]
    rnd = np.random.default_rng(0).uniform(0, BOX, (20000, 3))
    assert gtweb.score_finder(rnd, sp, BOX, tol=1.0)["purity"] < 0.6


def test_half_the_web(mock):
    sp = _spines(mock)["spines"]
    rng = np.random.default_rng(1)
    keep = [s for s in sp if rng.random() < 0.5]
    s = gtweb.score_finder(keep, sp, BOX, tol=1.0)
    assert s["purity"] > 0.9 and 0.3 < s["completeness"] < 0.75, s


def test_tolerance_monotonic(mock):
    sp = _spines(mock)["spines"]
    rng = np.random.default_rng(2)
    noisy = [np.asarray(s) + rng.normal(0, 0.8, np.asarray(s).shape) for s in sp]
    assert (gtweb.score_finder(noisy, sp, BOX, tol=2.0)["completeness"]
            >= gtweb.score_finder(noisy, sp, BOX, tol=0.5)["completeness"])


def test_per_filament(mock):
    gt = _spines(mock); sp = gt["spines"]
    pf = gtweb.score_per_filament(sp, sp, BOX, tol=1.0)
    assert len(pf) == len(sp) and pf.min() > 0.99

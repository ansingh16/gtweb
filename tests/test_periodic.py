"""Periodic wrapping: x % box rounds to box itself for tiny negative x (-1e-17 % 100.0 == 100.0), which scipy's
cKDTree(boxsize=box) rejects and np.histogramdd puts in the last bin.  gtweb.periodic.wrap maps it to 0."""
import numpy as np
import pytest

import gtweb
from gtweb.periodic import wrap, polyline_length, sample_polyline
from gtweb import geometry


BOX = 100.0


def test_polyline_length_uses_minimum_image():
    p = np.array([[99.0, 5.0, 5.0], [1.0, 5.0, 5.0], [1.0, 8.0, 9.0]])      # 2 across the boundary, then 5
    assert polyline_length(p, BOX) == pytest.approx(7.0)
    assert polyline_length(p[:1], BOX) == 0.0


def test_sample_polyline_equal_pieces_midpoints():
    p = np.array([[10.0, 20.0, 30.0], [13.0, 20.0, 30.0], [13.0, 24.0, 30.0]])  # L = 7
    q = sample_polyline(p, 0.1, BOX)
    assert len(q) == 70                                       # ceil(7 / 0.1)
    np.testing.assert_allclose(q[0], [10.05, 20.0, 30.0])     # first piece's midpoint
    np.testing.assert_allclose(q[-1], [13.0, 23.95, 30.0])
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    assert np.all(steps <= 0.1 + 1e-12)
    assert len(sample_polyline(p, 2.0, BOX)) == 4             # ceil(7 / 2), last piece shorter than step -> n pieces
    assert len(sample_polyline(np.array([p[0], p[0]]), 0.1, BOX)) == 0


def test_sample_polyline_across_the_boundary():
    p = np.array([[99.5, 50.0, 50.0], [100.5, 50.0, 50.0]])   # crosses x = 100 (stored unwrapped)
    q = sample_polyline(p, 0.25, BOX)
    assert np.all((q >= 0) & (q < BOX))
    np.testing.assert_allclose(np.sort(q[:, 0]), [0.125, 0.375, 99.625, 99.875])


def test_numpy_edge_case_exists():
    assert (-1e-17 % BOX) == BOX                                   # the rounding this module guards against
    assert np.mod(np.float32(-1e-6), np.float32(BOX)) == np.float32(BOX)


def test_wrap_bounds_values_and_dtype():
    x = np.array([-1e-17, -1e-15, -3e-15, 0.0, 100.0, 250.5, -0.5, 37.25])
    y = wrap(x, BOX)
    assert np.all((y >= 0) & (y < BOX))
    np.testing.assert_array_equal(y, [0.0, 0.0, 0.0, 0.0, 0.0, 50.5, 99.5, 37.25])
    x32 = np.array([-1e-6, 5.5, 100.0], np.float32)
    y32 = wrap(x32, np.float32(BOX))
    assert y32.dtype == np.float32 and np.all((y32 >= 0) & (y32 < BOX))


def test_wrap_equals_mod_away_from_the_edge():
    x = np.random.default_rng(0).uniform(-300, 300, 100000)
    y = wrap(x, BOX)
    ok = np.mod(x, BOX) < BOX
    np.testing.assert_array_equal(y[ok], np.mod(x, BOX)[ok])


def test_score_finder_accepts_a_vertex_just_below_zero():
    """Regression: a skeleton vertex at -1e-17 used to crash cKDTree(boxsize=box)."""
    gt = [np.array([[10.0, 50, 50], [30.0, 50, 50]]), np.array([[0.0, 20, 20], [5.0, 20, 20]])]
    near = [np.array([[-1e-17, 20, 20], [5.0, 20, 20]])]
    at0 = [np.array([[0.0, 20, 20], [5.0, 20, 20]])]
    s = gtweb.score_finder(near, gt, BOX, tol=1.0)
    assert s == gtweb.score_finder(at0, gt, BOX, tol=1.0)
    cloud = np.array([[-1e-17, 20.0, 20.0], [2.0, 20.0, 20.0]])
    assert gtweb.score_finder(cloud, gt, BOX, tol=1.0)["purity"] == 1.0
    per = gtweb.score_per_filament(near, gt, BOX, tol=1.0)
    assert per[1] == 1.0 and per[0] == 0.0


def test_velocity_field_and_sampling_treat_minus_zero_as_zero():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, BOX, (5000, 3))
    a = pos.copy(); a[:50, 0] = 0.0
    b = pos.copy(); b[:50, 0] = -1e-17
    va = gtweb.linear_velocity_field(a, BOX, ngrid=16, smooth=0.0)
    vb = gtweb.linear_velocity_field(b, BOX, ngrid=16, smooth=0.0)
    np.testing.assert_array_equal(va["vx"], vb["vx"])
    np.testing.assert_array_equal(gtweb.sample_velocity(va, np.array([[-1e-17, 3.0, 4.0]])),
                                  gtweb.sample_velocity(va, np.array([[0.0, 3.0, 4.0]])))


@pytest.mark.parametrize("path", ["numpy", "default"])
def test_cic_puts_a_particle_at_box_in_cell_zero(path):
    """The CIC deposits keep a bare % box: a particle at exactly x = box must still land where x = 0 does
    (the cell indices are wrapped with % grid)."""
    p_box = np.array([[BOX, 30.0, 70.0]])
    p_0 = np.array([[0.0, 30.0, 70.0]])
    f = (lambda p: geometry._cic_numpy(p, 10, BOX)) if path == "numpy" else (lambda p: geometry.cic_density(p, 10, BOX))
    np.testing.assert_array_equal(f(p_box), f(p_0))
    assert f(p_box).sum() == pytest.approx(1.0)

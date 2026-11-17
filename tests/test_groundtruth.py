"""gtweb.groundtruth: a particle or spine vertex just below zero is the same as one at zero (periodic box).

With a bare ``x % box``, -1e-17 became ``box`` and np.histogramdd put the particle in the LAST cell, so
spine_overdensity read a different density field.  See gtweb.periodic.wrap."""
import numpy as np
import pytest

from gtweb import groundtruth

BOX = 20.0


def _field(rng):
    pos = rng.uniform(0, BOX, (20000, 3))
    pos[:300, 0] = 0.0                                   # a slab of particles on the x = 0 face
    return pos


def test_spine_overdensity_minus_zero_particles_equal_zero():
    rng = np.random.default_rng(3)
    a = _field(rng)
    b = a.copy()
    b[:300, 0] = -1e-17                                  # same particles, just below zero
    spines = [np.array([[0.2, y, 10.0], [3.0, y, 10.0]]) for y in (2.0, 7.0, 13.0)]
    oa = groundtruth.spine_overdensity(spines, a, 1.0, BOX, ngrid=10, smooth=0.0)
    ob = groundtruth.spine_overdensity(spines, b, 1.0, BOX, ngrid=10, smooth=0.0)
    np.testing.assert_array_equal(oa, ob)


def test_spine_overdensity_minus_zero_vertex_equals_zero():
    rng = np.random.default_rng(4)
    pos = _field(rng)
    s0 = [np.array([[0.0, 5.0, 5.0], [2.0, 5.0, 5.0]])]
    sm = [np.array([[-1e-17, 5.0, 5.0], [2.0, 5.0, 5.0]])]
    np.testing.assert_array_equal(groundtruth.spine_overdensity(s0, pos, 1.0, BOX, ngrid=10, smooth=0.0),
                                  groundtruth.spine_overdensity(sm, pos, 1.0, BOX, ngrid=10, smooth=0.0))


# ------------------------------------------------------------------- 1.1.0: cell-centred lookup, along-line mean
RHO_BAR = groundtruth.cosmo.Om * groundtruth.cosmo.RHO_CRIT


def _cells(box, ngrid, mask, k):
    """k particles at the centre of every cell where mask(x_centre) is True; positions and centres."""
    c = (np.arange(ngrid) + 0.5) * (box / ngrid)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = pts[mask(pts)]
    return np.repeat(pts, k, axis=0)


def test_density_is_read_at_the_cell_centre():
    """One occupied cell, no smoothing: reading at that cell's centre returns its density.  A lookup at x/cell
    (the 1.0.x bug) reads half a cell further along each axis and returns 1/8 of it."""
    box, ngrid, k = 20.0, 10, 8
    target = np.array([5.0, 13.0, 17.0])                     # centre of cell (2, 6, 8); cell = 2
    pos = _cells(box, ngrid, lambda p: np.all(p == target, axis=1), k)
    assert len(pos) == k
    cell = box / ngrid
    m_p = 3.0 * RHO_BAR * cell ** 3 / k                       # that cell holds 3 rho_bar
    point = [np.array([target, target])]                     # zero-length spine: read at its stored vertex
    o = groundtruth.spine_overdensity(point, pos, m_p, box, ngrid=ngrid, smooth=0.0)
    assert o[0] == pytest.approx(3.0, rel=1e-6)


def test_average_is_along_the_line_not_over_vertices():
    """Half the box (x < 50) at 2 rho_bar, the other half empty.  A spine from x = 30 to x = 70 (every step shorter
    than box/2, so no minimum-image wrap) has five of its six vertices bunched at 30-34 in the dense half.  Along the
    line: 17.5 Mpc/h at 2, a 5 Mpc/h linear ramp across x = 50 (mean 1), 17.5 Mpc/h at 0, so the mean is 1.0.
    The mean over the six vertices (1.0.x) would be 10/6."""
    box, ngrid, k = 100.0, 20, 2
    pos = _cells(box, ngrid, lambda p: p[:, 0] < 50.0, k)
    cell = box / ngrid
    m_p = 2.0 * RHO_BAR * cell ** 3 / k
    y = z = 52.5                                              # a row of cell centres
    spine = np.array([[30.0, y, z], [31.0, y, z], [32.0, y, z], [33.0, y, z], [34.0, y, z], [70.0, y, z]])
    o = groundtruth.spine_overdensity([spine], pos, m_p, box, ngrid=ngrid, smooth=0.0)
    assert o[0] == pytest.approx(1.0, abs=2e-3)


def test_overdensity_is_translation_invariant():
    """Shifting particles and spines by a whole number of cells (periodically) leaves every value unchanged."""
    rng = np.random.default_rng(7)
    box, ngrid = 20.0, 10
    pos = rng.uniform(0, box, (30000, 3))
    spines = [rng.uniform(0, box, (6, 3)) for _ in range(5)]
    shift = np.array([3, -7, 11]) * (box / ngrid)
    a = groundtruth.spine_overdensity(spines, pos, 1.0, box, ngrid=ngrid, smooth=1.0)
    b = groundtruth.spine_overdensity([s + shift for s in spines], pos + shift, 1.0, box, ngrid=ngrid, smooth=1.0)
    np.testing.assert_allclose(a, b, rtol=1e-5)

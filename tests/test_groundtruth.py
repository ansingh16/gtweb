"""gtweb.groundtruth: a particle or spine vertex just below zero is the same as one at zero (periodic box).

With a bare ``x % box``, -1e-17 became ``box`` and np.histogramdd put the particle in the LAST cell, so
spine_overdensity read a different density field.  See gtweb.periodic.wrap."""
import numpy as np

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

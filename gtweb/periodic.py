"""
Periodic-box helpers.
"""
import numpy as np


def wrap(x, box):
    """Coordinates modulo `box`, every element in [0, box), dtype kept.

    Plain ``x % box`` rounds to ``box`` itself when x is a tiny negative number (``-1e-15 % 100.0 == 100.0`` in
    float64; float32 does the same for x down to about -4e-6 at box 100).  scipy's ``cKDTree(..., boxsize=box)``
    rejects such a point, and ``np.histogramdd`` puts it in the last bin instead of the first.  This maps it to 0.
    """
    y = np.mod(x, box)
    return np.where(y >= box, y - box, y)

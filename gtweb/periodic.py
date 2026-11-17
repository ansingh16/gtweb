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


def min_image(d, box):
    """Minimum-image version of separation vectors `d` (each component in [-box/2, box/2])."""
    d = np.asarray(d, float)
    return d - box * np.round(d / box)


def polyline_length(p, box):
    """Length of an (M, 3) polyline with minimum-image segments; 0 for fewer than 2 vertices."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(min_image(np.diff(p, axis=0), box), axis=1).sum())


def sample_polyline(p, step, box):
    """Equal-arc-length samples of a polyline: n = max(1, ceil(L/step)) pieces, one sample at the arc-length
    midpoint of each, wrapped into [0, box).  Every sample stands for the same length L/n, so a mean over the
    samples is a length-weighted mean along the line.  Returns (n, 3); empty if the polyline has zero length."""
    p = np.asarray(p, float)
    L = polyline_length(p, box)
    if L <= 0:
        return np.empty((0, 3))
    n = max(1, int(np.ceil(L / step)))
    s = (np.arange(n) + 0.5) * (L / n)
    d = min_image(np.diff(p, axis=0), box)
    seg = np.linalg.norm(d, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(seg[k] > 0, (s - cum[k]) / seg[k], 0.0)
    return wrap(p[:-1][k] + t[:, None] * d[k], box)

"""The Gaussian random field has the requested power spectrum and the Zel'dovich conventions hold."""
import numpy as np

import gtweb
from gtweb import ic, lpt, cosmology


def test_power_spectrum_recovered():
    box, n = 200.0, 96
    ratios = []
    for seed in range(3):
        d = ic.gaussian_psi1(box, n, seed=seed)
        delta, _ = lpt.linear_density_from_psi1(d["psi1"], box)
        k, p = ic.measure_pk(delta.astype(np.float64), box, nbins=20)
        m = (k > 0.2) & (k < 0.8 * np.pi * n / box)
        ratios.append(p[m] / cosmology.linear_Pk(k[m]))
    r = np.mean(ratios, axis=0)
    assert np.all(np.abs(r - 1) < 0.08), r


def test_deterministic_and_seed_dependent():
    a = ic.gaussian_psi1(50.0, 32, seed=7)["psi1"]
    b = ic.gaussian_psi1(50.0, 32, seed=7)["psi1"]
    c = ic.gaussian_psi1(50.0, 32, seed=8)["psi1"]
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_za_positions_in_box():
    d = ic.gaussian_psi1(50.0, 32, seed=1)
    x = gtweb.za_positions(d["psi1"], 50.0)
    assert x.shape == (32 ** 3, 3)
    assert (x >= 0).all() and (x < 50.0).all()

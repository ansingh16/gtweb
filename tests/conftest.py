"""Shared fixture: one small mock (40 Mpc/h, 48^3 particles) generated once per test session."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gtweb  # noqa: E402

BOX, NGRID, SEED = 40.0, 48, 0


@pytest.fixture(scope="session")
def mock(tmp_path_factory):
    out = tmp_path_factory.mktemp("mock")
    meta = gtweb.generate_mock(out, box=BOX, ngrid=NGRID, seed=SEED, fit_iters=120, verbose=False)
    return out, meta

"""DisPerSE readers / writers."""
import numpy as np

import gtweb

NDSKL = """ANDSKEL
3
#comment
BBOX [0 0 0] [10 10 10]
[CRITICAL POINTS]
0
[FILAMENTS]
2
0 1 3
1.0 1.0 1.0
2.0 2.0 2.0
3.0 3.0 3.0
1 2 2
5.0 5.0 5.0
6.0 5.0 5.0
"""


def test_read_disperse_skeleton(tmp_path):
    f = tmp_path / "toy.a.NDskl"
    f.write_text(NDSKL)
    fils = gtweb.read_disperse_skeleton(f)
    assert len(fils) == 2 and fils[0].shape == (3, 3) and fils[1].shape == (2, 3)
    assert np.allclose(fils[1][1], [6, 5, 5])


def test_write_disperse_pos(tmp_path):
    pts = np.random.default_rng(0).uniform(0, 10, (100, 3))
    gtweb.write_disperse_pos(pts, tmp_path / "p.pos", 10.0)
    txt = (tmp_path / "p.pos").read_text()
    assert len(txt.splitlines()) >= 100


NDSKL_CP = """ANDSKEL
3
BBOX [0 0 0] [10 10 10]
[CRITICAL POINTS]
2
3 1.0 2.0 3.0 50.0 1 0
 1
 1 0
2 4.0 5.0 6.0 10.0 0 0
 2
 0 0
 0 1
[FILAMENTS]
0
"""


def test_read_disperse_critical_points(tmp_path):
    f = tmp_path / "cp.a.NDskl"
    f.write_text(NDSKL_CP)
    cp = gtweb.read_disperse_critical_points(f)
    assert list(cp["type"]) == [3, 2]
    assert np.allclose(cp["positions"][1], [4, 5, 6]) and cp["value"][0] == 50.0

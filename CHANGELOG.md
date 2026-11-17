# Changelog

## 1.1.0 (2026-11-17)

**Filament contrast measured at the right place and along the whole spine** (`groundtruth.spine_overdensity`, the
`filament_target_contrast` / `filament_contrasts` / `filament_target_overdensity` dial). Two changes:
- the density grid is a histogram, so element i holds the value at the cell centre (i + 0.5) * cell; it was read at
  index x / cell, i.e. half a cell further along each axis (0.195 Mpc/h per axis on the default 256³ grid of a
  100 Mpc/h box). It is now read at x / cell - 0.5;
- the average was over the spine's stored vertices (6 per spine, so each node end weighed 1/6); it is now over
  equal arc-length samples of the spine line, 0.1 Mpc/h apart (new `periodic.sample_polyline`).

**Effect.** The filament network, particles, node and halo catalogues are unchanged; only the contrast dial
changes. On the 100 Mpc/h mock of Singh et al. (in preparation), the median contrast goes from 16.632 to 14.321,
per-filament new/old has median 0.895 (5-95 per cent range 0.522-1.276), the ranking is nearly unchanged
(Spearman 0.969), and 5923 instead of 5935 of the 5936 filaments are overdense. 1.1.0 reproduces that paper's
contrast values exactly (all 5936 identical in float32): the paper uses the corrected estimator.
`tests/test_groundtruth.py` checks both changes with known answers.

## 1.0.2 (2026-11-17)

**Periodic wrapping made exact** (new `gtweb/periodic.py`). Coordinates were wrapped with `x % box`, which
returns `box` itself for a tiny negative x (`-1e-17 % 100.0 == 100.0`; in float32 for x down to about -4e-6).
`cKDTree(..., boxsize=box)` rejects such a point, so `score_finder` crashed on a skeleton with a vertex just below
zero, and `np.histogramdd` put such a particle in the last cell instead of the first (`linear_velocity_field`,
`spine_overdensity`). Every coordinate wrap now goes through `periodic.wrap`, which maps that value to 0. The CIC
deposits were already correct (their cell indices are wrapped) and are unchanged.

**Effect.** None away from the edge case: a mock generated before and after (40 Mpc/h, 48³, seed 0) is
bit-identical in all 518 output arrays, so the 1.0.1 truth and particle fields are unchanged.
`tests/test_periodic.py` covers the helper, the scorer, the velocity field and the CIC;
`tests/test_groundtruth.py` covers `spine_overdensity`.

Also: the repository is a uv project (`uv.lock`, `.python-version` 3.12, a `dev` dependency group with pytest):
`uv sync` then `uv run pytest`. The pip install instructions are unchanged.

## 1.0.1 (2026-10-22)

**Skeleton builder corrected** (`gtweb/web.py`). The truth network is now the exact periodic power diagram of
the void cells: every node joins four filaments and each filament is stored once.

1.0.0 had three defects:
- edges were de-duplicated by endpoints rounded to 0.5 Mpc/h, so nodes closer than that were confused:
  some short edges were dropped, and in some geometries an edge was stored twice or ended on the wrong node;
- the periodic image shell was a fixed 0.32 of the box, which can be thinner than the largest cells,
  giving spurious edges near the box faces;
- edges longer than 0.45 of the box were dropped.

1.0.1 identifies each tetrahedron by its four (nucleus, periodic image) pairs, widens the image shell until
every node is provably exact and keeps all edges (`max_len` now defaults to no cut). The output gains `edges`
(node-index pairs) and `margin`. `tests/test_web.py` checks the result against a brute-force power-distance test.

**Effect.** On the 100 Mpc/h geometry of Singh et al. (in preparation), the image shell was wide enough, so only
the first defect applied: 1.0.1 keeps all 5842 filaments of 1.0.0 and adds 94 short ones (0.14 per cent of the
filament length). In the 50 Mpc/h notebook mock the filament count goes from 854 to 860 and all nodes now have
degree 4 (93 per cent before). The particle field is unchanged; only the truth labels differ.
Singh et al. (in preparation) use the 1.0.1 truth (README updated 2026-10-26).

Also: notebook 02 shows the filaments as plane piercings plus a projected slab.

## 1.0.0 (2026-10-20)

First release. Its skeleton builder drops some short edges (fixed in 1.0.1).

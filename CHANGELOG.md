# Changelog

## Unreleased

**Periodic wrapping made exact** (new `gtweb/periodic.py`). Coordinates were wrapped with `x % box`, which
returns `box` itself for a tiny negative x (`-1e-17 % 100.0 == 100.0`; in float32 for x down to about -4e-6).
`cKDTree(..., boxsize=box)` rejects such a point, so `score_finder` crashed on a skeleton with a vertex just below
zero, and `np.histogramdd` put such a particle in the last cell instead of the first (`linear_velocity_field`,
`spine_overdensity`). Every coordinate wrap now goes through `periodic.wrap`, which maps that value to 0. The CIC
deposits were already correct (their cell indices are wrapped) and are unchanged.

**Effect.** None away from the edge case: a mock generated before and after (40 Mpc/h, 48³, seed 0) is
bit-identical in all 518 output arrays, so the 1.0.1 truth and particle fields are unchanged.
`tests/test_periodic.py` covers the helper, the scorer, the velocity field and the CIC.

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

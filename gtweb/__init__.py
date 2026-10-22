"""
gtweb -- mock cosmic webs with a known filament network, for testing filament finders.

Make a mock:
    generate_mock          from a fresh Gaussian random field (no external data needed)
    generate_from_ic       from your own initial conditions (Zel'dovich positions and/or Psi1)
    gaussian_psi1, za_positions    the initial-condition building blocks

Read a mock:
    load_ground_truth, load_nodes, load_particles, load_halos, load_velocity_field

Score a filament finder against the ground truth:
    score_finder           completeness / purity at a spatial tolerance
    score_per_filament     completeness of every true filament (bin by contrast, length, ...)
    read_disperse_skeleton parse a DisPerSE ASCII .NDskl into polylines
    read_disperse_critical_points  its critical points (maxima, saddles, ...)
    write_disperse_pos     write particles as a DisPerSE input file

Lower-level building blocks live in the submodules gtweb.ic, gtweb.geometry, gtweb.web,
gtweb.particles, gtweb.lpt, gtweb.groundtruth, gtweb.velocity, gtweb.cosmology.
Everything is pure numpy/scipy; numba speeds up one binning step if it is installed.
"""
from .io import (load_ground_truth, load_nodes, load_particles, load_halos,
                 write_ground_truth, write_disperse_pos, read_disperse_skeleton,
                 read_disperse_critical_points, write_velocity_field, load_velocity_field, write_halos)
from .score import score_finder, score_per_filament
from .velocity import linear_velocity_field, sample_velocity
from .lpt import (load_psi1, truncated_2lpt, lpt_positions, lpt_velocity_field,
                  linear_density_from_psi1)
from .ic import gaussian_psi1, za_positions
from .particles import halo_catalogue
from .generate import generate_mock, generate_from_ic

__version__ = "1.0.1"

__all__ = [
    "generate_mock", "generate_from_ic", "gaussian_psi1", "za_positions",
    "load_ground_truth", "load_nodes", "load_particles", "load_halos",
    "write_ground_truth", "write_disperse_pos", "read_disperse_skeleton",
    "read_disperse_critical_points",
    "write_velocity_field", "load_velocity_field", "write_halos",
    "score_finder", "score_per_filament",
    "linear_velocity_field", "sample_velocity",
    "load_psi1", "truncated_2lpt", "lpt_positions", "lpt_velocity_field",
    "linear_density_from_psi1", "halo_catalogue",
]

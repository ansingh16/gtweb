"""
Top-level drivers: build a mock cosmic web with a known filament network.

    import gtweb
    # from a fresh Gaussian random field (no external data needed)
    gtweb.generate_mock("my_mock", box=100.0, ngrid=128, seed=0)
    # or from your own initial conditions
    gtweb.generate_from_ic(None, "my_mock", psi1_path="my_psi1.npz")

Chain (each step is a module of this package):
  1. ic          Zel'dovich displacement Psi1 of the initial conditions, grown to z=0
  2. geometry    void nuclei: spherical underdensities about the density minima, nested and
                 trimmed to tangency; Laguerre weights fitted so cell volume ~ void radius^3
  3. web         closed-form skeleton: nodes = tessellation vertices, filaments = cell edges
  4. particles   kinematic collapse onto the cells (void/wall/filament/node), wall and filament
                 thickness, halo model on the collapsed matter (NFW haloes)
     lpt         diffuse component: a fraction f_diffuse of the particles replaced by the 2LPT
                 realisation of the same initial conditions
  5. groundtruth spines + nodes + per-filament / per-node dials measured on the final field

The defaults are the settings of the paper's mock (a 100 Mpc/h box with 384^3 particles).
Physical scales (void search, smoothing, grids) are set in Mpc/h, so a different particle
number or box keeps the same physical construction.

Outputs written into `out_dir`:
    ground_truth_spines.npz, nodes_catalogue.npz, walls_catalogue.npz, web_geometry.npz,
    halos_catalogue.npz, mock_particles_s{seed}.npz (positions, stage, velocities),
    velocity_field.npz, metadata.json   [+ initial_conditions.npz if save_ic=True]
Deterministic for a given seed.
"""
import time
import numpy as np
from pathlib import Path

from . import geometry, web, particles, groundtruth, io, velocity, lpt, ic

# reference cell sizes of the paper's mock (100 Mpc/h box), in Mpc/h
_CIC_CELL = 100.0 / 128      # IC density grid for the void search
_FIT_CELL = 100.0 / 112      # grid of the Laguerre volume fit
_HALO_CELL = 100.0 / 512     # peak-finding grid of the halo model


def _grid_for(box, cell, lo=32, hi=1024):
    return int(np.clip(round(box / cell), lo, hi))


def _run(ic_pos, box, out_dir, psi1=None, seed=0, grid=None, rho_v=0.2, smooth=1.5,
         r_min=1.5, r_max=None, fit_grid=None, fit_iters=260, T_base=0.98, T_exp=0.5, scatter=0.10,
         sigma_wall=0.25, sigma_fil=0.25, halos=True, diffuse_frac=0.4, halo_grid=None,
         diffuse='2lpt', Rs=0.0, f_diffuse=0.32, velocities=True, vel_ngrid=128,
         source='', verbose=True):
    t0 = time.time()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    N = len(ic_pos)
    grid = grid or _grid_for(box, _CIC_CELL)
    fit_grid = fit_grid or _grid_for(box, _FIT_CELL)
    halo_grid = halo_grid or _grid_for(box, _HALO_CELL)
    if r_max is None:
        r_max = 0.22 * box
    use_2lpt = str(diffuse).lower() == '2lpt'
    if use_2lpt and psi1 is None:
        raise ValueError("diffuse='2lpt' needs the Psi1 displacement field of the initial conditions "
                         "(pass psi1_path), or use diffuse='none'")

    def log(*a):
        if verbose:
            print(*a, flush=True)

    log(f"[gtweb] {N:,} particles, box {box:g} Mpc/h, IC density grid {grid}^3")

    # 2: geometry (IC -> nuclei, weights, radii)
    log("[gtweb] void nuclei + Laguerre weight fit ...")
    geo = geometry.build_geometry(ic_pos, box, grid=grid, rho_v=rho_v, smooth=smooth,
                                  r_min=r_min, r_max=r_max, fit_grid=fit_grid, fit_iters=fit_iters,
                                  verbose=verbose)
    log(f"        {len(geo['nuclei'])} voids, radii P10/50/90 "
        f"{np.percentile(geo['void_radii'], [10, 50, 90]).round(2)} Mpc/h  [{time.time()-t0:.0f}s]")

    # 3: analytic web
    log("[gtweb] closed-form skeleton ...")
    w = web.build_web_from_geometry(geo['nuclei'], geo['nuclei_weights'], box)
    log(f"        {len(w['nodes'])} nodes, {int(w['n_filaments'])} filaments  [{time.time()-t0:.0f}s]")

    # 4: collapse onto the cells
    log("[gtweb] collapsing the particles onto the cells ...")
    pos, stage = particles.cascade_onto_geometry(ic_pos, geo['nuclei'], geo['nuclei_weights'],
                                                 geo['void_radii'], box, seed=seed, T_base=T_base,
                                                 T_exp=T_exp, scatter=scatter, verbose=verbose)
    if sigma_wall > 0 or sigma_fil > 0:
        rng_t = np.random.default_rng(2000 + seed)
        extra = np.zeros(len(pos)); extra[stage == 1] = sigma_wall; extra[stage == 2] = sigma_fil
        pos = (pos + rng_t.normal(0.0, 1.0, pos.shape) * extra[:, None]) % box

    n_halos = None
    if halos:
        log("[gtweb] halo model on the collapsed matter ...")
        pos, stage, halo_cat = particles.halo_model_population(
            pos, stage, box, seed=seed, grid=halo_grid, diffuse_frac=diffuse_frac, verbose=verbose,
            return_catalogue=True)
        io.write_halos(out_dir / "halos_catalogue.npz", halo_cat)
        n_halos = int(len(halo_cat['positions']))
        log(f"        {n_halos:,} haloes (>= {int(halo_cat['min_npart'])} particles)")

    # 4b: diffuse component from the 2LPT realisation of the same initial conditions
    lpt_fields = None
    if use_2lpt:
        log("[gtweb] 2LPT diffuse component ...")
        p1s, p2, r1, r2 = lpt.truncated_2lpt(psi1, box, Rs=Rs)
        lpt_fields = (p1s, p2)
        pic = lpt.lpt_positions(p1s, p2, box)
        Nmix = len(pos); nd = int(round(float(f_diffuse) * Nmix))
        rng_m = np.random.default_rng(3000 + seed)
        keep = rng_m.choice(len(pos), Nmix - nd, replace=False)
        take = rng_m.choice(len(pic), nd, replace=False)
        pos = np.vstack([pos[keep], pic[take]])
        stage = np.concatenate([stage[keep], np.zeros(nd, np.uint8)]).astype(np.uint8)
        log(f"        f_diffuse={f_diffuse:g} ({nd:,} particles), rms|Psi2|/|Psi1| = {r2/max(r1,1e-9):.3f}")
    log(f"        {len(pos):,} particles  [{time.time()-t0:.0f}s]")

    # 5: ground truth + dials on the final field
    log("[gtweb] ground truth + dials ...")
    gt, nodes_cat, walls_cat, web_geom = groundtruth.assemble_ground_truth(
        w, geo['nuclei'], geo['nuclei_weights'], pos, box)

    part_extra = {}
    if velocities:
        if use_2lpt:
            vf = lpt.lpt_velocity_field(lpt_fields[0], lpt_fields[1], box, ngrid=vel_ngrid)
        else:
            vf = velocity.linear_velocity_field(pos, box, ngrid=vel_ngrid)
        io.write_velocity_field(out_dir / "velocity_field.npz", vf)
        part_extra['velocities'] = velocity.sample_velocity(vf, pos, box).astype(np.float32)

    np.savez(out_dir / f"mock_particles_s{seed}.npz",
             positions=pos.astype(np.float32), box_size=np.float64(box),
             n_bar=np.float64(len(pos) / box ** 3), seed=np.int32(seed), stage=stage.astype(np.uint8),
             **part_extra)

    metadata = dict(
        source=str(source), box_size=box, n_particles=int(len(pos)), seed=int(seed),
        ic_grid=int(grid), fit_grid=int(fit_grid), halo_grid=int(halo_grid),
        rho_v=rho_v, smooth=smooth, r_min=r_min, r_max=float(r_max),
        T_base=T_base, T_exp=T_exp, scatter=scatter, sigma_wall=sigma_wall, sigma_fil=sigma_fil,
        halos=bool(halos), diffuse_frac=float(diffuse_frac), n_halos=n_halos,
        diffuse=str(diffuse), Rs=float(Rs) if use_2lpt else None,
        f_diffuse=float(f_diffuse) if use_2lpt else None,
        velocity_order=('2lpt' if use_2lpt else 'linear') if velocities else None,
        n_voids=int(len(geo['nuclei'])), n_nodes=int(len(w['nodes'])), n_filaments=int(w['n_filaments']),
        generator="gtweb", units="Mpc/h, Msun/h, km/s; periodic box",
    )
    io.write_ground_truth(out_dir, gt, nodes_cat, walls_cat, web_geom, metadata=metadata)
    log(f"[gtweb] done -> {out_dir}  [{time.time()-t0:.0f}s]")
    return metadata


def generate_mock(out_dir, box=100.0, ngrid=128, seed=0, save_ic=False, **kw):
    """Generate a mock web from a fresh Gaussian random field (LCDM, Planck cosmology).

    box    : periodic box side [Mpc/h]
    ngrid  : particles per side (ngrid^3 particles; the paper used 384 in a 100 Mpc/h box)
    seed   : sets the initial phases and every random step; the mock is deterministic per seed
    save_ic: also write initial_conditions.npz (Psi1, for a phase-matched N-body run)
    **kw   : any setting of the construction (see generate_from_ic)
    Returns the metadata dict (also written to out_dir/metadata.json).
    """
    d = ic.gaussian_psi1(box, ngrid, seed=seed)
    if save_ic:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(out_dir) / "initial_conditions.npz", **d)
    ic_pos = ic.za_positions(d['psi1'], box)
    return _run(ic_pos, float(box), out_dir, psi1=d['psi1'], seed=seed,
                source=f"gaussian_psi1(box={box}, ngrid={ngrid}, seed={seed})", **kw)


def generate_from_ic(ic_path, out_dir, psi1_path=None, seed=0, **kw):
    """Generate a mock web from your own initial conditions.

    ic_path   : .npz with `positions` (N,3) of the initial-condition particles grown to z=0 at
                linear amplitude (Zel'dovich), and `box_size`.  Set ic_path=None to build them
                from psi1_path.
    psi1_path : .npz with `psi1` (3,n,n,n) [Mpc/h] and `box_size` -- the Zel'dovich displacement
                on the Lagrangian grid (cell centres).  Needed for the 2LPT diffuse component.
    Settings (defaults = the paper's mock):
        grid, fit_grid, halo_grid  grids (default: set from the box, same cell sizes as the paper)
        rho_v=0.2       void barrier: mean enclosed density of a void sphere / mean density
        smooth=1.5      Gaussian smoothing [Mpc/h] of the IC density before the minima search
        r_min=1.5, r_max=0.22*box   void-radius range [Mpc/h]
        fit_iters=260   iterations of the Laguerre volume fit
        T_base=0.98, T_exp=0.5      collapse expansion T_i = T_base (R_i / median R)^T_exp
        scatter=0.10    isotropic scatter after the collapse [Mpc/h]
        sigma_wall=0.25, sigma_fil=0.25   Gaussian wall / filament thickness [Mpc/h]
        halos=True, diffuse_frac=0.4      halo model; fraction of filament matter left as bridges
        diffuse='2lpt' | 'none', f_diffuse=0.32, Rs=0.0   diffuse component (Rs: truncation scale)
        velocities=True, vel_ngrid=128    velocity cube + per-particle velocities
    """
    psi1 = None
    box = None
    if psi1_path is not None:
        d1 = lpt.load_psi1(psi1_path)
        psi1, box = d1['psi1'], float(d1['box_size'])
    if ic_path is not None:
        d = np.load(ic_path, allow_pickle=True)
        ic_pos = d['positions'].astype(np.float64)
        b = float(d['box_size'])
        if box is not None and abs(b - box) > 1e-6:
            raise ValueError(f"IC box {b} != Psi1 box {box}")
        box = b
    elif psi1 is not None:
        ic_pos = ic.za_positions(psi1, box)
    else:
        raise ValueError("give ic_path, psi1_path, or both")
    if psi1 is None and str(kw.get('diffuse', '2lpt')).lower() == '2lpt':
        raise ValueError("diffuse='2lpt' needs psi1_path; pass diffuse='none' to skip the diffuse component")
    return _run(ic_pos, box, out_dir, psi1=psi1, seed=seed, source=str(ic_path or psi1_path), **kw)

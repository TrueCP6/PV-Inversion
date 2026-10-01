"""Timestepped run diagnostics: how the flow evolves, and whether the CFL limit is where
prognostic_solver.dt() says it is.
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
import sweep
from parameters import PhysicalParams

DAY = 24 * 3600

# A run is called unstable once max|q| passes this multiple of its initial value. Growth
# that large is never physical here, and stopping there keeps a diverging scan point from
# running to the end in floating point garbage.
BLOWUP_FACTOR = 1e3

# A cfl scan point is called unstable once its roughness passes this multiple of the lowest
# Courant number's at the same time. An absolute cut can't tell grid-scale noise from the flow
# sharpening q: cfl_scan_3 crossed 7x its initial roughness at ~16 h at every Courant number
# from 1.3 to 2.3, all within 1% of each other. Sharpening is the same at every step size,
# noise from the step is not, so the ratio sees only the noise.
ROUGHNESS_RATIO = 1.5

# run() rewrites its whole results file each save, so saving every step costs I/O that grows
# with the run; every SAVE_EVERY steps loses at most that many on an interruption.
SAVE_EVERY = 10

# The panels of the diagnostics figure, in order, as (field, y label, scientific y ticks).
# Labelled as variator.py labels the same quantities, so the two figures read alike.
DIAGNOSTIC_PANELS = [
    ("trop_height", r"$z_\text{dyn}$ [\unit{\meter}]", False),
    ("min_pressure", r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]", False),
    ("max_wind", r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]", False),
    ("min_vort", r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]", True),
    ("dt", r"$\Delta t$ [\unit{\second}]", False),
]

ACCENT = '#004488'

# The timestepped run's domain: the anomaly starts well clear of the inflow wall, and Lx is
# stretched in run() to hold everything the jet carries it through
ANOMALY_X = 2.5e6
LY = 5e6
NY, NZ = 40, 50 # doubled at p=2

class _TimesteppingParams(PhysicalParams):
    @property
    def anomaly_x_pos(self):
        return ANOMALY_X

def _phys_params(Lx=PhysicalParams.Lx, overrides=None):
    overrides = dict(overrides or {})
    offset = overrides.pop("jet_y_pos", PhysicalParams.jet_y_pos) - PhysicalParams().anomaly_y_pos
    return _TimesteppingParams(**overrides, Lx=Lx, Ly=LY, jet_y_pos=LY / 2 + offset) # anomaly_y_pos is Ly/2 too

def _domain(T, p, jet_magnitude=PhysicalParams.jet_magnitude):
    """(nx, ny, nz, Lx) for a run to T at order p, with dx = dy.

    Lx leaves room for the anomaly to be carried T downstream at the jet's peak speed - an
    overestimate, since it steers with the flow averaged over its extent and propagates
    upstream against it - with the same clearance ahead of it as it starts with behind.
    """
    scale = 4.0/p
    ny, nz = int(NY * scale), int(NZ * scale)
    dx = LY / ny
    nx = int(np.ceil((2 * ANOMALY_X + jet_magnitude * T) / dx))
    return nx, ny, nz, nx * dx

@dataclass
class StepRecord:
    """One timestep's diagnostics. t is the time at the end of the step."""
    t : float
    dt : float
    mass : float
    enstrophy : float
    q_min : float
    q_max : float
    trop_height : float
    max_wind : float
    min_vort : float
    min_pressure : float
    # Along the jet axis at the points _hovmoller_x gives: Ertel PV [PVU] at the anomaly's height,
    # and the surface pressure anomaly [hPa]. None in results files written before they were added.
    epv_line : list = None
    pressure_line : list = None
    q_line : list = None # q - q_bar, in files written before epv_line replaced it
    # The minimum of the vorticity anomaly over each z level, lowest first, and those levels'
    # heights - which never change, so only the first record carries them.
    vort_profile : list = None
    level_heights : list = None
    Lx : float = None # first record only; None in files from runs on the default domain

@dataclass
class CflRecord:
    """One Courant number's outcome, over the same number of steps or the same simulated
    time as every other.

    steps is how many it actually took, which is fewer than asked for when it diverged.
    """
    courant : float
    steps : int
    growth : float
    stable : bool
    # Per step, or None in files written before they were added: the time the step ended at,
    # max|q|, ||q||_L2 and the roughness (see _roughness_meter) there as multiples of their
    # initial values, and the fraction of the domain the limiter touched (None in files from
    # scans that could run without it).
    t : list = None
    max_growth : list = None
    l2_growth : list = None
    limited : list = None
    roughness : list = None

def _hovmoller_x(Lx, count):
    """Cell-centred points along x - clear of the lateral walls, where a point can fall outside."""
    return (np.arange(count) + 0.5) * Lx / count

class _HovmollerSampling:
    """What the Hovmoller diagrams sample, built once per run: point meshes along the jet
    axis, y = anomaly_y_pos - one at the anomaly's height through the 3D mesh, one on the
    surface mesh - and the backgrounds the anomalies are taken against.
    """
    def __init__(self, solver):
        from firedrake import Function, VertexOnlyMesh
        from derived_quantities import ResolvedAtmosphere

        p = solver.atmos.phys_params
        s = solver.atmos.solver_params
        x = _hovmoller_x(p.Lx, s.nx * s.polynomial_order) # one point per node spacing
        mesh = solver.q.function_space().mesh()

        self.upper = VertexOnlyMesh(mesh, [[xi, p.anomaly_y_pos, p.anomaly_z_pos] for xi in x])
        self.surface = VertexOnlyMesh(mesh._base_mesh, [[xi, p.anomaly_y_pos] for xi in x])
        # The jet's own shear vorticity is larger than the anomaly's, so the profile is taken
        # against it. solver.psi is only there for its function space here.
        self.zeta_bar = Function(solver.psi.function_space()).interpolate(
            solver.atmos.geostrophic_vorticity())
        self.heights = ResolvedAtmosphere(solver.psi, solver.atmos).level_heights().tolist()

    @staticmethod
    def sample(f, vom):
        """f at vom's points, in the order they were given, as a list on the main rank
        (empty elsewhere, which is fine: only the main rank writes the records).
        """
        from firedrake import Function, FunctionSpace

        at_points = Function(FunctionSpace(vom, "DG", 0)).interpolate(f)
        ordered = Function(FunctionSpace(vom.input_ordering, "DG", 0)).interpolate(at_points)
        return ordered.dat.data_ro.tolist()

def _diagnostics(solver, lines):
    """Every value the figures need, from the state the solver is currently on."""
    from firedrake import assemble, dx
    from derived_quantities import ResolvedAtmosphere
    from math_utils import get_global_extrema
    from tropopause import PVU

    solver.resolve()
    derived = ResolvedAtmosphere(solver.psi, solver.atmos, solver.q)
    q_min, q_max = get_global_extrema(solver.q)

    return dict(
        mass=assemble(solver.q * dx),
        enstrophy=assemble(solver.q ** 2 * dx),
        q_min=q_min,
        q_max=q_max,
        trop_height=derived.min_dyn_tropopause_height(),
        max_wind=derived.max_surf_wind_speed(),
        min_vort=derived.min_surf_vort(),
        min_pressure=derived.min_surf_pressure_ano_hpa(),
        epv_line=lines.sample(solver.atmos.ertel_from_qgpv(solver.q) / PVU, lines.upper),
        pressure_line=lines.sample(derived.surf_pressure_ano_hpa(), lines.surface),
        vort_profile=derived.min_per_level(
            derived.geostrophic_vorticity() - lines.zeta_bar).tolist(),
    )

def _max_abs(diagnostics):
    """Largest |q| in a diagnostics dict - the blow-up detector."""
    return max(abs(diagnostics["q_min"]), abs(diagnostics["q_max"]))

def _overshoot(records):
    """How far q has strayed outside its initial range, as a fraction of that range.
    """
    q_min = np.array([r.q_min for r in records])
    q_max = np.array([r.q_max for r in records])
    beyond = np.maximum(np.maximum(q_max - q_max[0], q_min[0] - q_min), 0.0)
    return beyond / (q_max[0] - q_min[0])

def _format_duration(seconds):
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(int(seconds), 0)  # the step that lands past T leaves a negative estimate
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

def save_checkpoint(path, solver):
    """psi and q as they stand, with the mesh they live on and every parameter
    BarnesAtmosphere is built from, so load_checkpoint can rebuild the whole state.
    """
    from firedrake import CheckpointFile

    atmos = solver.atmos
    with CheckpointFile(path, 'w') as checkpoint:
        checkpoint.save_mesh(atmos.mesh)
        checkpoint.save_function(solver.psi, name="psi")
        checkpoint.save_function(solver.q, name="q")
        checkpoint.set_attr("/", "mesh_name", atmos.mesh.name)
        checkpoint.set_attr("/", "t", solver.t)
        checkpoint.set_attr("/", "phys_params", json.dumps(asdict(atmos.phys_params)))
        checkpoint.set_attr("/", "solver_params", json.dumps(asdict(atmos.solver_params)))

def load_checkpoint(path):
    """What save_checkpoint wrote, as (atmosphere, psi, q, t). The atmosphere is built on the
    checkpoint's own mesh, so psi and q sit on its spaces with no interpolation between meshes.
    """
    from firedrake import CheckpointFile
    from barnes_atmosphere import BarnesAtmosphere
    from domain_builder import DomainBuilder
    from parameters import SolverParams

    with CheckpointFile(path, 'r') as checkpoint:
        mesh = checkpoint.load_mesh(checkpoint.get_attr("/", "mesh_name"))
        psi = checkpoint.load_function(mesh, "psi")
        q = checkpoint.load_function(mesh, "q")
        t = checkpoint.get_attr("/", "t")
        phys_params = PhysicalParams(**json.loads(checkpoint.get_attr("/", "phys_params")))
        solver_params = SolverParams(**json.loads(checkpoint.get_attr("/", "solver_params")))

    atmos = BarnesAtmosphere(DomainBuilder(solver_params, phys_params, mesh))
    return atmos, psi, q, t

def _build_solver(nx, ny, nz, p, phys_params=None):
    from parameters import SolverParams
    from prognostic_solver import PrognosticSolver

    solver_params = SolverParams(nx=nx, ny=ny, nz=nz, polynomial_order=p, check_flux=False)
    return PrognosticSolver(solver_params, phys_params or PhysicalParams(), matfree=True)

def run(out_path, T=DAY, p=None, courant=None, backup=24, params=None):
    from firedrake.petsc import PETSc
    from prognostic_solver import SAFETY
    from parameters import SolverParams

    sweep.quiet_petsc()
    p = p if p is not None else SolverParams.polynomial_order
    courant = courant if courant is not None else SAFETY
    nx, ny, nz, Lx = _domain(T, p, _phys_params(overrides=params).jet_magnitude)

    solver = _build_solver(nx, ny, nz, p, _phys_params(Lx, params))
    lines = _HovmollerSampling(solver)
    initial = _diagnostics(solver, lines)
    records = [StepRecord(t=0.0, dt=float('nan'), level_heights=lines.heights, Lx=Lx, **initial)]

    PETSc.Sys.Print(f"Stepping to {T / 3600:.1f} h on {nx}x{ny}x{nz} cells, Lx={Lx / 1e3:.0f} km, "
                    f"p={p}, Courant={courant}")
    started = time.perf_counter()
    next_backup = backup * 3600

    while solver.t < T:
        dt = solver.step(solver.dt(safety=courant))
        diagnostics = _diagnostics(solver, lines)
        records.append(StepRecord(t=solver.t, dt=dt, **diagnostics))

        if sweep.is_main_rank() and (len(records) - 1) % SAVE_EVERY == 0:
            sweep.save_records(out_path, records)

        if backup and solver.t >= next_backup:
            # _diagnostics just resolved psi, so it matches q
            save_checkpoint(f"{out_path.removesuffix('.json')}_t{next_backup / 3600:g}h.h5", solver)
            next_backup = (np.floor(solver.t / (backup * 3600)) + 1) * backup * 3600

        elapsed = time.perf_counter() - started
        remaining = elapsed * (T - solver.t) / solver.t  # the clock tracks simulated time, not steps
        PETSc.Sys.Print(
            f"step {len(records) - 1:4d}  t = {solver.t / 3600:6.2f}/{T / 3600:.1f} h"
            f"  dt = {dt:7.1f} s  elapsed {_format_duration(elapsed)}"
            f"  remaining ~{_format_duration(remaining)}"
        )

        growth = _max_abs(diagnostics) / _max_abs(initial)
        if not np.isfinite(growth) or growth > BLOWUP_FACTOR:
            PETSc.Sys.Print(f"max|q| grew {growth:.3g}x - stopping, the run has gone unstable")
            break

    if sweep.is_main_rank():
        sweep.save_records(out_path, records)

    PETSc.Sys.Print(f"Done in {_format_duration(time.perf_counter() - started)}, "
                    f"{len(records) - 1} steps, wrote {out_path}")
    return records

def _roughness_meter(q, p):
    """A function giving the largest RMS over cells of q's horizontally degree-p part - what
    interpolating onto one degree lower in x and y misses. Grid-scale noise lives there; a
    smooth q has little, however sharp it is in z.
    """
    from firedrake import Function, FunctionSpace, TestFunction, assemble, dx, interval, quadrilateral
    from finat.ufl import FiniteElement, TensorProductElement
    from math_utils import get_global_max

    mesh = q.function_space().mesh()
    lowered = Function(FunctionSpace(mesh, TensorProductElement(
        FiniteElement("DQ", quadrilateral, p - 1), FiniteElement("DG", interval, p))))
    W0 = FunctionSpace(mesh, "DQ", 0)
    w = TestFunction(W0)
    volume = assemble(w * dx).dat.data_ro.copy()
    per_cell = Function(W0)

    def roughness():
        lowered.interpolate(q)
        per_cell.dat.data_wo[:] = assemble((q - lowered) ** 2 * w * dx).dat.data_ro / volume
        return np.sqrt(get_global_max(per_cell))
    return roughness

def cfl_scan(out_path, courants, steps=60, n=12, p=None, T=None):
    from firedrake import norm
    from firedrake.petsc import PETSc
    from math_utils import get_global_extrema
    from parameters import SolverParams

    sweep.quiet_petsc()
    p = p if p is not None else SolverParams.polynomial_order

    def max_abs_q(solver):
        low, high = get_global_extrema(solver.q)
        return max(abs(low), abs(high))

    records = []
    for courant in courants:
        solver = _build_solver(n, n, n, p)
        solver.resolve()
        initial, initial_l2 = max_abs_q(solver), norm(solver.q)
        roughness_of = _roughness_meter(solver.q, p)
        initial_roughness = roughness_of()

        growth, taken = 1.0, 0
        t, max_growth, l2_growth, limited, roughness = [], [], [], [], []
        while (solver.t < T) if T is not None else (taken < steps):
            dt = solver.dt(safety=courant)
            if T is not None:
                dt = min(dt, T - solver.t) # land exactly on T
            solver.step(dt)
            taken += 1
            solver.resolve()
            growth = max_abs_q(solver) / initial

            t.append(solver.t)
            max_growth.append(float(growth))
            l2_growth.append(norm(solver.q) / initial_l2)
            limited.append(solver.limiter.limited_fraction)
            roughness.append(roughness_of() / initial_roughness)

            if not np.isfinite(growth) or growth > BLOWUP_FACTOR:
                break

        # Only blow up here - the roughness verdict needs the other Courant numbers (roughness_ratios)
        stable = bool(np.isfinite(growth) and growth <= BLOWUP_FACTOR)
        records.append(CflRecord(courant=float(courant), steps=taken,
                                 growth=float(growth) if np.isfinite(growth) else BLOWUP_FACTOR,
                                 stable=stable, t=t, max_growth=max_growth, l2_growth=l2_growth,
                                 limited=limited, roughness=roughness))
        PETSc.Sys.Print(f"Courant {courant:.2f}: {taken:4d} steps to t = {solver.t / 3600:.2f} h, "
                        f"max|q| grew {growth:.4g}x, ||q||_L2 {l2_growth[-1]:.4g}x, "
                        f"roughness {roughness[-1]:.4g}x (peak {max(roughness):.4g}x), "
                        f"{'stable' if stable else 'UNSTABLE'}")

        if sweep.is_main_rank():
            sweep.save_records(out_path, records)

    return records

def _stacked_panels(records, panels, output_path, height=1.5):
    """One quantity per panel, stacked on a shared time axis.

    A panel each rather than one axes carrying several scales: these quantities differ in
    both unit and magnitude, and a second y-axis would put a crossing point on the figure
    that is an artefact of the two scalings rather than anything in the flow.
    """
    import matplotlib.pyplot as plt
    import plot_utils

    plot_utils.apply_style()
    hours = np.array([r.t for r in records]) / 3600

    fig, axes = plt.subplots(len(panels), 1, sharex=True,
                             figsize=(plot_utils.FIGURE_SIZE[0], height * len(panels)))

    for ax, (values, y_label, scientific) in zip(axes, panels):
        ax.plot(hours, values, color=ACCENT, linewidth=1.2)
        ax.set_ylabel(y_label)
        ax.set_xlim(hours[0], hours[-1])
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        if scientific:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    axes[-1].set_xlabel(r"$t$ [\unit{\hour}]")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def _run_params(records):
    """The parameters a results file was run with - the default domain if it predates Lx."""
    return PhysicalParams() if records[0].Lx is None else _phys_params(records[0].Lx)

def plot_conservation(json_path, output_path="tex/plots/timestepping_conservation.pdf"):
    """Drift of the domain integrals of q and q^2, and q's overshoot, against time.

    Neither integral is an exact invariant - the lateral boundaries carry q in and out - so
    a smooth bounded drift is the expected picture. What the figure is for is the other
    case: a knee or a run away in any panel is the scheme coming apart. The third panel is
    the sharper of the three, since it is zero for any scheme behaving itself.

    The first panel is scaled by the size of q rather than by its own initial value. q
    changes sign across the domain and very nearly cancels, so its mean comes out around a
    thousandth of its rms, and dividing by that mean magnifies the drift by the same factor
    - it reads as hundreds of percent where the field has moved by a fraction of one. Over
    rms(q) * V instead, the panel says how far the mean of q has moved in units of a
    typical q, which is the number that decides whether the run is still meaningful.
    Enstrophy is positive definite and needs no such care, so it stays a relative drift.
    """
    if not sweep.is_main_rank():
        return

    records = sweep.load_records(json_path, StepRecord)

    def change(field):
        values = np.array([getattr(r, field) for r in records])
        return values - values[0]

    volume = _run_params(records).domain_volume
    q_rms = np.sqrt(records[0].enstrophy / volume)

    _stacked_panels(records, [
        (change("mass") / (q_rms * volume),
         r"$\Delta \overline{q} \,/\, q_\text{rms}$", True),
        (change("enstrophy") / abs(records[0].enstrophy),
         r"Drift in $\int q^2 \,\mathrm{d}V$", True),
        (_overshoot(records), r"$q$ beyond its initial range", True),
    ], output_path, height=1.7)

def plot_diagnostics(json_path, output_path="tex/plots/timestepping_diagnostics.pdf"):
    """The resolved-atmosphere diagnostics against time, one panel each."""
    if not sweep.is_main_rank():
        return

    records = sweep.load_records(json_path, StepRecord)

    _stacked_panels(records, [
        (np.array([getattr(r, field) for r in records]), y_label, scientific)
        for field, y_label, scientific in DIAGNOSTIC_PANELS
    ], output_path)

def plot_hovmoller(json_path, output_path="tex/plots/timestepping_hovmoller.pdf"):
    """Hovmoller diagrams along the jet axis: the upper Ertel PV beside the surface low.

    The two share their axes, so a system's track reads as a diagonal whose slope is its
    phase speed, and any horizontal offset between the tracks is the tilt between the
    upper anomaly and the surface low it induces.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import CenteredNorm, TwoSlopeNorm
    import plot_utils

    if not sweep.is_main_rank():
        return

    records = sweep.load_records(json_path, StepRecord)
    if records[0].epv_line is None:
        print(f"{json_path} predates the jet axis EPV line - no Hovmoller diagram")
        return
    plot_utils.apply_style()

    hours = np.array([r.t for r in records]) / 3600
    panels = [
        (np.array([r.epv_line for r in records]), r"$Q$ [PVU]", r"$z = z_\text{trop}$", False),
        (np.array([r.pressure_line for r in records]), r"$p^*$ [\unit{\hecto\pascal}]",
         r"$z = 0$", True),
    ]
    x_km = _hovmoller_x(_run_params(records).Lx, panels[0][0].shape[1]) / 1e3

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(plot_utils.FIGURE_SIZE[0], 4.5))
    for ax, (values, label, title, centered) in zip(axes, panels):
        # Percentiles, not the extremes, set the scale: one grid-scale spike would otherwise
        # wash every coherent feature out to white
        if centered:
            half_range = np.nanpercentile(np.abs(values), 99.5)
            norm, cmap = CenteredNorm(halfrange=half_range), 'RdBu_r'
            # No zero contour - it would trace every sign flip of grid-scale noise
            levels = np.delete(np.linspace(-half_range, half_range, 9), 4)
        else:
            # Physical EPV has the sign of f; the inversion can leave nonphysical pockets of the
            # other sign, which take the opposite colour. The physical side is scaled by a
            # percentile as above, the wrong side by its true extreme so that no pocket is
            # clipped away however small it is.
            sign = np.sign(PhysicalParams().f)
            physical = sign * np.nanpercentile(sign * values, 99.5)
            wrong = -sign * max(np.nanmax(-sign * values), 1e-3 * abs(physical))
            norm, cmap = TwoSlopeNorm(0, *sorted([physical, wrong])), 'RdBu_r'
            levels = np.linspace(physical, 0, 9)[1:-1]
            # The dynamical tropopause, 1.5 PVU with the sign of f, and the edge of any
            # nonphysical pocket
            ax.contour(x_km, hours, values, levels=sorted([sign * 1.5, 0]),
                       colors='k', linewidths=1.2, linestyles='solid')
        mesh = ax.pcolormesh(x_km, hours, values, cmap=cmap, norm=norm,
                             shading='nearest', rasterized=True)
        ax.contour(x_km, hours, values, levels=sorted(levels), colors='k', linewidths=0.4)
        extend = 'both' if centered else ('min' if physical < 0 else 'max')
        colorbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.12, extend=extend)
        # TwoSlopeNorm would give each sign half the bar - linear sizes each by its range
        colorbar.ax.set_xscale('linear')
        colorbar.set_label(label)
        ax.set_title(title)
        ax.set_xlabel(r"$x$ [\unit{\kilo\meter}]")

    axes[0].set_ylabel(r"$t$ [\unit{\hour}]")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_height_time(json_path, output_path="tex/plots/timestepping_height_time.pdf"):
    """The strongest cyclonic vorticity anomaly at each height against time, with the lowest
    point of the dynamical tropopause over it.

    A minimum over each level rather than a column through a fixed point, so nothing has to
    track the anomaly as it moves: what the figure shows is how far down the circulation it
    induces reaches, and how that changes as the tropopause is pulled down over it.
    """
    import matplotlib.pyplot as plt
    import plot_utils

    if not sweep.is_main_rank():
        return

    records = sweep.load_records(json_path, StepRecord)
    if records[0].vort_profile is None:
        print(f"{json_path} predates the vorticity profile - no height-time diagram")
        return
    plot_utils.apply_style()

    hours = np.array([r.t for r in records]) / 3600
    z_km = np.array(records[0].level_heights) / 1e3
    values = np.array([r.vort_profile for r in records]).T
    # A minimum is never positive, so a sequential map down from zero. As in plot_hovmoller,
    # a percentile sets the far end, so one spike can't wash the rest out.
    strongest = -np.nanpercentile(np.abs(values), 99.5)

    fig, ax = plt.subplots(figsize=(plot_utils.FIGURE_SIZE[0], 3.5))
    mesh = ax.pcolormesh(hours, z_km, values, cmap='Blues_r', vmin=strongest, vmax=0,
                         shading='nearest', rasterized=True)
    levels = np.linspace(strongest, 0, 5)[:-1]
    ax.contour(hours, z_km, values, levels=levels, colors='k', linewidths=0.4)
    ax.plot(hours, np.array([r.trop_height for r in records]) / 1e3, color='k',
            linestyle='--', linewidth=1.2, label=r"$z_\text{dyn}$")

    colorbar = fig.colorbar(mesh, ax=ax, extend='min')
    colorbar.set_label(r"$\min_{x,y}\, (\zeta_g - \bar{\zeta}_g)$ [\unit{\per\second}]")
    colorbar.formatter.set_powerlimits((0, 0))
    ax.set_xlabel(r"$t$ [\unit{\hour}]")
    ax.set_ylabel(r"$z$ [\unit{\kilo\meter}]")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def roughness_ratios(records):
    """Each record's peak roughness as a multiple of the lowest Courant number's at the same
    time, over the time both reached. The lowest is the reference, so it reads 1.
    """
    reference = min(records, key=lambda r: r.courant)
    peaks = []
    for r in records:
        t, roughness = np.array(r.t), np.array(r.roughness)
        shared = t <= reference.t[-1]
        peaks.append(float(np.max(roughness[shared] / np.interp(t[shared], reference.t, reference.roughness))))
    return peaks

def plot_cfl(json_paths, output_path="tex/plots/cfl_verification.pdf"):
    """Growth of max|q| against Courant number, either side of the predicted limit. Prints each
    point's roughness verdict too, which needs the whole scan - one file per Courant number
    from slurm_jobs/cfl_scan.sh, so this takes several.
    """
    import matplotlib.pyplot as plt
    import plot_utils

    if not sweep.is_main_rank():
        return

    records = sorted((r for path in json_paths for r in sweep.load_records(path, CflRecord)),
                     key=lambda r: r.courant)
    # cfl_scan and cfl_scan_2 predate the roughness, and get the plot alone
    for r, ratio in zip(records, roughness_ratios(records) if records[0].roughness else []):
        print(f"Courant {r.courant:.2f}: roughness peaks at {ratio:.3g}x the lowest's, "
              f"{'stable' if r.stable and ratio <= ROUGHNESS_RATIO else 'UNSTABLE'}")
    plot_utils.apply_style()

    courant = np.array([r.courant for r in records])
    growth = np.array([r.growth for r in records])

    plt.figure(figsize=plot_utils.SQUARE_HALF_FIGURE_SIZE)
    limit = 1.9 # 1D von Neumann, upwind DG + SSPRK3 on Gauss-Legendre nodes (see cfl_scan)
    plt.axvspan(limit, max(courant.max(), limit), color='0.9', zorder=0)
    plt.axvline(limit, color='0.4', linestyle='--', linewidth=1.2,
                label=rf'Predicted limit, $\mathrm{{C}}={limit}$')
    plt.semilogy(courant, growth, color=ACCENT, marker='o', markersize=4, linewidth=1.5,
                 label=r'$\max|q| / \max|q|_{t=0}$')

    plt.xlabel(r'Courant number $\mathrm{C} = \Delta t\, \max(|u| + |v|) / \Delta x_\text{eff}$')
    plt.ylabel(r'Growth of $\max|q|$')
    plot_utils.finish_figure(output_path, legend_kwargs={'loc': 'upper left', 'fontsize': 7})

def main():
    parser = argparse.ArgumentParser(description='Run and plot a timestepped simulation')
    parser.add_argument('-T', '--hours', type=float, default=24, help='Simulated hours to run for')
    parser.add_argument('-p', '--polynomial_order', type=int, default=None)
    parser.add_argument('-c', '--courant', type=float, default=None,
                        help='Multiple of dx_eff / max(|u| + |v|) to step at (default: prognostic_solver.SAFETY)')
    parser.add_argument('--backup', type=float, default=24, metavar='H',
                        help='Checkpoint psi, q and the parameters every H simulated hours (0: never)')
    parser.add_argument('--params', metavar='JSON_PATH',
                        help='PhysicalParams overrides, e.g. param_sampler.py\'s optimised_*.json, '
                             'suffixing the results file names with the JSON\'s')

    parser.add_argument('--cfl-scan', action='store_true',
                        help='Scan Courant numbers to check where the scheme goes unstable')
    parser.add_argument('--scan-n', type=int, default=12, help='Cells per axis for the scan')
    parser.add_argument('--scan-points', type=int, default=11)
    parser.add_argument('--scan-range', type=float, nargs=2, default=(0.5, 3.0))
    parser.add_argument('--scan-steps', type=int, default=60,
                        help='Steps taken at every Courant number in the scan')
    parser.add_argument('--scan-hours', type=float, default=None, metavar='H',
                        help='Run every Courant number to H simulated hours instead of --scan-steps')
    parser.add_argument('--plot-cfl', metavar='JSON_PATH', nargs='+',
                        help='Plot cfl scan results files, one per Courant number or not, then exit')
    sweep.add_common_arguments(parser)
    args = parser.parse_args()

    if args.plot:
        plot_conservation(args.plot)
        plot_diagnostics(args.plot)
        plot_hovmoller(args.plot)
        plot_height_time(args.plot)
        return

    if args.plot_cfl:
        plot_cfl(args.plot_cfl)
        return

    if args.cfl_scan:
        courants = np.linspace(*args.scan_range, args.scan_points)
        T = args.scan_hours * 3600 if args.scan_hours is not None else None
        cfl_scan(f"cfl_scan_{args.job_id}.json", courants, steps=args.scan_steps,
                 n=args.scan_n, p=args.polynomial_order, T=T)
        return

    params, suffix = None, ""
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        suffix = f"_{Path(args.params).stem}"
    run(f"timestepping_{args.job_id}{suffix}.json", T=args.hours * 3600, p=args.polynomial_order,
        courant=args.courant, backup=args.backup, params=params)

if __name__ == '__main__':
    main()

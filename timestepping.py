"""Timestepped run diagnostics: how the flow evolves, and whether the CFL limit is where
prognostic_solver.dt() says it is.
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
from dataclasses import dataclass
import time
import numpy as np
import sweep
from parameters import PhysicalParams

DAY = 24 * 3600

# A run is called unstable once max|q| passes this multiple of its initial value. Growth
# that large is never physical here, and stopping there keeps a diverging scan point from
# running to the end in floating point garbage.
BLOWUP_FACTOR = 1e3

# The panels of the diagnostics figure, in order, as (field, y label, scientific y ticks).
# Labelled as variator.py labels the same quantities, so the two figures read alike.
DIAGNOSTIC_PANELS = [
    ("trop_height", r"$\min\, z_\text{trop}$ [\unit{\meter}]", False),
    ("min_pressure", r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]", False),
    ("max_wind", r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]", False),
    ("min_vort", r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]", True),
    ("dt", r"$\Delta t$ [\unit{\second}]", False),
]

ACCENT = '#004488'

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
    # Along the jet axis at the points _hovmoller_x gives: q - q_bar at the anomaly's height,
    # and the surface pressure anomaly [hPa]. None in results files written before they were added.
    q_line : list = None
    pressure_line : list = None
    # The minimum of the vorticity anomaly over each z level, lowest first, and those levels'
    # heights - which never change, so only the first record carries them.
    vort_profile : list = None
    level_heights : list = None

@dataclass
class CflRecord:
    """One Courant number's outcome, over the same number of steps as every other.

    steps is how many it actually took, which is fewer than asked for when it diverged.
    """
    courant : float
    steps : int
    growth : float
    stable : bool

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
        # The background q the anomaly sits on, so the upper line shows the anomaly alone
        self.q_bar = Function(solver.q.function_space()).interpolate(solver.atmos.q_bar())
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
        q_line=lines.sample(solver.q - lines.q_bar, lines.upper),
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

def _build_solver(n, p):
    from parameters import SolverParams, PhysicalParams
    from prognostic_solver import PrognosticSolver

    solver_params = SolverParams(nx=n, ny=n, nz=n, polynomial_order=p, check_flux=False)
    return PrognosticSolver(solver_params, PhysicalParams(), matfree=True)

def run(out_path, T=DAY, n=None, p=None, courant=None):
    """Step to T, recording diagnostics every step, and write them to out_path as we go.

    The records are rewritten after every step rather than at the end, so a run that is
    interrupted - or that a laptop sleeps through - still leaves everything it reached.
    """
    from firedrake.petsc import PETSc
    from prognostic_solver import SAFETY
    from parameters import SolverParams

    sweep.quiet_petsc()
    n = n if n is not None else SolverParams.nx
    p = p if p is not None else SolverParams.polynomial_order
    courant = courant if courant is not None else SAFETY

    solver = _build_solver(n, p)
    lines = _HovmollerSampling(solver)
    initial = _diagnostics(solver, lines)
    records = [StepRecord(t=0.0, dt=float('nan'), level_heights=lines.heights, **initial)]

    PETSc.Sys.Print(f"Stepping to {T / 3600:.1f} h at n={n}, p={p}, Courant={courant}")
    started = time.perf_counter()

    while solver.t < T:
        dt = solver.step(solver.dt(safety=courant))
        diagnostics = _diagnostics(solver, lines)
        records.append(StepRecord(t=solver.t, dt=dt, **diagnostics))

        if sweep.is_main_rank():
            sweep.save_records(out_path, records)

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

    PETSc.Sys.Print(f"Done in {_format_duration(time.perf_counter() - started)}, "
                    f"{len(records) - 1} steps, wrote {out_path}")
    return records

def cfl_scan(out_path, courants, steps=60, n=12, p=None):
    """Take the same number of steps at each Courant number and record how far max|q| ran away.

    prognostic_solver.dt() claims the limit is dt = dx_eff / |u|, where dx_eff is the gap
    between a cell edge and the nearest interior Gauss-Lobatto node, (1 - x_max) dx / 2.
    A run at Courant number 1 sits exactly on it. If that is right, everything below 1
    stays bounded and everything above it diverges.

    Every point takes the same number of steps rather than running to the same final time.
    Instability is a per-step amplification that compounds, so what has to be held equal
    across the scan is how many chances it had to compound - running to a fixed time would
    hand the largest steps the fewest of them, which is backwards.

    Coarser than a production run by default: this is a check on a dimensionless number,
    and the scan has to step through it many times over.
    """
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
        solver = _build_solver(n, p)
        solver.resolve()
        initial = max_abs_q(solver)

        growth, taken = 1.0, 0
        for _ in range(steps):
            solver.step(solver.dt(safety=courant))
            taken += 1
            solver.resolve()
            growth = max_abs_q(solver) / initial

            if not np.isfinite(growth) or growth > BLOWUP_FACTOR:
                break

        stable = bool(np.isfinite(growth) and growth <= BLOWUP_FACTOR)
        records.append(CflRecord(courant=float(courant), steps=taken,
                                 growth=float(growth) if np.isfinite(growth) else BLOWUP_FACTOR,
                                 stable=stable))
        PETSc.Sys.Print(f"Courant {courant:.2f}: {taken:4d} steps, max|q| grew {growth:.4g}x, "
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

    volume = PhysicalParams().domain_volume
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
    """Hovmoller diagrams along the jet axis: the upper PV anomaly beside the surface low.

    The two share their axes, so a system's track reads as a diagonal whose slope is its
    phase speed, and any horizontal offset between the tracks is the tilt between the
    upper anomaly and the surface low it induces.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import CenteredNorm
    import plot_utils

    if not sweep.is_main_rank():
        return

    records = sweep.load_records(json_path, StepRecord)
    if records[0].q_line is None:
        print(f"{json_path} predates the jet axis lines - no Hovmoller diagram")
        return
    plot_utils.apply_style()

    hours = np.array([r.t for r in records]) / 3600
    panels = [
        (np.array([r.q_line for r in records]), r"$q - \bar{q}$ [\unit{\per\second}]",
         r"$z = z_\text{trop}$", True),
        (np.array([r.pressure_line for r in records]), r"$p^*$ [\unit{\hecto\pascal}]",
         r"$z = 0$", False),
    ]
    x_km = _hovmoller_x(PhysicalParams().Lx, panels[0][0].shape[1]) / 1e3

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(plot_utils.FIGURE_SIZE[0], 4.5))
    for ax, (values, label, title, scientific) in zip(axes, panels):
        # A percentile, not the max, sets the scale: one grid-scale spike would otherwise
        # wash every coherent feature out to white
        half_range = np.nanpercentile(np.abs(values), 99.5)
        mesh = ax.pcolormesh(x_km, hours, values, cmap='RdBu_r',
                             norm=CenteredNorm(halfrange=half_range),
                             shading='nearest', rasterized=True)
        # No zero contour - it would trace every sign flip of grid-scale noise
        levels = np.delete(np.linspace(-half_range, half_range, 9), 4)
        ax.contour(x_km, hours, values, levels=levels, colors='k', linewidths=0.4)
        colorbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.12, extend='both')
        colorbar.set_label(label)
        if scientific:
            colorbar.formatter.set_powerlimits((0, 0))
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
            linestyle='--', linewidth=1.2, label=r"$\min\, z_\text{trop}$")

    colorbar = fig.colorbar(mesh, ax=ax, extend='min')
    colorbar.set_label(r"$\min_{x,y}\, (\zeta_g - \bar{\zeta}_g)$ [\unit{\per\second}]")
    colorbar.formatter.set_powerlimits((0, 0))
    ax.set_xlabel(r"$t$ [\unit{\hour}]")
    ax.set_ylabel(r"$z$ [\unit{\kilo\meter}]")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_cfl(json_path, output_path="tex/plots/cfl_verification.pdf"):
    """Growth of max|q| against Courant number, either side of the predicted limit."""
    import matplotlib.pyplot as plt
    import plot_utils

    if not sweep.is_main_rank():
        return

    records = sorted(sweep.load_records(json_path, CflRecord), key=lambda r: r.courant)
    plot_utils.apply_style()

    courant = np.array([r.courant for r in records])
    growth = np.array([r.growth for r in records])

    plt.figure(figsize=plot_utils.SQUARE_HALF_FIGURE_SIZE)
    plt.axvspan(1.0, max(courant.max(), 1.0), color='0.9', zorder=0)
    plt.axvline(1.0, color='0.4', linestyle='--', linewidth=1.2,
                label=r'Predicted limit, $\mathrm{C}=1$')
    plt.semilogy(courant, growth, color=ACCENT, marker='o', markersize=4, linewidth=1.5,
                 label=r'$\max|q| / \max|q|_{t=0}$')

    plt.xlabel(r'Courant number $\mathrm{C} = \Delta t\, |\mathbf{u}| / \Delta x_\text{eff}$')
    plt.ylabel(r'Growth of $\max|q|$')
    plot_utils.finish_figure(output_path, legend_kwargs={'loc': 'upper left', 'fontsize': 7})

def main():
    parser = argparse.ArgumentParser(description='Run and plot a timestepped simulation')
    parser.add_argument('-T', '--hours', type=float, default=24, help='Simulated hours to run for')
    parser.add_argument('-n', type=int, default=None, help='Cells per axis (default: SolverParams.nx)')
    parser.add_argument('-p', '--polynomial_order', type=int, default=None)
    parser.add_argument('-c', '--courant', type=float, default=None,
                        help='Fraction of the CFL limit to step at (default: prognostic_solver.SAFETY)')

    parser.add_argument('--cfl-scan', action='store_true',
                        help='Scan Courant numbers to check where the scheme goes unstable')
    parser.add_argument('--scan-n', type=int, default=12, help='Cells per axis for the scan')
    parser.add_argument('--scan-points', type=int, default=11)
    parser.add_argument('--scan-range', type=float, nargs=2, default=(0.5, 3.0))
    parser.add_argument('--scan-steps', type=int, default=60,
                        help='Steps taken at every Courant number in the scan')
    parser.add_argument('--plot-cfl', metavar='JSON_PATH', help='Plot a cfl scan results file, then exit')
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
        cfl_scan(f"cfl_scan_{args.job_id}.json", courants, steps=args.scan_steps,
                 n=args.scan_n, p=args.polynomial_order)
        return

    run(f"timestepping_{args.job_id}.json", T=args.hours * 3600, n=args.n,
        p=args.polynomial_order, courant=args.courant)

if __name__ == '__main__':
    main()

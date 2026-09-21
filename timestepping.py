"""Timestepped run diagnostics: how the flow evolves, and whether the CFL limit is where
prognostic_solver.dt() says it is.

Solving and plotting are separate entry points, as in time_complexity.py - the solve writes
a json and the plotting reads one, so the figures can be drawn wherever LaTeX is installed
rather than only where Firedrake is. Firedrake is imported inside the solve functions for
the same reason.
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
from dataclasses import dataclass
import time
import numpy as np
import sweep

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

@dataclass
class CflRecord:
    """One Courant number's outcome, over the same number of steps as every other.

    steps is how many it actually took, which is fewer than asked for when it diverged.
    """
    courant : float
    steps : int
    growth : float
    stable : bool

def _diagnostics(solver):
    """Every scalar the two figures need, from the state the solver is currently on."""
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
    )

def _max_abs(diagnostics):
    """Largest |q| in a diagnostics dict - the blow-up detector."""
    return max(abs(diagnostics["q_min"]), abs(diagnostics["q_max"]))

def _overshoot(records):
    """How far q has strayed outside its initial range, as a fraction of that range.

    Advection moves extrema around but makes no new ones, and the inflow carries in the
    initial field, so nothing here should push q past where it started. Anything above zero
    is the discretisation overshooting - the Gibbs oscillation a high polynomial order
    throws off a feature the mesh cannot resolve, which is what puts spurious specks of
    stratospheric air either side of the tropopause.
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
    initial = _diagnostics(solver)
    records = [StepRecord(t=0.0, dt=float('nan'), **initial)]

    PETSc.Sys.Print(f"Stepping to {T / 3600:.1f} h at n={n}, p={p}, Courant={courant}")
    started = time.perf_counter()

    while solver.t < T:
        dt = solver.step(solver.dt(safety=courant))
        diagnostics = _diagnostics(solver)
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

    prognostic_solver.dt() claims the limit is dt = dx / (|u|(2p+1)), so a run at Courant
    number 1 sits exactly on it. If that is right, everything below 1 stays bounded and
    everything above it diverges.

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
    """
    records = sweep.load_records(json_path, StepRecord)

    def drift(field):
        values = np.array([getattr(r, field) for r in records])
        return (values - values[0]) / abs(values[0])

    _stacked_panels(records, [
        (drift("mass"), r"Drift in $\int q \,\mathrm{d}V$", True),
        (drift("enstrophy"), r"Drift in $\int q^2 \,\mathrm{d}V$", True),
        (_overshoot(records), r"$q$ beyond its initial range", True),
    ], output_path, height=1.7)

def plot_diagnostics(json_path, output_path="tex/plots/timestepping_diagnostics.pdf"):
    """The resolved-atmosphere diagnostics against time, one panel each."""
    records = sweep.load_records(json_path, StepRecord)

    _stacked_panels(records, [
        (np.array([getattr(r, field) for r in records]), y_label, scientific)
        for field, y_label, scientific in DIAGNOSTIC_PANELS
    ], output_path)

def plot_cfl(json_path, output_path="tex/plots/cfl_verification.pdf"):
    """Growth of max|q| against Courant number, either side of the predicted limit."""
    import matplotlib.pyplot as plt
    import plot_utils

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

    plt.xlabel(r'Courant number $\mathrm{C} = \Delta t\, |\mathbf{u}|(2p+1) / \Delta x$')
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

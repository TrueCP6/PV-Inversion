"""Measure how the psi solution converges as the mesh is refined.

The manufactured solution of mms_checker.py is the exact solution here, so every sweep point
is measured against it directly - no reference solve, and nothing held in memory between
points. It is shaped like the solution at the control parameters (see MMSChecker), so the
orders measured are the ones the real problem sees.
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
from dataclasses import dataclass
import numpy as np
import sweep
from parameters import PhysicalParams
from petsc4py import PETSc

# Define a way to store run results
@dataclass
class ErrorRecord:
    p : int
    N : int
    dx : float
    error : float

    def dofs(self):
        return sweep.dof_count(self.p, self.N)

def _measure_error(N, p, args):
    """Solve psi at one (p, N) against the manufactured solution and hand back just the error,
    so the solver - and the PMG hierarchy it holds - can be collected before the next point."""
    from mms_checker import MMSChecker

    solver = sweep.build_solver(N, p, matfree=True, ksp_rtol=args.ksp_rtol,
                               quadrature_degree=args.quadrature_degree, atmos_cls=MMSChecker)
    solver.solve_psi()
    return solver.atmos.calc_error(solver.psi_soln)

def run_sweep(args):
    """Sweep every (p, N) point against the manufactured solution, writing the results so far
    after each one."""
    sweep.quiet_petsc()
    from firedrake.petsc import PETSc

    out_path = f"error_convergence_{args.job_id}.json"
    records, skipped = [], []

    for p in range(2, args.max_p + 1):
        for N in sweep.resolutions_for_dofs(args.max_dofs, args.num_resolutions, p):
            try:
                error = _measure_error(int(N), p, args)
            # a point that dies takes the sweep with it if it dies on only some ranks
            except Exception as exc:
                PETSc.Sys.Print(f"p = {p}, N = {N} failed ({exc}), skipping it")
                skipped.append((p, int(N)))
                continue

            PETSc.Sys.Print(f"p = {p}, N = {N}: relative error {error:.3e}")
            records.append(ErrorRecord(p=int(p), N=int(N),
                                       dx=PhysicalParams().Lx / int(N), error=error))

            PETSc.garbage_cleanup(PETSc.COMM_WORLD)

            # Rewritten every point, so a run that is killed part way through still
            # leaves the errors measured before it.
            sweep.save_records(out_path, records, indent=2)

    # A lost point leaves nothing behind in the results file, so a sweep that lost a third
    # of its points looks exactly like one that was asked for fewer. Say so at the end.
    if skipped:
        PETSc.Sys.Print(f"{len(skipped)}/{len(records) + len(skipped)} points skipped: "
                        + ", ".join(f"p={p} N={N}" for p, N in sorted(skipped)))

def plot_error_convergence(json_path, output_path="tex/plots/error_convergence.pdf"):
    """
    Create a log-log plot of relative error vs mesh spacing from an
    error_convergence_*.json results file, one line per polynomial order.
    """
    if not sweep.is_main_rank():
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, LogLocator
    import plot_utils
    plot_utils.apply_style()

    records = sweep.load_records(json_path, ErrorRecord)

    orders = sorted({record.p for record in records})
    # p is ordinal, so shade the lines through a sequential colour map rather than picking arbitrary colours
    colours = plt.cm.viridis(np.linspace(0, 1, len(orders)))
    markers = ['o', 's', '^', 'D', 'v']

    plt.figure(figsize=plot_utils.FIGURE_SIZE)

    for index, (p, colour) in enumerate(zip(orders, colours)):
        points = sorted((r for r in records if r.p == p), key=lambda r: r.dx)
        dz = PhysicalParams.H / np.array([r.N for r in points])
        errors = np.array([r.error for r in points])

        plt.plot(dz, errors, color=colour, marker=markers[index % len(markers)],
                 linewidth=1.5, markersize=4, label=rf'$p = {p}$')

        PETSc.Sys.Print(f"p = {p}: average log-log slope = {plot_utils.log_log_slope(dz, errors):.3f}")

        # Points are finest-first, so a resolved series rises with dx. Where it does not, the
        # errors have likely bottomed out on the Krylov tolerance rather than the
        # discretisation, which makes the slope above meaningless rather than merely noisy.
        stalled = sum(1 for a, b in zip(errors, errors[1:]) if b <= a)
        if stalled:
            PETSc.Sys.Print(f"p = {p}: WARNING {stalled} of {len(errors) - 1} refinement steps did not "
                  f"reduce the error - check ksp_rtol is tight enough to measure this")

    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 4, 6, 8]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # avoid unlabeled minor-tick clutter

    plt.xlabel(r'Mesh spacing $\Delta z = L_z / N$ [\unit{\meter}]')
    plt.ylabel(r'Relative $L^2$ error')
    plot_utils.finish_figure(output_path)

def main():
    parser = argparse.ArgumentParser(description='Get error convergence results for the psi solver')
    parser.add_argument('-mp', '--max_p', type=int, default=10, help='Highest polynomial order to sweep; orders run 2, 3, ... max_p')
    parser.add_argument('-nr', '--num_resolutions', type=int, default=5)
    parser.add_argument('-md', '--max_dofs', type=float, default=12e6, help='Skip any (p, N) pair needing more degrees of freedom than this')
    parser.add_argument('--ksp_rtol', type=float, default=1e-8, help='Krylov tolerance for every solve, tight enough that discretisation error dominates')
    parser.add_argument('-qd', '--quadrature_degree', type=int, default=None, help="Quadrature degree for every form. Defaults to 3p, which integrates the bilinear form exactly. Pass -1 to go back to UFL's own estimate of roughly 6p.")
    sweep.add_common_arguments(parser)
    args = parser.parse_args()

    if args.plot:
        plot_error_convergence(args.plot)
        return

    run_sweep(args)

if __name__ == '__main__':
    main()

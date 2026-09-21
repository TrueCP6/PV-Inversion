"""Measure how the psi solution converges as the mesh is refined.

There is no analytical solution to compare against, so a faux exact solution is solved
once on a mesh far finer and of far higher order than any sweep point, to a tight Krylov
tolerance, and held in memory while every sweep point is measured against it in turn.
"""

from hash_seed import use_same_hash
use_same_hash()
import argparse
from dataclasses import dataclass
import numpy as np
import sweep
from parameters import PhysicalParams

# Define a way to store run results
@dataclass
class ErrorRecord:
    p : int
    N : int
    dx : float
    error : float

    def dofs(self):
        return sweep.dof_count(self.p, self.N)

def _solve_psi(N, p, args):
    """Solve psi at one (p, N) and hand back just the solution, so the solver behind it -
    and the PMG hierarchy it holds - can be collected before the next point is built."""
    solver = sweep.build_solver(N, p, matfree=True, ksp_rtol=args.ksp_rtol,
                               quadrature_degree=args.quadrature_degree)
    solver.solve_psi()
    return solver.psi_soln

def run_sweep(args):
    """Solve the reference, then sweep every (p, N) point against it, writing the results
    so far after each one."""
    sweep.quiet_petsc()
    from firedrake.petsc import PETSc
    from math_utils import relative_error

    exact = _solve_psi(args.exact_N, args.exact_p, args)
    PETSc.Sys.Print(f"Solved the reference at p = {args.exact_p}, N = {args.exact_N} "
                    f"({sweep.dof_count(args.exact_p, args.exact_N):.3g} dofs)")

    out_path = f"error_convergence_{args.job_id}.json"
    records, skipped = [], []

    for p in range(2, args.max_p + 1):
        for N in sweep.resolutions_for_dofs(args.max_dofs, args.num_resolutions, p):
            try:
                psi = _solve_psi(int(N), p, args)
                error = relative_error(exact, psi)
            # a point that dies takes the sweep with it if it dies on only some ranks
            except Exception as exc:
                PETSc.Sys.Print(f"p = {p}, N = {N} failed ({exc}), skipping it")
                skipped.append((p, int(N)))
                continue

            PETSc.Sys.Print(f"p = {p}, N = {N}: relative error {error:.3e}")
            records.append(ErrorRecord(p=int(p), N=int(N),
                                       dx=PhysicalParams().Lx / int(N), error=error))

            del psi
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

        plt.scatter(dz, errors, color=colour, marker=markers[index % len(markers)],
                   s=30, label=rf'$p = {p}$')

        print(f"p = {p}: average log-log slope = {plot_utils.log_log_slope(dz, errors):.3f}")

        # Points are finest-first, so a resolved series rises with dx. Where it does not, the
        # reference is likely too close in resolution to the sweep points, which makes the
        # slope above meaningless rather than merely noisy.
        stalled = sum(1 for a, b in zip(errors, errors[1:]) if b <= a)
        if stalled:
            print(f"p = {p}: WARNING {stalled} of {len(errors) - 1} refinement steps did not "
                  f"reduce the error - check the reference is fine enough to measure this")

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
    parser.add_argument('--exact_N', type=int, default=32, help='Mesh resolution of the faux exact solution')
    parser.add_argument('--exact_p', type=int, default=12, help='Polynomial order of the faux exact solution')
    parser.add_argument('--ksp_rtol', type=float, default=1e-12, help='Krylov tolerance for every solve, tight enough that discretisation error dominates')
    parser.add_argument('-qd', '--quadrature_degree', type=int, default=None, help="Quadrature degree for every form. Defaults to 3p, which integrates the bilinear form exactly. Pass -1 to go back to UFL's own estimate of roughly 6p.")
    sweep.add_common_arguments(parser)
    args = parser.parse_args()

    if args.plot:
        plot_error_convergence(args.plot)
        return

    run_sweep(args)

if __name__ == '__main__':
    main()

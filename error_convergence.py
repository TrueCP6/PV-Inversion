"""Measure how the psi solution converges as the mesh is refined.

There is no analytical solution to compare against, so a faux exact solution is
solved once on a mesh far finer and of far higher order than any sweep point,
to a tight Krylov tolerance, and checkpointed to disk. Every (p, N) point then
reloads that checkpoint in its own process - see sweep.py for why each point
needs a process to itself - and measures its relative error against it.
"""

import argparse
import os
from dataclasses import dataclass
import numpy as np

import sweep
from parameters import PhysicalParams

EXACT_FUNCTION_NAME = "psi_exact"
MESH_NAME_ATTR = "mesh_name"

# Define a way to store run results
@dataclass
class ErrorRecord:
    p : int
    N : int
    dx : float
    error : float

    def dofs(self):
        return sweep.dof_count(self.p, self.N)

def _load_exact(path):
    """Reopen the faux exact solution written by _run_exact_solve."""
    from firedrake import CheckpointFile

    with CheckpointFile(path, "r") as checkpoint:
        mesh = checkpoint.load_mesh(checkpoint.get_attr("/", MESH_NAME_ATTR))
        return checkpoint.load_function(mesh, EXACT_FUNCTION_NAME)

def _run_exact_solve(args):
    """Solve psi on the fine reference mesh and checkpoint it to args.out."""
    sweep.quiet_petsc()
    from firedrake import CheckpointFile

    solver = sweep.build_solver(args.N, args.polynomial_order, matfree=True, ksp_rtol=args.ksp_rtol)
    solver.solve_psi()

    psi = solver.psi_soln
    psi.rename(EXACT_FUNCTION_NAME)

    with CheckpointFile(args.out, "w") as checkpoint:
        checkpoint.save_function(psi)
        # load_mesh needs the name Firedrake derived for the extruded mesh, which
        # the reading process has no other way of knowing.
        checkpoint.set_attr("/", MESH_NAME_ATTR, psi.function_space().mesh().name)

def _run_point(args):
    """Solve one (p, N) point and write its error against the exact solution to args.out."""
    sweep.quiet_petsc()
    from math_utils import relative_error
    from firedrake.petsc import PETSc

    solver = sweep.build_solver(args.N, args.polynomial_order, matfree=True, ksp_rtol=args.ksp_rtol)
    solver.solve_psi()

    exact = _load_exact(args.exact)
    rel_error = relative_error(exact, solver.psi_soln)
    PETSc.Sys.Print(f"Relative error: {rel_error:.3f}")

    record = ErrorRecord(
        p=args.polynomial_order,N=args.N,
        dx=PhysicalParams().Lx / args.N,
        error=rel_error
    )

    if sweep.is_main_rank():
        sweep.save_records(args.out, [record])

def _resolutions(args, p):
    """Mesh resolutions to sweep, geometrically spaced. N is limited by p based on the max dofs."""
    return sweep.resolutions_for_dofs(args.min_dofs, args.max_dofs, args.num_resolutions, p)

def _exact_solution_path(args):
    """Solve for the faux exact solution unless a usable checkpoint already exists."""
    path = os.path.abspath(args.exact or f"psi_exact_{args.job_id}.h5")

    if not os.path.exists(path):
        sweep.run_script(__file__, args.ranks,
                         ["--exact-solve", "-N", args.exact_N, "-p", args.exact_p,
                          "--ksp_rtol", args.ksp_rtol, "-o", path])
    return path

def plot_error_convergence(json_path, output_path="tex/error_convergence.pdf"):
    """
    Create a log-log plot of relative error vs mesh spacing from an
    error_convergence_*.json results file, one line per polynomial order.
    """
    import matplotlib.pyplot as plt

    import plot_utils
    plot_utils.apply_style()

    records = sweep.load_records(json_path, ErrorRecord)

    orders = sorted({record.p for record in records})
    # p is ordinal, so shade the lines through a sequential colour map rather than
    # picking arbitrary colours. The pale end of viridis is cut off to keep every
    # line readable on white.
    colours = plt.cm.viridis(np.linspace(0, 0.8, len(orders)))
    markers = ['o', 's', '^', 'D', 'v']

    plt.figure(figsize=plot_utils.FIGURE_SIZE)

    for index, (p, colour) in enumerate(zip(orders, colours)):
        points = sorted((r for r in records if r.p == p), key=lambda r: r.dx)
        dx_km = np.array([r.dx for r in points]) / 1e3  # metres to kilometres
        errors = np.array([r.error for r in points])

        plt.loglog(dx_km, errors, color=colour, marker=markers[index % len(markers)],
                   linewidth=1.5, markersize=4, label=rf'$p = {p}$')

        print(f"p = {p}: average log-log slope = {plot_utils.log_log_slope(dx_km, errors):.3f}")

    plt.xlabel(r'Mesh spacing $\Delta x = L_x / N$ [\unit{\kilo\meter}]')
    plt.ylabel(r'Relative $L^2$ error')
    plot_utils.finish_figure(output_path)

def main():
    parser = argparse.ArgumentParser(description='Get error convergence results for the psi solver')
    parser.add_argument('-mp', '--max_p', type=int, default=10, help='Highest polynomial order to sweep; orders run 2, 4, ... max_p')
    parser.add_argument('-nr', '--num_resolutions', type=int, default=5)
    parser.add_argument('-md', '--max_dofs', type=float, default=12e6, help='Skip any (p, N) pair needing more degrees of freedom than this')
    parser.add_argument('--exact_N', type=int, default=32, help='Mesh resolution of the faux exact solution')
    parser.add_argument('--exact_p', type=int, default=12, help='Polynomial order of the faux exact solution')
    parser.add_argument('--ksp_rtol', type=float, default=1e-12, help='Krylov tolerance for every solve, tight enough that discretisation error dominates')
    parser.add_argument('-e', '--exact', metavar='H5_PATH',help='Checkpoint holding the faux exact solution. Solved for and written here if the file does not exist yet, so an expensive exact solve can be reused')
    sweep.add_common_arguments(parser)
    sweep.add_point_arguments(parser)
    # Internal re-exec entry points - not for direct use.
    parser.add_argument('-p', '--polynomial_order', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--exact-solve', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_point:
        _run_point(args)
        return

    if args.exact_solve:
        _run_exact_solve(args)
        return

    if args.plot:
        plot_error_convergence(args.plot)
        return

    exact_path = _exact_solution_path(args)

    records = []
    max_ranks = args.ranks
    for p in range(2, args.max_p + 1, 2):
        for N in _resolutions(args, p):

            ranks = sweep.calc_ranks(p, N, max_ranks) # Use the optimal number of ranks if max_ranks * 50000 > dofs
            data = sweep.run_point(
                __file__, ranks,
                ["-N", N, "-p", p, "--ksp_rtol", args.ksp_rtol, "-e", exact_path],
            ErrorRecord)
            records.extend(data)

    sweep.save_records(f"error_convergence_{args.job_id}.json", records, indent=2)

if __name__ == '__main__':
    main()

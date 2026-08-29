"""Measure how the psi solution converges as the mesh is refined.

There is no analytical solution to compare against, so a faux exact solution is
solved once on a mesh far finer and of far higher order than any sweep point,
to a tight Krylov tolerance, and checkpointed to disk. Every (p, N) point then
measures its relative error against it.

A point takes two processes, not one: it solves and checkpoints psi, then reopens
that checkpoint alongside the reference to measure the error. See sweep.py for why
a point cannot share a process with the next one, and _evaluate_point for why the
solve cannot share a process with its own error measurement.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
import numpy as np
import sweep
from parameters import PhysicalParams

# todo: fix cache collisions

EXACT_FUNCTION_NAME = "psi_exact"
POINT_FUNCTION_NAME = "psi_point"
MESH_NAME_ATTR = "mesh_name"
# Stamped onto the reference checkpoint so a stale file cannot be reused as if it matched the
# flags it is being reused under - the checkpoint carries no other record of how it was made.
EXACT_N_ATTR = "exact_N"
EXACT_P_ATTR = "exact_p"

# Define a way to store run results
@dataclass
class ErrorRecord:
    p : int
    N : int
    dx : float
    error : float

    def dofs(self):
        return sweep.dof_count(self.p, self.N)

def _save_solution(path, psi, name, attrs=None):
    """Checkpoint psi under the given name, with whatever the reader needs to reopen it."""
    from firedrake import CheckpointFile

    psi.rename(name)
    with CheckpointFile(path, "w") as checkpoint:
        checkpoint.save_function(psi)
        # load_mesh needs the name Firedrake derived for the extruded mesh, which
        # the reading process has no other way of knowing.
        checkpoint.set_attr("/", MESH_NAME_ATTR, psi.function_space().mesh().name)
        for key, value in (attrs or {}).items():
            checkpoint.set_attr("/", key, value)

def _load_solution(path, name):
    """Reopen a function checkpointed by _save_solution, mesh and all."""
    from firedrake import CheckpointFile

    with CheckpointFile(path, "r") as checkpoint:
        mesh = checkpoint.load_mesh(checkpoint.get_attr("/", MESH_NAME_ATTR))
        return checkpoint.load_function(mesh, name)

def _load_exact(path, expected_N=None, expected_p=None):
    """Reopen the faux exact solution written by _run_exact_solve.

    _exact_solution_path deliberately reuses an existing checkpoint rather than paying for
    the fine solve again, so nothing otherwise stops a reference solved at one (N, p) being
    measured against as if it were another - which silently changes what every error in the
    results file means. Refuse the mismatch rather than reporting it as data.
    """
    from firedrake import CheckpointFile
    from firedrake.petsc import PETSc

    with CheckpointFile(path, "r") as checkpoint:
        mesh = checkpoint.load_mesh(checkpoint.get_attr("/", MESH_NAME_ATTR))
        exact = checkpoint.load_function(mesh, EXACT_FUNCTION_NAME)
        try:
            actual = (checkpoint.get_attr("/", EXACT_N_ATTR), checkpoint.get_attr("/", EXACT_P_ATTR))
        except (KeyError, RuntimeError):
            actual = None

    if actual is None:
        PETSc.Sys.Print(f"WARNING: {path} predates the exact_N/exact_p stamp, so the reference "
                        f"resolution cannot be checked. Delete it to have it re-solved.")
    elif expected_N is not None and tuple(int(v) for v in actual) != (expected_N, expected_p):
        raise ValueError(
            f"{path} holds the reference solved at N={actual[0]}, p={actual[1]}, but this run "
            f"asked for N={expected_N}, p={expected_p}. Delete the file or point --exact "
            f"somewhere else."
        )
    return exact

def _run_exact_solve(args):
    """Solve psi on the fine reference mesh and checkpoint it to args.out."""
    sweep.quiet_petsc()

    solver = sweep.build_solver(args.N, args.polynomial_order, matfree=True,
                                ksp_rtol=args.ksp_rtol, quadrature_degree=args.quadrature_degree)
    solver.solve_psi()

    _save_solution(args.out, solver.psi_soln, EXACT_FUNCTION_NAME,
                   {EXACT_N_ATTR: args.N, EXACT_P_ATTR: args.polynomial_order})

def _run_point(args):
    """Solve one (p, N) point and checkpoint psi to args.out for the error stage to pick up."""
    sweep.quiet_petsc()

    solver = sweep.build_solver(args.N, args.polynomial_order, matfree=True,
                                ksp_rtol=args.ksp_rtol, quadrature_degree=args.quadrature_degree)
    solver.solve_psi()

    _save_solution(args.out, solver.psi_soln, POINT_FUNCTION_NAME)

def _run_error(args):
    """Measure the checkpoint at args.psi against the reference and write the record to args.out."""
    sweep.quiet_petsc()
    from math_utils import relative_error
    from firedrake.petsc import PETSc

    numerical = _load_solution(args.psi, POINT_FUNCTION_NAME)
    PETSc.Sys.Print("Loaded point solution")

    exact = _load_exact(args.exact, args.exact_N, args.exact_p)
    PETSc.Sys.Print("Loaded reference solution")

    rel_error = relative_error(exact, numerical)
    PETSc.Sys.Print(f"Relative error: {rel_error:.3e}")

    record = ErrorRecord(
        p=args.polynomial_order,N=args.N,
        dx=PhysicalParams().Lx / args.N,
        error=rel_error
    )

    if sweep.is_main_rank():
        sweep.save_records(args.out, [record])

def _evaluate_point(args, p, N, exact_path, ranks):
    """Solve one (p, N) point and measure its error, in two processes rather than one.

    sweep.py's module docstring explains why a point needs a process to itself: the solver
    pins its mesh's PETSc state for the life of the process and dropping the Python
    references does not release it. The same argument applies within a point. Measuring the
    error builds a second mesh, a second function space - the reference is the finer of the
    two, so up to 22M dofs - and a cross-mesh interpolation onto it, and doing that in the
    solver's process holds both peaks at once. Splitting them means neither has to fit
    alongside the other.
    """
    # The checkpoint of psi is tens of MB, and the point of --exact is that its directory is
    # somewhere with room, unlike whatever /tmp happens to be on a compute node.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(exact_path)) as tmpdir:
        psi_path = os.path.join(tmpdir, "psi.h5")
        out_path = os.path.join(tmpdir, "point.json")

        shared = ["-N", N, "-p", p]
        if args.quadrature_degree is not None:
            shared += ["--quadrature_degree", args.quadrature_degree]

        stages = [
            ("solve", ["--single-point", "-o", psi_path, "--ksp_rtol", args.ksp_rtol, *shared]),
            ("error", ["--error-point", "-o", out_path, "--psi", psi_path, "-e", exact_path,
                       "--exact_N", args.exact_N, "--exact_p", args.exact_p, *shared]),
        ]

        for stage, stage_args in stages:
            try:
                sweep.run_script(__file__, ranks, stage_args)
            except subprocess.CalledProcessError as exc:
                print(f"Point -p {p} -N {N} failed in the {stage} stage "
                      f"({sweep.describe_exit(exc.returncode)}), skipping it", file=sys.stderr)
                return None

        return sweep.load_records(out_path, ErrorRecord)

def _resolutions(args, p):
    """Mesh resolutions to sweep, geometrically spaced. N is limited by p based on the max dofs."""
    return sweep.resolutions_for_dofs(args.min_dofs, args.max_dofs, args.num_resolutions, p)

def _exact_solution_path(args):
    """Solve for the faux exact solution unless a usable checkpoint already exists."""
    path = os.path.abspath(args.exact or f"psi_exact_{args.job_id}.h5")

    if os.path.exists(path):
        print(f"Reusing the reference solution already at {path} rather than re-solving it; "
              f"_load_exact will reject it if it was not solved at N={args.exact_N}, "
              f"p={args.exact_p}.", file=sys.stderr)
        return path

    exact_args = ["--exact-solve", "-N", args.exact_N, "-p", args.exact_p,
                  "--ksp_rtol", args.ksp_rtol, "-o", path]
    if args.quadrature_degree is not None:
        exact_args += ["--quadrature_degree", args.quadrature_degree]

    sweep.run_script(__file__, sweep.calc_ranks(args.exact_p, args.exact_N, args.ranks), exact_args)
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

        # Points are finest-first, so a resolved series rises with dx. Where it does not,
        # refining the mesh failed to reduce the error - which is what a reference too close
        # in resolution to the sweep points looks like, and it makes the slope above
        # meaningless rather than merely noisy.
        stalled = sum(1 for a, b in zip(errors, errors[1:]) if b <= a)
        if stalled:
            print(f"p = {p}: WARNING {stalled} of {len(errors) - 1} refinement steps did not "
                  f"reduce the error - check the reference is fine enough to measure this")

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
    parser.add_argument('-qd', '--quadrature_degree', type=int, default=None, help="Quadrature degree for every form. Defaults to 3p, which integrates the bilinear form exactly. Pass -1 to go back to UFL's own estimate of roughly 6p.")
    parser.add_argument('-e', '--exact', metavar='H5_PATH',help='Checkpoint holding the faux exact solution. Solved for and written here if the file does not exist yet, so an expensive exact solve can be reused')
    sweep.add_common_arguments(parser)
    sweep.add_point_arguments(parser)
    # Internal re-exec entry points - not for direct use.
    parser.add_argument('-p', '--polynomial_order', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--exact-solve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--error-point', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--psi', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_point:
        _run_point(args)
        return

    if args.error_point:
        _run_error(args)
        return

    if args.exact_solve:
        _run_exact_solve(args)
        return

    if args.plot:
        plot_error_convergence(args.plot)
        return

    exact_path = _exact_solution_path(args)

    records = []
    skipped = []
    max_ranks = args.ranks
    for p in range(2, args.max_p + 1, 2):
        for N in _resolutions(args, p):

            ranks = sweep.calc_ranks(p, N, max_ranks) # Capped by dofs and by base-mesh columns per rank
            data = _evaluate_point(args, p, N, exact_path, ranks)

            if data is None:
                skipped.append((p, N))
            else:
                records.extend(data)

    sweep.save_records(f"error_convergence_{args.job_id}.json", records, indent=2)

    # A skipped point leaves nothing behind in the results file, so a sweep that lost a third
    # of its points looks exactly like one that was asked for fewer. Say so at the end, where
    # it is not buried under the mpiexec output of everything that ran afterwards.
    if skipped:
        print(f"{len(skipped)}/{len(records) + len(skipped)} points skipped: "
              + ", ".join(f"p={p} N={N}" for p, N in skipped), file=sys.stderr)

if __name__ == '__main__':
    main()

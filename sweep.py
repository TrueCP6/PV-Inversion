"""Shared machinery for the (p, N) sweeps run by time_complexity.py and
error_convergence.py.

Building a PMG LinearVariationalSolver pins its mesh's PETSc/UFL state (Mat, DM,
compiled kernels) for the rest of the process - confirmed by direct profiling,
not released by lru_cache clearing, PETSc.garbage_cleanup, or explicit
destroy(). Every data point therefore gets a process of its own: the driver
calls run_point(), which re-enters the calling script under a fresh mpiexec
through that script's hidden --single-point flag.

Firedrake is imported inside the functions that need it rather than at module
scope, so a driver process or a plotting run never pays for it.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict

import numpy as np

from parameters import PhysicalParams, SolverParams

def quiet_petsc():
    """Stop PETSc reporting the options Firedrake sets on our behalf but never reads."""
    from firedrake.petsc import PETSc
    PETSc.Options().setValue("options_left", "false")

def is_main_rank():
    """True on the one rank that should write results out."""
    from mpi4py import MPI
    return MPI.COMM_WORLD.rank == 0

def dof_count(p : int, N : int):
    """Degrees of freedom of a Q_p space on an N x N x N mesh."""
    return (p * N + 1) ** 3

def resolutions_for_dofs(min_dofs : int, max_dofs : int, num_resolutions : int, p : int):
    """Resolutions whose dof counts are geometrically spaced across the given range."""
    dofs = np.geomspace(min_dofs, max_dofs, num_resolutions)
    return np.unique(np.round((np.cbrt(dofs) - 1) / p).astype(int))

def build_solver(N : int, p : int, matfree : bool, ksp_rtol : float = SolverParams.ksp_rtol,
                 quadrature_degree : int = SolverParams.quadrature_degree):
    """Set up the psi solver on an N x N x N mesh of Q_p elements."""
    from domain_builder import DomainBuilder
    from barnes_atmosphere import BarnesAtmosphere
    from solver import Solver

    phys_params = PhysicalParams()
    solver_params = SolverParams(nx=N, ny=N, nz=N, check_flux=False, polynomial_order=p,
                                 ksp_rtol=ksp_rtol, quadrature_degree=quadrature_degree)

    domain = DomainBuilder(solver_params, phys_params)
    atmos = BarnesAtmosphere(domain.mesh(), domain.func_space(), phys_params)

    return Solver(atmos, solver_params, matfree)

def run_script(script_path, ranks : int, args):
    """Run script under a fresh mpiexec, so its Firedrake state dies with the process."""
    subprocess.run([
        "mpiexec", "-n", str(ranks),
        sys.executable, os.path.abspath(script_path), *map(str, args)
    ], check=True)

def describe_exit(returncode : int):
    """Explain a child mpiexec's exit status.

    A point killed for memory and a point that failed to converge both surface here as a
    non-zero return code but need completely different fixes, so say which one it was.
    mpiexec reports a SIGKILLed rank as -9 or as 128+9 depending on how it was launched.
    """
    if returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
        return (f"exit {returncode}, a rank was SIGKILLed - almost always the OOM killer; "
                "check the .err for oom_kill events and sacct for MaxRSS")
    return f"exit {returncode}"

def run_point(script_path, ranks : int, point_args, record_cls):
    """Evaluate one sweep point in its own process and return the records it wrote.

    A point that fails - the coarsest meshes can leave PMG's Chebyshev smoother
    indefinite, which CG refuses - costs that point only, rather than the hours
    of sweeping either side of it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "point.json")
        try:
            run_script(script_path, ranks, ["--single-point", "-o", out_path, *point_args])
        except subprocess.CalledProcessError as exc:
            print(f"Point {' '.join(map(str, point_args))} failed "
                  f"({describe_exit(exc.returncode)}), skipping it", file=sys.stderr)
            return []
        return load_records(out_path, record_cls)

def save_records(path, records, indent=None):
    """Write a list of dataclass records out as JSON."""
    with open(path, "w") as f:
        json.dump([asdict(record) for record in records], f, indent=indent)

def load_records(path, record_cls):
    """Read a JSON results file back into a list of record_cls instances."""
    with open(path) as f:
        return [record_cls(**record) for record in json.load(f)]

def add_common_arguments(parser):
    """Options every sweep script takes."""
    parser.add_argument('-j', '--job_id', type=int, default=0)
    parser.add_argument('-r', '--ranks', type=int, default=1, help='MPI ranks to use for each data point (each point runs in its own mpiexec process)')
    parser.add_argument('-mind', '--min_dofs', type=int, default=100000) # todo: only use a sufficient number of dofs for the available ranks with time_complexity
    parser.add_argument('--plot', metavar='JSON_PATH', help='Plot the given results file instead of generating new data, then exit')

def add_point_arguments(parser):
    """Internal re-exec entry point for a single data point - not for direct use."""
    parser.add_argument('--single-point', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-N', type=int, help=argparse.SUPPRESS)
    parser.add_argument('-o', '--out', help=argparse.SUPPRESS)

MIN_COLUMNS_PER_RANK = 32
MIN_DOFS_PER_RANK = 50000

def calc_ranks(p : int, n : int, max_ranks : int):
    """Get the ideal number of ranks for this (p,N).

    Keeps at least MIN_DOFS_PER_RANK dofs and MIN_COLUMNS_PER_RANK base-mesh columns on every rank.
    """
    by_dofs = int(dof_count(p, n) / MIN_DOFS_PER_RANK)
    by_columns = int(n * n / MIN_COLUMNS_PER_RANK)
    return max(1, min(by_dofs, by_columns, max_ranks))

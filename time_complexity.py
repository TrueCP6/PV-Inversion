import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict

import numpy as np

# Define a way to store run results
@dataclass
class TimeRecord:
    initial_run : bool
    p : int
    matfree : bool
    N : int
    time : float

    def dofs(self):
        return (self.N * self.p + 1) ** 3

def test_solve(n : int, matfree : bool, num_solves : int, polynomial_order : int, phys_params = None):
    from parameters import PhysicalParams, SolverParams
    from domain_builder import DomainBuilder
    from barnes_atmosphere import BarnesAtmosphere
    from solver import Solver

    if phys_params is None:
        phys_params = PhysicalParams()

    solver_params = SolverParams(nx=n, ny=n, nz=n, check_flux=False, polynomial_order=polynomial_order)
    domain = DomainBuilder(solver_params, phys_params)

    mesh = domain.mesh()
    func_space = domain.func_space()

    atmos = BarnesAtmosphere(mesh, func_space, phys_params)

    solver = Solver(atmos, solver_params, matfree)

    solve_times = [solver.solve_psi(True) for _ in range(num_solves)]

    return solve_times

def _run_single_point(args):
    # Building a LinearVariationalSolver pins its mesh's PETSc/UFL state (Mat, DM,
    # compiled kernels) for the rest of the process - confirmed by direct profiling,
    # not released by lru_cache clearing, PETSc.garbage_cleanup, or explicit destroy().
    # Each (N, matfree) point therefore runs as its own process via a fresh mpiexec,
    # driven by eval_ns below.
    from firedrake.petsc import PETSc
    PETSc.Options().setValue("options_left", "false")
    from mpi4py import MPI

    times = test_solve(args.N, args.matfree, args.num_solves, args.polynomial_order)

    if MPI.COMM_WORLD.rank == 0:
        records = [
            asdict(TimeRecord(initial_run=idx == 0, p=args.polynomial_order,
                              matfree=args.matfree, N=args.N, time=t))
            for idx, t in enumerate(times)
        ]
        with open(args.out, "w") as f:
            json.dump(records, f)

def eval_ns(Ns, p : int, matfree : bool, num_solves : int, ranks : int):
    records = []
    for N in Ns:
        N = int(N) # Ensure not in numpy format
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "point.json")
            cmd = [
                "mpiexec", "-n", str(ranks), sys.executable, "-u", os.path.abspath(__file__),
                "--single-point", "-N", str(N), "-p", str(p), "-ns", str(num_solves),
                "-o", out_path,
            ]
            if matfree:
                cmd.append("--matfree")
            subprocess.run(cmd, check=True)
            with open(out_path) as f:
                records.extend(TimeRecord(**r) for r in json.load(f))
    return records

def main():
    parser = argparse.ArgumentParser(description='Get performance results for ')
    parser.add_argument('-p', '--polynomial_order', type=int, default=4)
    parser.add_argument('-ns', '--num_solves', type=int, default=2)
    parser.add_argument('-ad', '--max_dofs_assembled', type=int, default=3e6)
    parser.add_argument('-md', '--max_dofs_matfree', type=int, default=6e6)
    parser.add_argument('-nr', '--num_resolutions', type=int, default=5)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    parser.add_argument('-r', '--ranks', type=int, default=1,
                        help='MPI ranks to use for each data point (each point runs in its own mpiexec process)')
    # Internal re-exec entry point for a single (N, matfree) point - not for direct use.
    parser.add_argument('--single-point', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-N', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--matfree', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-o', '--out', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_point:
        _run_single_point(args)
        return

    p = args.polynomial_order
    num_solves = args.num_solves

    min_dofs = 50000 * args.ranks # Use at least 50k dofs per rank
    if min_dofs > args.max_dofs_matfree or min_dofs > args.max_dofs_assembled:
        print("Less than 50k DoFs per rank. Use less ranks.")
        sys.exit(0)

    # Get a logarithmically spaced distribution of dofs between min and max
    dofs_assembled = np.geomspace(min_dofs, args.max_dofs_assembled, args.num_resolutions)
    dofs_matfree = np.geomspace(min_dofs, args.max_dofs_matfree, args.num_resolutions)

    # Calculate the N corresponding to each dof value

    n_assembled = (np.cbrt(dofs_assembled) - 1) / p
    n_matfree = (np.cbrt(dofs_matfree) - 1) / p

    # Convert to ints
    n_assembled = np.round(n_assembled).astype(int)
    n_matfree = np.round(n_matfree).astype(int)

    records = eval_ns(n_assembled, p, False, num_solves, args.ranks)
    records.extend(eval_ns(n_matfree, p, True, num_solves, args.ranks))

    with open(f"time_complexity_{args.job_id}.json", "w") as f:
        json.dump([asdict(record) for record in records], f, indent=2)

if __name__ == '__main__':
    main()

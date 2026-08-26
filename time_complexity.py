import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dataclasses import asdict
from parameters import *
from barnes_atmosphere import *
from domain_builder import *
from solver import *
import argparse

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

def test_solve(n : int, matfree : bool, num_solves : int, polynomial_order : int, phys_params = PhysicalParams()):
    solver_params = SolverParams(nx=n, ny=n, nz=n, check_flux=False, polynomial_order=polynomial_order)
    domain = DomainBuilder(solver_params, phys_params)

    mesh = domain.mesh()
    func_space = domain.func_space()

    atmos = BarnesAtmosphere(mesh, func_space, phys_params)

    solver = Solver(atmos, solver_params, matfree)

    solve_times = [solver.solve_psi(True) for _ in range(num_solves)]

    return solve_times

def eval_ns(Ns, p : int, matfree : bool, num_solves : int):
    records = []
    for N in Ns:
        N = int(N) # Ensure not in numpy format
        times = test_solve(N, matfree, num_solves, p)
        for idx, time in enumerate(times):
            records.append(TimeRecord(
                initial_run = idx==0,
                p = p,
                matfree = matfree,
                N = N,
                time = time,
            ))
    return records

def main():
    # Prevents a warning
    from firedrake.petsc import PETSc
    PETSc.Options().setValue("options_left", "false")

    parser = argparse.ArgumentParser(description='Get performance results for ')
    parser.add_argument('-p', '--polynomial_order', type=int, default=4)
    parser.add_argument('-ns', '--num_solves', type=int, default=2)
    parser.add_argument('-ad', '--max_dofs_assembled', type=int, default=3e6)
    parser.add_argument('-md', '--max_dofs_matfree', type=int, default=6e6)
    parser.add_argument('-nr', '--num_resolutions', type=int, default=5)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    args = parser.parse_args()
    p = args.polynomial_order
    num_solves = args.num_solves

    min_dofs = 50000 * COMM_WORLD.size # Use at least 50k dofs per rank
    if min_dofs > args.max_dofs_matfree or min_dofs > args.max_dofs_assembled:
        PETSc.Sys.Print("Less than 50k DoFs per rank. Use less ranks.")
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

    records = eval_ns(n_assembled, p, False, num_solves)
    records.extend(eval_ns(n_matfree, p, True, num_solves))

    if COMM_WORLD.rank == 0:
        with open(f"time_complexity_{args.job_id}.json", "w") as f:
            json.dump([asdict(record) for record in records], f, indent=2)

if __name__ == '__main__':
    main()
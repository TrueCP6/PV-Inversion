from firedrake import *
import math_utils
from domain_builder import DomainBuilder
from solver import Solver
from barnes_atmosphere import BarnesAtmosphere
from parameters import SolverParams, PhysicalParams
import sys

def main():
    solver_params = SolverParams()
    phys_params = PhysicalParams()

    domain = DomainBuilder(solver_params, phys_params)
    atmos = BarnesAtmosphere(domain)

    solver = Solver(atmos, False)

    solver.solve_psi() # Run the solver once as spinup

    solve_times = []
    for i in range(3):
        solve_times.append(solver.solve_psi(True))
    avg_solve_time = sum(solve_times)/len(solve_times)
    PETSc.Sys.Print(f"Average solve completed in {avg_solve_time:0.2f} sec")

if __name__ == "__main__":
    main()
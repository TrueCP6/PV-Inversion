from firedrake import *
import math_utils
from derived_quantities import ResolvedAtmosphere
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

    solver = Solver(atmos, True)
    solver.solve_psi()
    derived = ResolvedAtmosphere(solver.psi_soln, atmos)

    PETSc.Sys.Print(f'Min pressure: {derived.min_surf_pressure_hpa()}')
    PETSc.Sys.Print(f'Min vort: {derived.min_surf_vort()}')
    PETSc.Sys.Print(f'Min trop height: {derived.min_dyn_tropopause_height()}')

if __name__ == "__main__":
    main()
from firedrake import *
import math_utils
from domain_builder import DomainBuilder
from solver import Solver
from barnes_atmosphere import BarnesAtmosphere
from parameters import SolverParams, PhysicalParams

def main():
    solver_params = SolverParams()
    phys_params = PhysicalParams()

    domain_builder = DomainBuilder(solver_params, phys_params)
    mesh = domain_builder.mesh()
    V = domain_builder.func_space()

    atmos = BarnesAtmosphere(mesh, V, phys_params)
    slver = Solver(atmos, solver_params)

    slver.solve_psi() # Run the solver once as spinup

    solve_times = []
    for i in range(5):
        solve_times.append(slver.solve_psi())
    avg_solve_time = sum(solve_times)/len(solve_times)
    PETSc.Sys.Print(f"Average solve completed in {avg_solve_time:0.2f} sec")

    psi = slver.psi_soln
    u = -psi.dx(1)
    v = psi.dx(0)
    speed = sqrt(u**2 + v**2)
    func = Function(V).interpolate(speed)

    math_utils.plot_func_slice(func)

if __name__ == "__main__":
    main()
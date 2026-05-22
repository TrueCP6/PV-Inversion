from firedrake import *
import math_utils
from MMSChecker import MMSChecker
from Solver import Solver
from barnes_atmosphere import BarnesAtmosphere
from parameters import SolverParams, PhysicalParams


def main():
    solver_params = SolverParams()
    phys_params = PhysicalParams()

    # todo move this into its own method
    # todo use different function spaces for different variables
    temp_mesh = RectangleMesh(solver_params.nx, solver_params.ny, phys_params.Lx, phys_params.Ly)
    mesh = ExtrudedMesh(temp_mesh, layers=solver_params.nz, layer_height=(phys_params.H / solver_params.nz))
    V = FunctionSpace(mesh, "CG", 1)
    PETSc.Sys.Print("Built extruded mesh")

    atmos = BarnesAtmosphere(mesh, V, phys_params)
    slver = Solver(atmos, solver_params)

    psi = slver.solve_psi()

    u = -psi.dx(1)
    v = psi.dx(0)
    speed = sqrt(u**2 + v**2)
    func = Function(V).interpolate(speed)

    math_utils.plot_func_slice(func)


if __name__ == "__main__":
    main()
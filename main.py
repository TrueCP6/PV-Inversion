from firedrake import *
import math_utils
from MMSChecker import MMSChecker
from Solver import Solver
from Parameters import SolverParams, PhysicalParams


def main():
    slv = SolverParams()
    phys = PhysicalParams()

    # todo move this into its own method
    base_mesh = RectangleMesh(slv.nx, slv.ny, phys.Lx, phys.Ly)
    mesh = ExtrudedMesh(base_mesh, layers=slv.nz, layer_height=(phys.H / slv.nz))
    V = FunctionSpace(mesh, "CG", 1)
    PETSc.Sys.Print("Built extruded mesh")

    phys_params = PhysicalParams()
    checker = MMSChecker(mesh, V, phys_params)
    solver_params = SolverParams(nx=200, ny=200, nz=200)
    solver = Solver(checker, solver_params)
    psi = solver.solve_psi()

    err = checker.calc_error(solver.psi_soln)
    PETSc.Sys.Print("Error is ", err)

    x_vel = - psi.dx(1)
    y_vel = psi.dx(0)
    speed = sqrt(x_vel**2 + y_vel**2)

    speed_fn = Function(V).interpolate(speed)
    math_utils.plot_func_slice(speed_fn)

if __name__ == "__main__":
    main()
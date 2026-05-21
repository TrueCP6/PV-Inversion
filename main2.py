from firedrake import *
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
    solver.solve_psi()

    err = checker.calc_error(solver.psi_soln)
    PETSc.Sys.Print("Error is ", err)

if __name__ == "__main__":
    main()
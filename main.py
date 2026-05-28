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
    temp_mesh = RectangleMesh(
        solver_params.nx, solver_params.ny,
        phys_params.Lx, phys_params.Ly,
        #quadrilateral=True
    )
    mesh = ExtrudedMesh(
        temp_mesh,
        layers=solver_params.nz,
        layer_height=(phys_params.H / solver_params.nz)
    )
    PETSc.Sys.Print("Built extruded mesh")

    # todo maybe fdm
    V = FunctionSpace(mesh, "Q", 4)
    total_dofs = V.dof_dset.layout_vec.getSize() # can also be calculated as (degree*N+1)^3
    PETSc.Sys.Print(f"Created function space with {total_dofs} degrees of freedom")

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
from firedrake import *
import math_utils
from MMSChecker import MMSChecker
from Solver import Solver
from barnes_atmosphere import BarnesAtmosphere
from parameters import SolverParams, PhysicalParams


def main():
    slv = SolverParams()
    phys = PhysicalParams()

    # todo move this into its own method
    # todo use different function spaces for different variables
    temp_mesh = RectangleMesh(slv.nx, slv.ny, phys.Lx, phys.Ly)
    mesh = ExtrudedMesh(temp_mesh, layers=slv.nz, layer_height=(phys.H / slv.nz))
    V = FunctionSpace(mesh, "CG", 1)
    PETSc.Sys.Print("Built extruded mesh")

    phys_params = PhysicalParams()
    solver_params = SolverParams(nx=200, ny=200, nz=200)
    atmos = BarnesAtmosphere(mesh, V, phys_params)
    p = atmos.ertel_pv()

    math_utils.plot_func_slice(Function(V).interpolate(p))

if __name__ == "__main__":
    main()
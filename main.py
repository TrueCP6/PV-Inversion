from firedrake import *
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
        quadrilateral=solver_params.quadrilateral,
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

    firedrake_params = {
        # --- Top Level Solver ---
        "ksp_type": "cg",
        "ksp_rtol": 1e-6,
        "ksp_monitor": None,

        # Enable matrix-free evaluation globally
        "mat_type": "matfree",

        # --- p-Multigrid Preconditioner ---
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",

        # --- Fine & Intermediate Levels (Smoothers) ---
        # Chebyshev is generally superior to default Richardson for multigrid smoothing
        "pmg_mg_levels_ksp_type": "chebyshev",
        "pmg_mg_levels_pc_type": "jacobi",

        # --- Coarse Level ---
        "pmg_mg_coarse_ksp_type": "preonly",
        "pmg_mg_coarse_pc_type": "lu",
        # CRITICAL: Force assembly of the coarse matrix so LU can factorize it
        "pmg_mg_coarse_mat_type": "aij"
    }

    atmos = BarnesAtmosphere(mesh, V, phys_params)
    slver = Solver(atmos, solver_params, firedrake_params)

    slver.solve_psi() # Run the solver once as spinup
    solve_times = []
    for i in range(5):
        solve_times.append(slver.solve_psi())

    avg_solve_time = sum(solve_times)/len(solve_times)

    PETSc.Sys.Print(f"Average solve completed in {avg_solve_time:0.2f} sec")

    # psi = slver.psi_soln
    # u = -psi.dx(1)
    # v = psi.dx(0)
    # speed = sqrt(u**2 + v**2)
    # func = Function(V).interpolate(speed)
    #
    # math_utils.plot_func_slice(func)

if __name__ == "__main__":
    main()
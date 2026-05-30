from functools import cache
from firedrake import *
from mpi4py import MPI
import pyvista as pv

@cache
def _get_vertical_integral_solver(main_func_space):
    mesh = main_func_space.mesh()

    # Create the temporary DQ space
    h_degree, v_degree = main_func_space.ufl_element().degree()
    dq_space = FunctionSpace(mesh, "DQ", 0, vdegree=v_degree) # Use a horizontal degree of zero as there is no horizontal variation in the integrand

    # Setup placeholder functions
    integrand_placeholder = Function(main_func_space)
    solution_dq = Function(dq_space)

    I_trial = TrialFunction(dq_space)
    W_test = TestFunction(dq_space)
    n = FacetNormal(mesh)

    # Specify weak form
    vol = -I_trial * W_test.dx(2) * dx
    interior_surf_flux = jump(W_test, n[2]) * I_trial('+') * dS_h  # Upwinding
    top_exterior_surf_flux = W_test * I_trial * n[2] * ds_t

    a_I = vol + top_exterior_surf_flux + interior_surf_flux
    L_I = integrand_placeholder * W_test * dx

    int_solver_params = {
        "ksp_type": "preonly",
        "pc_type": "sor",
        "snes_lag_jacobian": -2,  # -2 tells SNES to compute the matrix once and never rebuild it
        "snes_lag_preconditioner": -2,
    }

    problem = LinearVariationalProblem(a_I, L_I, solution_dq)
    solver = LinearVariationalSolver(problem, solver_parameters=int_solver_params)

    PETSc.Sys.Print("Cached vertical integral solver")

    return solver, integrand_placeholder, solution_dq

def compute_vertical_integral(integrand, main_func_space):
    """
    Computes the indefinite vertical integral of an arbitrary UFL integrand
    from z=0 to z. Solves the ODE: dI/dz = integrand with I(z=0) = 0.
    """

    solver, integrand_placeholder, solution_dq = _get_vertical_integral_solver(main_func_space)
    integrand_placeholder.interpolate(integrand) # Write the integrand into the placeholder function

    # Used for logging to profile how long each operation is taking
    solve_event = PETSc.Log.Event("DG_Extruded_Solve")

    with solve_event:
        solver.solve() # Solver writes to solution to solution_dq

    return Function(main_func_space).interpolate(solution_dq)

def kink_function(x, delta):
    """
    Returns a UFL expression for the kink profile kappa_delta(x).
    """
    # Define the bounds and values for each segment
    val_less_than_0 = Constant(0)
    val_lower_mid = 0.5 * (2 * x) ** delta
    val_upper_mid = 1 - 0.5 * (2 * (1 - x)) ** delta
    val_greater_than_1 = Constant(1)

    # Build the nested conditionals (evaluated from outside in)
    return conditional(x <= 0.0, val_less_than_0,
           conditional(x <= 0.5, val_lower_mid,
           conditional(x <= 1.0, val_upper_mid, val_greater_than_1)))

def scaled_kink(x, delta, left_val, right_val, kink_width, kink_centre):
    return (right_val - left_val) * kink_function((x-kink_centre)/kink_width + 0.5, delta) + left_val

def plot_func_slice(func : Function):
    file_name = "temp_func.pvd"
    func_name = "Provided Function"
    func.rename(func_name)

    # Temporarily save function as a pvd file
    outfile = VTKFile(file_name)
    outfile.write(func)

    # Get the current MPI process running this method
    rank = MPI.COMM_WORLD.Get_rank()

    # Only let the main process plot the figure
    if rank != 0:
        return

    # Load file and create slice
    multiblock = pv.read(file_name)
    mesh = multiblock.combine()
    slice_yz = mesh.slice(normal='x', origin=(500e3, 500e3, 10000))

    # Do actual plotting
    plotter = pv.Plotter()
    plotter.add_mesh(
        slice_yz,
        scalars=func_name,
        cmap="viridis",
        show_edges=True
    )
    plotter.camera_position = 'yz'
    plotter.set_scale(zscale=25)
    plotter.show()


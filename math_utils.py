from firedrake import *
from mpi4py import MPI
import pyvista as pv

def compute_vertical_integral(integrand, main_func_space):
    """
    Computes the indefinite vertical integral of an arbitrary UFL integrand
    from z=0 to z. Solves the ODE: dI/dz = integrand with I(z=0) = 0.
    """
    I_trial = TrialFunction(main_func_space)
    W_test = TestFunction(main_func_space)

    a_I = I_trial.dx(2) * W_test * dx
    L_I = integrand * W_test * dx

    bc_I = DirichletBC(main_func_space, Constant(0.0), "bottom")
    I_func = Function(main_func_space)

    # Maximally efficient solver for Extruded column integrals
    int_solver_params = {
        "ksp_type": "gmres",        # Fast iterative solver
        "ksp_rtol": 1e-7,           # Standard tolerance
        "pc_type": "bjacobi",       # Isolates matrix blocks to individual MPI ranks (Zero network communication)
        "sub_pc_type": "ilu",       # Local incomplete LU factorization (extremely fast)
        "sub_pc_factor_shift_type": "NONZERO"  # Prevents the "Diverged Linear Solve" zero-diagonal crash!
    }

    solve(a_I == L_I, I_func, bcs=[bc_I], solver_parameters=int_solver_params)

    return I_func

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


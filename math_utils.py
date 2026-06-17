from firedrake import *
from mpi4py import MPI
import pyvista as pv
import numpy as np
from scipy.interpolate import CubicSpline
from firedrake import Function, interpolate, SpatialCoordinate


def compute_vertical_integral(integrand, func_space):
    """
    Computes the vertical integral int_{z_bottom}^z f(z') dz' using SciPy.
    Uses CubicSpline antiderivatives to maintain high-order accuracy
    compatible with higher-degree Firedrake function spaces.
    """
    mesh = func_space.mesh()

    # 1. Extract exact DoF coordinates and values
    z_expr = SpatialCoordinate(mesh)[2]

    z_fd = Function(func_space).interpolate(z_expr)
    f_fd = Function(func_space).interpolate(integrand)

    z_data = z_fd.dat.data_ro
    f_data = f_fd.dat.data_ro

    # 2. Sort the data strictly by the z-coordinate
    sort_idx = np.argsort(z_data)
    z_sorted = z_data[sort_idx]
    f_sorted = f_data[sort_idx]

    # 3. Collapse into a strict 1D vertical profile
    z_rounded = np.round(z_sorted, decimals=8)
    _, unique_idx = np.unique(z_rounded, return_index=True)

    z_1d = z_sorted[unique_idx]
    f_1d = f_sorted[unique_idx]

    # 4. Fit a Cubic Spline to the 1D integrand
    # CubicSpline generates piecewise 3rd-degree polynomials
    spline = CubicSpline(z_1d, f_1d)

    # 5. Compute the exact analytical antiderivative of the spline
    # This automatically steps the interpolation up to a 4th-degree polynomial
    integral_spline = spline.antiderivative()

    # 6. Evaluate the antiderivative at the Firedrake DoF coordinates
    # We subtract integral_spline(z_1d[0]) to ensure the integral is exactly 0
    # at the bottom boundary of the mesh.
    integral_dofs = integral_spline(z_data) - integral_spline(z_1d[0])

    # 7. Assign the data to a new Firedrake Function
    result = Function(func_space, name="Vertical_Integral")
    result.dat.data[:] = integral_dofs

    return result

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

def plot_func_slice(func : Function, z_scale=50):
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

    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    slice_yz = mesh.slice(normal='x', origin=(x_mid, y_mid, z_mid))

    # Do actual plotting
    plotter = pv.Plotter()
    plotter.add_mesh(
        slice_yz,
        scalars=func_name,
        cmap="viridis",
        show_edges=True
    )
    plotter.camera_position = 'yz'
    plotter.set_scale(zscale=z_scale)
    plotter.show()


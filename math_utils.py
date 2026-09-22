from firedrake import *
from mpi4py import MPI
import numpy as np
from scipy.interpolate import CubicSpline
from firedrake import Function, interpolate, SpatialCoordinate

# this will only work with extruded meshes, as MPI ranks only take vertical chunks and do not do any splitting in the z-direction
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

def smooth_max(a, b, smoothing):
    if smoothing <= 0:
        return max_value(a, b)
    return max_value(a, b) + smoothing * ln(1 + exp(-abs(a - b) / smoothing))

def scaled_kink(x, delta, left_val, right_val, kink_width, kink_centre):
    return (right_val - left_val) * kink_function((x-kink_centre)/kink_width + 0.5, delta) + left_val

def relative_error(exact, numerical : Function, norm_type='L2'):
    """Relative error norm between exact and numerical, both living on numerical's mesh.

    The solver pins psi only up to an additive constant, so the mean offset between the two
    is removed before the norm is taken.
    """
    # High enough that integrating the squared error of a Q_p field against a
    # non-polynomial exact solution is not itself what sets the floor.
    degree = 3 * numerical.function_space().ufl_element().embedded_superdegree + 6
    dx_q = dx(domain=numerical.function_space().mesh(), degree=degree)

    def mean_removed(expr):
        return expr - assemble(expr * dx_q) / volume

    def norm_of(expr):
        squared = expr ** 2
        if norm_type == 'H1':
            squared += dot(grad(expr), grad(expr))
        elif norm_type != 'L2':
            raise ValueError(f"unsupported norm_type {norm_type!r}")
        return np.sqrt(assemble(squared * dx_q))

    volume = assemble(Constant(1) * dx_q)

    return float(norm_of(mean_removed(numerical - exact)) / norm_of(mean_removed(exact)))

def get_global_extrema(func : Function):
    with func.dat.vec_ro as v:
        global_min = v.min()[1]
        global_max = v.max()[1]
    return global_min, global_max

def get_global_min(func : Function):
    with func.dat.vec_ro as v:
        global_min = v.min()[1]
    return global_min

def get_global_max(func : Function):
    with func.dat.vec_ro as v:
        global_max = v.max()[1]
    return global_max

def get_regional_extrema(func : Function, in_region):
    V = func.function_space()
    x,y,z = SpatialCoordinate(V.mesh())
    f = Function(V)
    in_region = in_region(x,y,z)

    max_expr = conditional(in_region, func, -1e30)
    max = get_global_max(f.interpolate(max_expr))

    min_expr = conditional(in_region, func, 1e30)
    min = get_global_min(f.interpolate(min_expr))

    return min, max
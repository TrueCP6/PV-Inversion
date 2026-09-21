from firedrake import *
from mpi4py import MPI
import pyvista as pv
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

CROSS_MESH_CHUNKS = 16

def chunked_interpolate(source : Function, target_func_space, chunks : int = CROSS_MESH_CHUNKS):
    """Cross-mesh interpolate source into target_func_space a slice of the target's dofs
    at a time, giving the same result as a plain interpolate for a fraction of the memory.

    Firedrake's cross-mesh interpolation builds a VertexOnlyMesh over the target space's
    dofs, and VertexOnlyMesh gathers every target point onto every rank (an Allgatherv in
    mesh._parent_mesh_embedding, plus a point location run on the full global array). That
    replication costs roughly 200 bytes per global target dof on each rank, so it scales
    with target dofs times ranks - interpolating a 5M dof solution onto a 64M dof reference
    across 40 ranks needs hundreds of GB, none of it the actual data.

    Interpolating a slice at a time divides that replication by the chunk count. The parent
    mesh's rtree is cached across the calls, so the point location itself is not repeated,
    only split up - the measured cost is under a quarter of the interpolation time, which
    is itself small next to the solve that produced source.
    """
    source_mesh = source.function_space().mesh()
    target_mesh = target_func_space.mesh()

    # Nothing to gain unless this is genuinely the cross-mesh path, and the slice-wise dof
    # write below only makes sense for a scalar space whose dofs are point evaluations.
    if (chunks <= 1 or target_mesh is source_mesh or target_func_space.value_shape != ()
            or not target_func_space.finat_element.has_pointwise_dual_basis):
        return Function(target_func_space).interpolate(source)

    coord_space = VectorFunctionSpace(target_mesh, target_func_space.ufl_element())
    coords = assemble(interpolate(target_mesh.coordinates, coord_space)).dat.data_ro.reshape(
        -1, target_mesh.geometric_dimension)

    result = Function(target_func_space)
    # VertexOnlyMesh is collective, so every rank must make the same number of calls. Each
    # rank splits its own dofs into that many slices, contributing an empty one where it
    # holds fewer dofs than chunks.
    bounds = np.linspace(0, len(coords), chunks + 1).astype(int)
    for low, high in zip(bounds, bounds[1:]):
        vom = VertexOnlyMesh(source_mesh, coords[low:high], redundant=False)
        values = assemble(interpolate(source, FunctionSpace(vom, "DG", 0)))
        # The points were redistributed to whichever rank owns the containing cell, so come
        # back through the input ordering to line them up with coords[low:high] again.
        values = assemble(interpolate(values, FunctionSpace(vom.input_ordering, "DG", 0)))
        result.dat.data[low:high] = values.dat.data_ro

    return result

def relative_error(exact, numerical : Function, norm_type='L2'):
    """Relative error norm between exact and numerical, cross-mesh-interpolating the
    coarser onto the finer's function space if they differ.
    """
    # Make sure exact is not a plain UFL expression
    if not isinstance(exact, Function):
        exact = Function(numerical.function_space()).interpolate(exact)

    # If the two function spaces are not the same, interpolate one onto the other's mesh
    if exact.function_space() != numerical.function_space():
        if exact.function_space().dim() > numerical.function_space().dim():
            target_func_space = exact.function_space()
            numerical = chunked_interpolate(numerical, target_func_space)
        else:
            target_func_space = numerical.function_space()
            exact = chunked_interpolate(exact, target_func_space)
    else:
        target_func_space = exact.function_space()

    # Define a mesh-specific measure on the target mesh to prevent integration ambiguity
    dx_target = dx(domain=target_func_space.mesh())

    # calculate mean offset between numerical and analytical solutions, as we don't know what constant the solver added to psi
    total_offset = assemble((numerical - exact) * dx_target)
    volume = assemble(Constant(1) * dx_target)
    mean_offset = total_offset / volume

    shifted = Function(target_func_space)

    # shift the numerical solution by that constant we have worked out
    shifted.assign(numerical)
    shifted.dat.data[:] -= mean_offset

    absolute_error = errornorm(exact, shifted, norm_type=norm_type)

    # compute mean of exact solution
    exact_mean = assemble(exact * dx_target) / volume
    # shift exact solution, to prevent similar problem to before
    shifted.assign(exact)
    shifted.dat.data[:] -= exact_mean

    exact_norm = norm(shifted, norm_type=norm_type)

    return absolute_error / exact_norm

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
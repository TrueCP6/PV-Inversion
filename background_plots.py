import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake import PointEvaluator, Function
import math_utils
from parameters import *
from barnes_atmosphere import *
from domain_builder import *
from solver import *
from plot_utils import apply_style
from derived_quantities import *

# Global parameters for plot styling
apply_style()

def get_global_mesh_bounds(mesh):
    """
    Helper function to safely compute the global bounding box of a 3D mesh
    across all MPI ranks. Returns ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    """
    comm = mesh.comm
    coords = mesh.coordinates.dat.data_ro

    def safe_min(arr, col):
        return arr[:, col].min() if arr.shape[0] > 0 else np.inf

    def safe_max(arr, col):
        return arr[:, col].max() if arr.shape[0] > 0 else -np.inf

    bounds = []
    for dim in range(3):
        local_min = safe_min(coords, dim)
        local_max = safe_max(coords, dim)
        g_min = comm.allreduce(local_min, op=MPI.MIN)
        g_max = comm.allreduce(local_max, op=MPI.MAX)
        bounds.append((g_min, g_max))

    return bounds


def plot_function_vs_z(f, plot_title, x_title, x_coord=None, y_coord=None, num_points=200):
    """
    Evaluates and plots a 3D Firedrake function along the z-axis.
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # 1. Calculate global bounds safely via helper function
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_global_mesh_bounds(mesh)

    # Default to the global mesh bounding-box center for x and y if not provided
    if x_coord is None:
        x_coord = (x_max + x_min) / 2.0
    if y_coord is None:
        y_coord = (y_max + y_min) / 2.0

    # 2. Generate points along the z-axis
    z_values = np.linspace(z_min, z_max, num_points)
    points = np.array([[x_coord, y_coord, z] for z in z_values])

    # 3. Evaluate the function
    evaluator = PointEvaluator(mesh, points)
    f_values = evaluator.evaluate(f)

    # 4. Plot only on the root rank
    if comm.rank == 0:
        plt.figure(figsize=(3.15, 4.5))
        plt.plot(f_values, z_values, color='#004488', linestyle='-', linewidth=1.5)

        plt.xlabel(x_title)
        plt.ylabel(r'$z$ [\unit{\meter}]')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

        lwr_case = plot_title.replace(" ", "_").lower()
        plt.savefig(f"tex/plots/{lwr_case}.pdf", bbox_inches='tight')
        plt.close()


def plot_slice_heatmap(f, plot_title, cbar_title, levels, normal_dir='x', slice_coord=None,
                       num_points_h=200, num_points_v=200, figsize=(3.15*2, 4.0), cbar_min = None, cbar_max = None,
                       vector_field=None, quiver_density=15, highlight_level=None):
    """
    Evaluates and plots a 2D heatmap (with contours) of a 3D Firedrake function
    along a plane normal to the specified axis (x, y, or z).

    Parameters
    ----------
    f : firedrake.Function
        The function to evaluate.
    plot_title : str
        The title for the resulting plot.
    cbar_title : str
        The label for the colorbar (function value).
    levels : int or array-like
        Determines the number and positions of the contour lines.
    normal_dir : str, optional
        The axis normal to the slice plane ('x', 'y', or 'z'). Defaults to 'x'.
    slice_coord : float, optional
        The coordinate of the slice plane along the normal_dir.
        Defaults to the global mesh center for that axis.
    num_points_h : int, optional
        Number of sampling points along the horizontal axis of the plot. Defaults to 200.
    num_points_v : int, optional
        Number of sampling points along the vertical axis of the plot. Defaults to 200.
    figsize : tuple, optional
        Figure size.
    vector_field : (firedrake.Function, firedrake.Function), optional
        Components (f_h, f_v) aligned with the plot's horizontal and vertical axes.
        When given, unit-vector arrows showing their direction are overlaid on a
        coarser grid on top of the heatmap.
    quiver_density : int, optional
        Number of arrows per axis when vector_field is given. Defaults to 15.
    highlight_level : float, optional
        A single contour value to redraw as a heavy solid line, picking it out of
        the surrounding contours (e.g. the dynamical tropopause).
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # 1. Get global bounds using the helper
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_global_mesh_bounds(mesh)

    normal_dir = normal_dir.lower()
    if normal_dir not in ['x', 'y', 'z']:
        raise ValueError("normal_dir must be 'x', 'y' or 'z'.")

    # Axes 'x' and 'y' are displayed in kilometers; 'z' (altitude) stays in meters.
    def axis_label_and_scale(axis):
        if axis in ('x', 'y'):
            return rf'${axis}$ [\unit{{\kilo\meter}}]', 1e-3
        return r'$z$ [\unit{\meter}]', 1.0

    # 2. Determine bounds, labels, and slice coordinate based on normal direction
    if normal_dir == 'x':
        if slice_coord is None:
            slice_coord = (x_max + x_min) / 2.0
        h_min, h_max = y_min, y_max
        v_min, v_max = z_min, z_max
        h_label, h_scale = axis_label_and_scale('y')
        v_label, v_scale = axis_label_and_scale('z')
    elif normal_dir == 'y':
        if slice_coord is None:
            slice_coord = (y_max + y_min) / 2.0
        h_min, h_max = x_min, x_max
        v_min, v_max = z_min, z_max
        h_label, h_scale = axis_label_and_scale('x')
        v_label, v_scale = axis_label_and_scale('z')
    else:  # normal_dir == 'z'
        if slice_coord is None:
            slice_coord = (z_max + z_min) / 2.0
        h_min, h_max = x_min, x_max
        v_min, v_max = y_min, y_max
        h_label, h_scale = axis_label_and_scale('x')
        v_label, v_scale = axis_label_and_scale('y')

    # 3. Flatten a grid and insert the constant slice coordinate, in mesh (x, y, z) order
    def slice_points(H, V):
        if normal_dir == 'x':
            return np.column_stack((np.full(H.size, slice_coord), H.flatten(), V.flatten()))
        elif normal_dir == 'y':
            return np.column_stack((H.flatten(), np.full(H.size, slice_coord), V.flatten()))
        else:  # normal_dir == 'z'
            return np.column_stack((H.flatten(), V.flatten(), np.full(H.size, slice_coord)))

    # 4. Generate meshgrid for the horizontal (H) and vertical (V) axes of the plot
    h_values = np.linspace(h_min, h_max, num_points_h)
    v_values = np.linspace(v_min, v_max, num_points_v)
    H, V = np.meshgrid(h_values, v_values)

    # 5. Evaluate the function using PointEvaluator
    evaluator = PointEvaluator(mesh, slice_points(H, V))
    f_values_flat = evaluator.evaluate(f)

    # 5b. Evaluate the (optional) direction field on a coarser grid, as unit vectors
    if vector_field is not None:
        Hq, Vq = np.meshgrid(
            np.linspace(h_min, h_max, quiver_density),
            np.linspace(v_min, v_max, quiver_density),
        )
        qpoints = slice_points(Hq, Vq)
        f_h, f_v = vector_field
        Uh = PointEvaluator(mesh, qpoints).evaluate(f_h).reshape(Hq.shape)
        Uv = PointEvaluator(mesh, qpoints).evaluate(f_v).reshape(Hq.shape)
        speed = np.hypot(Uh, Uv)
        speed[speed == 0] = 1.0
        Uh, Uv = Uh / speed, Uv / speed

    # 6. Plot only on the root rank
    if comm.rank == 0:
        # Reshape evaluated 1D array back to 2D meshgrid shape
        F = f_values_flat.reshape(H.shape)

        plt.figure(figsize=figsize)

        H_plot, V_plot = H * h_scale, V * v_scale

        heatmap = plt.pcolormesh(H_plot, V_plot, F, cmap='viridis', shading='auto', rasterized=True, vmin=cbar_min, vmax=cbar_max)

        # Superimposed solid contours
        plt.contour(H_plot, V_plot, F, levels=levels, colors='black', linewidths=0.5, alpha=0.5)

        if highlight_level is not None:
            plt.contour(H_plot, V_plot, F, levels=[highlight_level], colors='black',
                        linewidths=1.5, linestyles='solid')

        if vector_field is not None:
            plt.quiver(Hq * h_scale, Vq * v_scale, Uh, Uv, color='white', pivot='mid', alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label(cbar_title)

        plt.xlabel(h_label)
        plt.ylabel(v_label)

        plt.tight_layout()

        # Save as PDF
        lwr_case = plot_title.replace(" ", "_").lower()
        plt.savefig(f"tex/plots/{lwr_case}.pdf", bbox_inches='tight')
        plt.close()

def multislice(func : Function, title : str, cbar_title : str, levels, normals ='xyz', cbar_bound_region = None,
               highlight_level = None):
    if cbar_bound_region is None:
        min, max = math_utils.get_global_extrema(func)
    else:
        min, max = math_utils.get_regional_extrema(func, cbar_bound_region)

    if max - min >= 1:
        min, max = np.floor(min), np.ceil(max)

    if 'z' in normals:
        plot_slice_heatmap(
            func,
            title + " Surface",
            cbar_title, levels,
            normal_dir='z', slice_coord=0,
            cbar_min=min, cbar_max=max,
            highlight_level=highlight_level
        )

    if 'x' in normals:
        plot_slice_heatmap(
            func,
            title + " X",
            cbar_title, levels,
            normal_dir='x',
            cbar_min=min, cbar_max=max,
            highlight_level=highlight_level
        )

    if 'y' in normals:
        plot_slice_heatmap(
            func,
            title + " Y",
            cbar_title, levels,
            normal_dir='y',
            cbar_min=min, cbar_max=max,
            highlight_level=highlight_level
        )

def main():
    N = 40
    solver_params = SolverParams(
        nx=N, ny=N, nz=N,
        check_flux=False
    )
    phys_params = PhysicalParams()

    domain = DomainBuilder(solver_params, phys_params)
    func_space = domain.func_space()

    atmos = BarnesAtmosphere(domain)

    solver = Solver(atmos, True)
    solver.solve_psi()
    derived = ResolvedAtmosphere(solver.psi_soln, atmos)

    plot_slice_heatmap(
        derived.horizontal_wind_speed(),
        "Surface Wind",
        r"$\left|\mathbf{u}\right|$ [\unit{\meter \per \second}]",
        levels=np.arange(0, 3, 0.2),
        normal_dir="z",
        slice_coord=0,
        vector_field=(derived.u(), derived.v())
    )

    multislice(
        Function(func_space).interpolate(atmos.ertel_pv() * 1e6),
        "EPV",
        r"$Q$ [\unit{PVU}]",
        levels=np.arange(-5, 0, 0.5),
        normals='xy',
        highlight_level=-1.5  # dynamical tropopause
    )

    PETSc.Sys.Print("Saved EPV plots")

    plot_function_vs_z(
        atmos.N_bar(),
        "Reference Brunt–Väisälä Frequency",
        r"$\overline{N}$ [\unit{\per\second}]"
    )

    plot_function_vs_z(
        atmos.rho_bar(),
        "Reference Density Profile",
        r"$\overline{\rho}$ [\unit{\kg\per\meter\cubed}]"
    )

    plot_function_vs_z(
        Function(func_space).interpolate(atmos.p_bar() / 1e2),
        "Reference Pressure Profile",
        r"$\overline{p}$ [\unit{\hecto\pascal}]"
    )

    plot_function_vs_z(
        atmos.theta_bar(),
        "Reference Potential Temperature Profile",
        r"$\overline{\theta}$ [\unit{\kelvin}]"
    )

    plot_slice_heatmap(
        Function(func_space).interpolate(atmos.u()),
        "Jet Stream",
        r"$\overline{u}$ [\unit{\meter\per\second}]",
        levels=np.arange(0, 35, 5),
        normal_dir='y'
    )

    # todo add reference in writeup to geostrophic vorticity being zero with the jet stream

    multislice(
        derived.geostrophic_vorticity(),
        "Geostrophic Vorticity",
        r"$\zeta_g$ [\unit{\per\second}]",
        levels=10,
        normals='xyz'
    )

    multislice(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly",
        r"$\theta^*$ [\unit{\kelvin}/\unit{\celsius}]",
        levels=np.arange(-20, 20, 1),
        normals='xy',
    )

    multislice(
        derived.potential_temperature(),
        "Potential Temperature",
        r"$\theta$ [\unit{\kelvin}]",
        levels = np.arange(250, 800, 10),
        normals='xy'
    )

    multislice(
        derived.pressure_anomaly_hpa(),
        "Pressure Anomaly",
        r"$p^*$ [\unit{\hecto\pascal}]",
        levels=np.arange(-10, 10, 1),
        normals='xyz'
    )

    multislice(
        derived.pressure_hpa(),
        "Pressure",
        r"$p$ [\unit{\hecto\pascal}]",
        levels=np.arange(-900, 1100, 1),
        normals='z',
        cbar_bound_region=lambda x,y,z: z<10
    )

    multislice(
        derived.temperature_anomaly(),
        "Temperature Anomaly",
        r"$T^*$ [\unit{\kelvin}/\unit{\celsius}]",
        levels=np.arange(-20, 20, 1),
        normals='xyz'
    )

if __name__ == "__main__":
    main()
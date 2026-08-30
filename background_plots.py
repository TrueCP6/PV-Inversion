import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake import PointEvaluator, Function
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
                       num_points_h=200, num_points_v=200, figsize=(3.15*2, 4.0)):
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
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # 1. Get global bounds using the helper
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_global_mesh_bounds(mesh)

    normal_dir = normal_dir.lower()
    if normal_dir not in ['x', 'y', 'z']:
        raise ValueError("normal_dir must be 'x', 'y' or 'z'.")

    # 2. Determine bounds, labels, and slice coordinate based on normal direction
    if normal_dir == 'x':
        if slice_coord is None:
            slice_coord = (x_max + x_min) / 2.0
        h_min, h_max = y_min, y_max
        v_min, v_max = z_min, z_max
        h_label, v_label = r'$y$ [\unit{\meter}]', r'$z$ [\unit{\meter}]'
    elif normal_dir == 'y':
        if slice_coord is None:
            slice_coord = (y_max + y_min) / 2.0
        h_min, h_max = x_min, x_max
        v_min, v_max = z_min, z_max
        h_label, v_label = r'$x$ [\unit{\meter}]', r'$z$ [\unit{\meter}]'
    else:  # normal_dir == 'z'
        if slice_coord is None:
            slice_coord = (z_max + z_min) / 2.0
        h_min, h_max = x_min, x_max
        v_min, v_max = y_min, y_max
        h_label, v_label = r'$x$ [\unit{\meter}]', r'$y$ [\unit{\meter}]'

    # 3. Generate meshgrid for the horizontal (H) and vertical (V) axes of the plot
    h_values = np.linspace(h_min, h_max, num_points_h)
    v_values = np.linspace(v_min, v_max, num_points_v)
    H, V = np.meshgrid(h_values, v_values)

    # 4. Flatten the grid and insert the constant slice coordinate
    if normal_dir == 'x':
        points = np.column_stack((np.full(H.size, slice_coord), H.flatten(), V.flatten()))
    elif normal_dir == 'y':
        points = np.column_stack((H.flatten(), np.full(H.size, slice_coord), V.flatten()))
    else:  # normal_dir == 'z'
        points = np.column_stack((H.flatten(), V.flatten(), np.full(H.size, slice_coord)))

    # 5. Evaluate the function using PointEvaluator
    evaluator = PointEvaluator(mesh, points)
    f_values_flat = evaluator.evaluate(f)

    # 6. Plot only on the root rank
    if comm.rank == 0:
        # Reshape evaluated 1D array back to 2D meshgrid shape
        F = f_values_flat.reshape(H.shape)

        plt.figure(figsize=figsize)

        heatmap = plt.pcolormesh(H, V, F, cmap='viridis', shading='auto', rasterized=True)

        # Superimposed solid contours
        plt.contour(H, V, F, levels=levels, colors='black', linewidths=0.5, alpha=0.5)

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
        "surface_wind",
        r"$\left|\mathbf{u}\right|$ [\unit{\meter \per \second}]",
        levels=np.arange(0, 3, 0.2),
        normal_dir="z",
        slice_coord=0
    )

    epv_cbar_title = r"$Q$ [\unit{PVU}]"
    epv_plot_size = (3.15 * 2, 3)
    epv = Function(func_space).interpolate(atmos.ertel_pv() * 1e6)

    plot_slice_heatmap(
        epv,
        "EPV_X",
        epv_cbar_title,
        levels=np.arange(-5, 0, 0.5),
        normal_dir='x',
        figsize=epv_plot_size
    )

    plot_slice_heatmap(
        epv,
        "EPV_Y",
        epv_cbar_title,
        levels=np.arange(-5, 0, 0.5),
        normal_dir='y',
        figsize=epv_plot_size
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

    plot_slice_heatmap(
        Function(func_space).interpolate(atmos.geostrophic_vorticity()),
        "Background Geostrophic Vorticity",
        r"$\overline{\zeta}_g$ [\unit{\per\second}]",
        normal_dir='y',
        levels=10
    )

    plot_slice_heatmap(
        derived.geostrophic_vorticity(),
        "Geostrophic Vorticity X",
        r"$\zeta_g$ [\unit{\per\second}]",
        normal_dir='x',
        levels=10,
    )

    plot_slice_heatmap(
        derived.geostrophic_vorticity(),
        "Geostrophic Vorticity Y",
        r"$\zeta_g$ [\unit{\per\second}]",
        normal_dir='y',
        levels=10,
    )


    plot_slice_heatmap(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly X",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(-20, 20, 1),
        normal_dir='x',
    )

    plot_slice_heatmap(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly Y",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(-20, 20, 1),
        normal_dir='y',
    )

    plot_slice_heatmap(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly X",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(-20, 20, 1),
        normal_dir='x',
    )

    plot_slice_heatmap(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly Y",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(-20, 20, 1),
        normal_dir='y',
    )

    plot_slice_heatmap(
        derived.potential_temperature(),
        "Potential Temperature X",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(250, 800, 10),
        normal_dir='x',
    )

    plot_slice_heatmap(
        derived.potential_temperature(),
        "Potential Temperature Y",
        r"$\theta^*$ [\unit{\kelvin}]",
        levels=np.arange(250, 800, 10),
        normal_dir='y',
    )

if __name__ == "__main__":
    main()
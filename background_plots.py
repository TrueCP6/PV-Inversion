import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake import PointEvaluator, Function
from parameters import *
from barnes_atmosphere import *
from domain_builder import *

# 1. Global parameters for academic styling
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX to render text
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Computer Modern"],  # Match default LaTeX font
    "text.latex.preamble": r"\usepackage{siunitx}",
    "font.size": 11,  # Match typical LaTeX document font size
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,  # Thicker axes frames
    "xtick.direction": "in",  # Inward facing ticks
    "ytick.direction": "in"
})


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
        plt.savefig(f"tex/{lwr_case}.pdf", bbox_inches='tight')
        plt.close()


def plot_yz_heatmap(f, plot_title, cbar_title, x_coord=None, num_points_y=200, num_points_z=200):
    """
    Evaluates and plots a 2D heatmap (with contours) of a 3D Firedrake function
    along the y-z plane at a constant x coordinate.

    Parameters
    ----------
    f : firedrake.Function
        The function to evaluate.
    plot_title : str
        The title for the resulting plot.
    cbar_title : str
        The label for the colorbar (function value).
    x_coord : float, optional
        The x-coordinate of the vertical plane. Defaults to the global mesh center.
    num_points_y : int, optional
        Number of sampling points along the y-axis. Defaults to 100.
    num_points_z : int, optional
        Number of sampling points along the z-axis. Defaults to 100.
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # 1. Get global bounds using the helper
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_global_mesh_bounds(mesh)

    if x_coord is None:
        x_coord = (x_max + x_min) / 2.0

    # 2. Generate meshgrid for Y and Z
    y_values = np.linspace(y_min, y_max, num_points_y)
    z_values = np.linspace(z_min, z_max, num_points_z)
    Y, Z = np.meshgrid(y_values, z_values)

    # 3. Flatten the grid and append constant x to build an (N, 3) array of coordinates
    points = np.column_stack((
        np.full(Y.size, x_coord),
        Y.flatten(),
        Z.flatten()
    ))

    # 4. Evaluate the function using PointEvaluator
    evaluator = PointEvaluator(mesh, points)
    f_values_flat = evaluator.evaluate(f)

    # 5. Plot only on the root rank
    if comm.rank == 0:
        # Reshape evaluated 1D array back to 2D meshgrid shape
        F = f_values_flat.reshape(Y.shape)

        # Make figure slightly wider than the 1D profile to accommodate the colorbar
        plt.figure(figsize=(3.15*2, 4.0))

        heatmap = plt.pcolormesh(Y, Z, F, cmap='viridis', shading='auto', rasterized=True)

        # Superimposed solid contours
        levels = np.arange(0, 35, 5)
        plt.contour(Y, Z, F, levels=levels, colors='black', linewidths=0.5, alpha=0.5)

        # Add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label(cbar_title)

        plt.xlabel(r'$y$ [\unit{\meter}]')
        plt.ylabel(r'$z$ [\unit{\meter}]')

        plt.tight_layout()

        # Save as PDF
        lwr_case = plot_title.replace(" ", "_").lower()
        plt.savefig(f"tex/{lwr_case}.pdf", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    solver_params = SolverParams()
    phys_params = PhysicalParams()

    domain_builder = DomainBuilder(solver_params, phys_params)
    mesh = domain_builder.mesh()
    V = domain_builder.func_space()

    atmos = BarnesAtmosphere(mesh, V, phys_params)

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
        Function(V).interpolate(atmos.p_bar() / 1e2),
        "Reference Pressure Profile",
        r"$\overline{p}$ [\unit{\hecto\pascal}]"
    )

    plot_function_vs_z(
        atmos.theta_bar(),
        "Reference Potential Temperature Profile",
        r"$\overline{\theta}$ [\unit{\kelvin}]"
    )

    plot_yz_heatmap(
        Function(V).interpolate(atmos.u()),
        "Jet Stream",
        r"$u$ [\unit{\meter\per\second}]"
    )
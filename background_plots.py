import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake import PointEvaluator
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

def plot_function_vs_z(f, plot_title, x_title, x_coord=None, y_coord=None, num_points=200):
    """
    Evaluates and plots a 3D Firedrake function along the z-axis.

    Parameters
    ----------
    f : firedrake.Function
        The function to evaluate.
    plot_title : str
        The title for the resulting plot.
    x_title : str
        The label for the x-axis (function value).
    x_coord : float, optional
        The x-coordinate of the vertical line. Defaults to the global mesh center.
    y_coord : float, optional
        The y-coordinate of the vertical line. Defaults to the global mesh center.
    num_points : int, optional
        Number of sampling points along the z-axis. Defaults to 100.
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # Extract read-only access to local mesh coordinates
    coords = mesh.coordinates.dat.data_ro

    # Helper to prevent ValueError on MPI ranks that might own zero mesh cells
    def safe_min(arr, col):
        return arr[:, col].min() if arr.shape[0] > 0 else np.inf

    def safe_max(arr, col):
        return arr[:, col].max() if arr.shape[0] > 0 else -np.inf

    # 1. Calculate global bounds safely across all MPI ranks
    local_z_min, local_z_max = safe_min(coords, 2), safe_max(coords, 2)
    z_min = comm.allreduce(local_z_min, op=MPI.MIN)
    z_max = comm.allreduce(local_z_max, op=MPI.MAX)

    # Default to the global mesh bounding-box center for x and y if not provided
    if x_coord is None:
        local_x_min, local_x_max = safe_min(coords, 0), safe_max(coords, 0)
        x_min = comm.allreduce(local_x_min, op=MPI.MIN)
        x_max = comm.allreduce(local_x_max, op=MPI.MAX)
        x_coord = (x_max + x_min) / 2.0

    if y_coord is None:
        local_y_min, local_y_max = safe_min(coords, 1), safe_max(coords, 1)
        y_min = comm.allreduce(local_y_min, op=MPI.MIN)
        y_max = comm.allreduce(local_y_max, op=MPI.MAX)
        y_coord = (y_max + y_min) / 2.0

    # 2. Generate points along the z-axis
    z_values = np.linspace(z_min, z_max, num_points)
    points = np.array([[x_coord, y_coord, z] for z in z_values])

    # 3. Evaluate the function using PointEvaluator
    # By default, redundant=True, meaning it uses the points from rank 0,
    # creates a VertexOnlyMesh under the hood, and broadcasts results back.
    evaluator = PointEvaluator(mesh, points)
    f_values = evaluator.evaluate(f)

    # 4. Plot only on the root rank to avoid duplicate windows in parallel runs
    if comm.rank == 0:
        # 2. Calculate exact figure size for your document
        # A4 width (8.27in) - margins (1.97in) = ~6.3 inches of text width.
        # (3.15, 4.5) represents exactly half the page width, perfect for 2-column or text wrap.
        plt.figure(figsize=(3.15, 4.5))

        # 3. Adjusted plotting colors and weights
        # Using a darker, colorblind-friendly blue ('#004488') instead of default 'b'
        plt.plot(f_values, z_values, color='#004488', linestyle='-', linewidth=1.5)

        plt.xlabel(x_title)
        plt.ylabel(r'$z$ [\unit{\meter}]')

        # 5. Softer grid
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()

        # 6. Save as PDF (Vector format)
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
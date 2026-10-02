from hash_seed import use_same_hash
use_same_hash()
import argparse
import json
from pathlib import Path
import numpy as np
import sweep
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
import contourpy
from mpi4py import MPI
from firedrake import PointEvaluator, Function, And
import math_utils
from parameters import *
from barnes_atmosphere import *
from domain_builder import *
from diagnostic_solver import *
from plot_utils import apply_style, FIGURE_SIZE
from derived_quantities import *
import petsctools

# Global parameters for plot styling
apply_style()

# Appended to every plot's file name, so a checkpoint's plots don't overwrite the background's
FILE_SUFFIX = ""

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
        plt.plot(f_values, z_values / 1e3, color='#004488', linestyle='-', linewidth=1.5)

        plt.xlabel(x_title)
        plt.ylabel(r'$z$ [\unit{\kilo\meter}]')
        plt.grid(True, linestyle=':', alpha=0.5)
        # Fixed margins, not tight_layout: every profile gets the same axes box; the tight bbox only crops whitespace
        plt.subplots_adjust(left=0.25, right=0.93, bottom=0.12, top=0.97)

        lwr_case = plot_title.replace(" ", "_").lower()
        plt.savefig(f"tex/plots/{lwr_case}{FILE_SUFFIX}.pdf", bbox_inches='tight')
        plt.close()


def plot_slice_heatmap(f, plot_title, cbar_title, levels, normal_dir='x', slice_coord=None,
                       num_points_h=200, num_points_v=200, figsize=(3.15*2, 4.0), cbar_min = None, cbar_max = None,
                       vector_field=None, quiver_density=15, highlight_level=None, ax=None, bounds=None):
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
    ax : matplotlib.axes.Axes, optional
        Draw into this axis (rank 0 only) and return the heatmap, leaving the colorbar
        and saving to the caller. Otherwise a standalone figure is saved.
    bounds : ((x_min, x_max), (y_min, y_max), (z_min, z_max)), optional
        The window to plot. Defaults to the whole domain.
    """
    mesh = f.function_space().mesh()
    comm = mesh.comm

    # 1. Get global bounds using the helper
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds or get_global_mesh_bounds(mesh)

    normal_dir = normal_dir.lower()
    if normal_dir not in ['x', 'y', 'z']:
        raise ValueError("normal_dir must be 'x', 'y' or 'z'.")

    # All axes are displayed in kilometers
    def axis_label_and_scale(axis):
        return rf'${axis}$ [\unit{{\kilo\meter}}]', 1e-3

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

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=figsize)

        H_plot, V_plot = H * h_scale, V * v_scale

        heatmap = ax.pcolormesh(H_plot, V_plot, F, cmap='viridis', shading='auto', rasterized=True, vmin=cbar_min, vmax=cbar_max)

        # Superimposed solid contours
        ax.contour(H_plot, V_plot, F, levels=levels, colors='black', linewidths=0.5, alpha=0.5)

        if highlight_level is not None:
            # Heavy line on the lowest piece of the level spanning the full width (e.g. the tropopause, not a crossing aloft)
            lines = [l for l in contourpy.contour_generator(H_plot, V_plot, F, line_type='Separate').lines(highlight_level)
                     if np.isclose(l[:, 0].min(), H_plot.min()) and np.isclose(l[:, 0].max(), H_plot.max())]
            if lines:
                h, v = min(lines, key=lambda l: l[:, 1].mean()).T
                ax.plot(h, v, color='black', linewidth=1.5)

        if vector_field is not None:
            ax.quiver(Hq * h_scale, Vq * v_scale, Uh, Uv, color='white', pivot='mid', alpha=0.8)

        ax.set_xlabel(h_label)
        ax.set_ylabel(v_label)

        if not standalone:
            return heatmap

        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label(cbar_title)

        plt.tight_layout()

        # Save as PDF
        lwr_case = plot_title.replace(" ", "_").lower()
        plt.savefig(f"tex/plots/{lwr_case}{FILE_SUFFIX}.pdf", bbox_inches='tight')
        plt.close()

def multislice(func : Function, title : str, cbar_title : str, levels, normals ='xyz', cbar_bound_region = None,
               highlight_level = None):
    # Plot only the middle of the domain horizontally, and below z_top
    half_width, z_top = 2500e3, 20e3
    (x_min, x_max), (y_min, y_max), (z_min, _) = get_global_mesh_bounds(func.function_space().mesh())
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    bounds = ((cx - half_width, cx + half_width), (cy - half_width, cy + half_width), (z_min, z_top))

    # Colour scale from the plotted window unless told otherwise
    if cbar_bound_region is None:
        cbar_bound_region = lambda x, y, z: And(And(abs(x - cx) < half_width, abs(y - cy) < half_width), z < z_top)

    min, max = math_utils.get_regional_extrema(func, cbar_bound_region)

    if max - min >= 1:
        min, max = np.floor(min), np.ceil(max)

    # The surface is a standalone plot, on the same colour scale as the vertical sections
    if 'z' in normals:
        plot_slice_heatmap(
            func, title + " Surface", cbar_title, levels,
            normal_dir='z', slice_coord=0,
            cbar_min=min, cbar_max=max,
            highlight_level=highlight_level,
            bounds=bounds
        )

    # The vertical sections are stacked panels in the order given, sharing a single colorbar, in one file.
    # Height/width of each panel: flattened to show the domain's anisotropy
    aspects = {'x': 0.3, 'y': 0.3}
    panels = [n for n in normals if n in 'xy']
    if panels:
        fig = axes = None
        if func.function_space().mesh().comm.rank == 0:
            ratios = [aspects[n] for n in panels]
            # ~0.8 of the width is axes (the rest is the colorbar to the right), plus room for each panel's labels.
            # Any taller and constrained layout pads the gap between panels; any shorter and it narrows the axes
            fig, axes = plt.subplots(len(panels), 1, squeeze=False, layout='constrained', height_ratios=ratios,
                                     figsize=(FIGURE_SIZE[0], 0.8 * FIGURE_SIZE[0] * sum(ratios) + 0.6 * len(panels)))
            axes = axes[:, 0]
            for ax, r in zip(axes, ratios):
                ax.set_box_aspect(r)

        for i, n in enumerate(panels):
            heatmap = plot_slice_heatmap(
                func, title, cbar_title, levels,
                normal_dir=n,
                cbar_min=min, cbar_max=max,
                highlight_level=highlight_level,
                ax=axes[i] if axes is not None else None,
                bounds=bounds
            )
            if fig is not None and len(panels) > 1:
                axes[i].set_title(f"({'abc'[i]})")

        if fig is not None:
            # Spans most of the stacked panels' height; aspect (length / thickness) raised from the default 20 to keep it slim
            fig.colorbar(heatmap, ax=axes, location='right', shrink=0.8, aspect=25).set_label(cbar_title)
            lwr_case = title.replace(" ", "_").lower()
            fig.savefig(f"tex/plots/{lwr_case}{FILE_SUFFIX}.pdf", bbox_inches='tight')
            plt.close(fig)

def plot_alpha_sparsity(orders=(1, 2, 4), dofs_per_dir=9):
    """
    Spy plots of the matrix assembled from the bilinear form alpha, one panel per
    polynomial order, to show the non-zeros per row growing with p. Each panel's mesh
    is N = (dofs_per_dir - 1) / p cells a side, so every matrix has the same size.
    Kept tiny so individual entries remain visible. Built on rank 0 alone, so
    the dof numbering (and so the pattern) doesn't depend on the MPI partition.
    """
    if COMM_WORLD.rank != 0:
        return

    phys_params = PhysicalParams()
    fig, axes = plt.subplots(1, len(orders), figsize=(FIGURE_SIZE[0], FIGURE_SIZE[0] / len(orders)))
    for ax, p in zip(axes, orders):
        N = (dofs_per_dir - 1) // p
        # Same mesh as DomainBuilder.mesh, but on COMM_SELF
        mesh = ExtrudedMesh(RectangleMesh(N, N, phys_params.Lx, phys_params.Ly, quadrilateral=True, comm=COMM_SELF),
                            layers=N, layer_height=phys_params.H / N)
        solver_params = SolverParams(nx=N, ny=N, nz=N, polynomial_order=p)
        atmos = BarnesAtmosphere(DomainBuilder(solver_params, phys_params, mesh=mesh))
        a, _ = DiagnosticSolver(atmos, False)._specify_equation()
        indptr, indices, vals = assemble(a, form_compiler_parameters=solver_params.form_compiler_params).petscmat.getValuesCSR()
        A = scipy.sparse.csr_matrix((vals, indices, indptr))

        # Dense image rather than markers, so every entry fills its cell whatever the matrix size
        ax.spy(A.toarray(), cmap=matplotlib.colors.ListedColormap(['white', '#004488']))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(rf'$p={p}$, nnz/row $\approx {A.nnz / A.shape[0]:.0f}$')

    plt.tight_layout()
    plt.savefig("tex/plots/alpha_sparsity.pdf", bbox_inches='tight')
    plt.close()


def main():
    global FILE_SUFFIX

    parser = argparse.ArgumentParser(description='Plot the background state, or a checkpointed one')
    parser.add_argument('--checkpoint', metavar='H5_PATH',
                        help='Plot the state timestepping.py --backup saved here instead, '
                             'suffixing every file name with the checkpoint\'s')
    parser.add_argument('--params', metavar='JSON_PATH',
                        help='PhysicalParams overrides, e.g. param_sampler.py\'s optimised_*.json, '
                             'suffixing every file name with the JSON\'s')
    args = parser.parse_args()
    sweep.quiet_petsc() # PETSc reads the command line too, and would warn about --checkpoint

    # Must run before any other atmosphere is built: its throwaway atmospheres on rank 0 alone would
    # evict the other's lru_cache(maxsize=1) entries on that rank only, and the rebuild deadlocks MPI
    if not args.checkpoint:
        plot_alpha_sparsity()

    if args.checkpoint:
        from timestepping import load_checkpoint

        atmos, psi, q, t = load_checkpoint(args.checkpoint)
        FILE_SUFFIX = f"_{Path(args.checkpoint).stem}"
        PETSc.Sys.Print(f"Plotting the state at t = {t / 3600:g} h")

        derived = ResolvedAtmosphere(psi, atmos, q)
        epv = atmos.ertel_from_qgpv(q)
    else:
        N = 40
        solver_params = SolverParams(
            nx=N, ny=N, nz=N,
            check_flux=True
        )
        phys_params = PhysicalParams()
        if args.params:
            with open(args.params) as f:
                phys_params = PhysicalParams(**json.load(f))
            FILE_SUFFIX = f"_{Path(args.params).stem}"

        atmos = BarnesAtmosphere(DomainBuilder(solver_params, phys_params))
        PETSc.Sys.Print(f"Background Ro = {atmos.rossby_number()}")
        PETSc.Sys.Print(f"Frac of volume that has invalid EPV = {atmos.invalid_epv()}")

        solver = DiagnosticSolver(atmos, True)
        solver.solve_psi()
        derived = ResolvedAtmosphere(solver.psi_soln, atmos)
        q = atmos.q_init()
        epv = atmos.ertel_pv()

    cg_space = atmos.cg_space
    # Optimised states reach ~25 m/s and ~-50 hPa at the surface, far past the control's levels
    surf_wind_levels = np.arange(0, 100, 5) if args.params else np.arange(0, 10, 0.5)
    pres_ano_levels = np.arange(-100, 100, 5) if args.params else np.arange(-30, 30, 1)
    pres_levels = np.arange(800, 1100, 5) if args.params else np.arange(-900, 1100, 1)
    PETSc.Sys.Print(f"z_dyn = {derived.min_dyn_tropopause_height()}")
    PETSc.Sys.Print(f"Ro after inversion = {derived.rossby_number()}")
    PETSc.Sys.Print(f"Fr after inversion = {derived.froude_number()}")

    plot_slice_heatmap(
        derived.horizontal_wind_speed(),
        "Surface Wind",
        r"$\left|\mathbf{u}\right|$ [\unit{\meter \per \second}]",
        levels=surf_wind_levels,
        normal_dir="z",
        slice_coord=0,
        vector_field=(derived.u(), derived.v())
    )

    multislice(
        derived.horizontal_wind_speed(),
        "Wind Speed",
        r"$\left|\mathbf{u}\right|$ [\unit{\meter \per \second}]",
        levels=np.arange(0, 50, 5),
        normals="xy"
    )

    multislice(
        Function(cg_space).interpolate(epv * 1e6),
        "EPV",
        r"$Q$ [\unit{PVU}]",
        levels=np.arange(-3, 0.5, 0.5),
        normals='xy',
        highlight_level=-1.5  # dynamical tropopause
    )

    PETSc.Sys.Print("Saved PV plots")

    # The reference state and the jet never change, so a checkpoint would only repeat them
    if not args.checkpoint:
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
            Function(cg_space).interpolate(atmos.p_bar() / 1e2),
            "Reference Pressure Profile",
            r"$\overline{p}$ [\unit{\hecto\pascal}]"
        )

        plot_function_vs_z(
            atmos.theta_bar(),
            "Reference Potential Temperature Profile",
            r"$\overline{\theta}$ [\unit{\kelvin}]"
        )

        multislice(
            Function(cg_space).interpolate(atmos.u()),
            "Jet Stream",
            r"$\overline{u}$ [\unit{\meter\per\second}]",
            levels=np.arange(5, 100, 5),
            normals='x'
        )

        multislice(
            Function(cg_space).interpolate(atmos.geostrophic_vorticity()),
            "Background Geostrophic Vorticity",
            r"$\overline{\zeta_g}$ [\unit{\per\second}]",
            levels=np.linspace(-5e-5, 5e-5, 11),
            normals='x'
        )

    multislice(
        derived.geostrophic_vorticity(),
        "Geostrophic Vorticity",
        r"$\zeta_g$ [\unit{\per\second}]",
        levels=np.linspace(-0.0002, 0.00005, 11),
        normals='xy'
    )

    multislice(
        derived.potential_temperature_anomaly(),
        "Potential Temperature Anomaly",
        r"$\theta^*$ [\unit{\kelvin}/\unit{\celsius}]",
        levels=np.arange(-20, 20, 3),
        normals='xy',
    )

    multislice(
        derived.potential_temperature(),
        "Potential Temperature",
        r"$\theta$ [\unit{\kelvin}]",
        levels = np.arange(250, 350, 5),
        normals='xy'
    )

    multislice(
        derived.pressure_anomaly_hpa(),
        "Pressure Anomaly",
        r"$p^*$ [\unit{\hecto\pascal}]",
        levels=pres_ano_levels,
        normals='xy'
    )

    multislice(
        derived.pressure_hpa(),
        "Pressure",
        r"$p$ [\unit{\hecto\pascal}]",
        levels=pres_levels,
        normals='z',
        cbar_bound_region=lambda x,y,z: z<10
    )

    multislice(
        derived.temperature_anomaly(),
        "Temperature Anomaly",
        r"$T^*$ [\unit{\kelvin}/\unit{\celsius}]",
        levels=np.arange(-20, 20, 3),
        normals='xy'
    )

if __name__ == "__main__":
    petsctools.print_citations_at_exit()
    main()
"""Surface wind of a timestepped run at each daily checkpoint, stacked into one full-page figure."""
from hash_seed import use_same_hash
use_same_hash()
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake.petsc import PETSc
from background_plots import plot_slice_heatmap
from barnes_atmosphere import BarnesAtmosphere
from derived_quantities import ResolvedAtmosphere
from domain_builder import DomainBuilder
from plot_utils import FIGURE_SIZE
from timestepping import load_checkpoint
import sweep

CHECKPOINTS = [f"timestepping_data/timestepping_1382023_t{h}h.h5" for h in range(24, 169, 24)]
X_MAX = 20e6  # Past here the flow is still undisturbed by 168 h


def clear_caches():
    """The lru_caches on these classes hold the last instance (and its Functions) alive,
    so without this the previous checkpoint is still in memory while the next loads."""
    for cls in (BarnesAtmosphere, DomainBuilder, ResolvedAtmosphere):
        for klass in cls.__mro__:
            for attr in vars(klass).values():
                if hasattr(attr, 'cache_clear'):
                    attr.cache_clear()
    # Firedrake defers PETSc destroys in parallel, so nothing is freed until the cleanup
    gc.collect()
    PETSc.garbage_cleanup(PETSc.COMM_WORLD)


def main():
    sweep.quiet_petsc()
    cbar_title = r"$\left|\mathbf{u}\right|$ [\unit{\meter \per \second}]"

    fig = axes = None
    if MPI.COMM_WORLD.rank == 0:
        # main.tex's text block is 6.30 in x 10.1 in, leaving room for a caption
        fig, axes = plt.subplots(len(CHECKPOINTS), 1, sharex=True, layout='constrained',
                                 figsize=(FIGURE_SIZE[0], 9.0))

    heatmaps = []
    for i, path in enumerate(CHECKPOINTS):
        atmos, psi, q, t = load_checkpoint(path)
        derived = ResolvedAtmosphere(psi, atmos, q)
        # Pixels square despite the wide domain, and arrows on a square grid: 9 rows, 0.4 of a spacing long
        nx = round(200 * X_MAX / atmos.Ly)
        spacing = atmos.Ly / 8
        heatmap = plot_slice_heatmap(
            derived.horizontal_wind_speed(), "", cbar_title,
            levels=np.arange(0, 12, 0.5),
            normal_dir="z", slice_coord=0,
            vector_field=(derived.u(), derived.v()),
            num_points_h=nx,
            quiver_density=(round(X_MAX / spacing) + 1, 9),
            quiver_kwargs=dict(angles='xy', scale_units='xy', scale=2.5e3 / spacing,
                               width=0.003, headlength=3, headaxislength=2.5),
            bounds=((0, X_MAX), (0, atmos.Ly), (0, atmos.H)),
            ax=axes[i] if axes is not None else None,
        )
        if fig is not None:
            heatmaps.append(heatmap)
            axes[i].set_box_aspect(atmos.Ly / X_MAX)
            axes[i].set_title(rf"({'abcdefg'[i]}) $t = {t / 3600:.0f}$ \unit{{\hour}}")
            axes[i].label_outer()

        del atmos, psi, q, derived
        clear_caches()

    if fig is not None:
        # One colour scale across all panels, so the growth is comparable
        vmax = max(h.get_array().max() for h in heatmaps)
        for h in heatmaps:
            h.set_clim(0, vmax)
        fig.colorbar(heatmaps[0], ax=axes, location='right', shrink=0.6, aspect=40).set_label(cbar_title)
        fig.savefig("tex/plots/surface_wind_evolution.pdf", bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()

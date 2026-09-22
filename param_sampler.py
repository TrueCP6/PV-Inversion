from hash_seed import use_same_hash
use_same_hash()
from variator import Variator, trop_correlation_axes
from firedrake import *
import numpy as np
from parameters import PhysicalParams
from dataclasses import asdict
import argparse
import json
import sweep
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

class ParamSampler:
    def __init__(self, optimise_for="pres", seed=4623, job_id=0):
        self.variator = Variator()
        self.param_tuple = self._get_effectual_params()
        self.dim = len(self.param_tuple)
        self.comm = COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.rng = np.random.default_rng(seed) # not shared between all ranks but all ranks will have the same seed
        self.normalised_control = self.record_to_normalised(PhysicalParams()) # A representation of the control in normalised space
        self.optimise_for = optimise_for
        self.job_id = job_id

    def _get_effectual_params(self): # Reduce dimensionality by removing parameters that don't make a difference
        ineffectual_params = ["delta", "N_strat_variation", "trop_width"]
        all_params = Variator.quantities_to_vary()
        return [param for param in all_params if param[0] not in ineffectual_params]

    def sample_normalised(self): # Gets a random vector of normalised parameter values
        return self.rng.uniform(low=0.0, high=1.0, size=self.dim)

    def dist_to_control(self, normalised):
        return float(np.linalg.norm(normalised - self.normalised_control))

    def normalised_to_dict(self, normalised_arr):
        dict = {}
        for i in range(self.dim):
            param_name, bound, _ = self.param_tuple[i]
            a, b = bound
            unnorm = a + (b-a)*normalised_arr[i]
            dict[param_name] = float(unnorm)
        return dict

    def record_to_normalised(self, record : PhysicalParams):
        dict = asdict(record)
        normalised = np.zeros(self.dim)

        for i in range(self.dim):
            param_name, bound, _ = self.param_tuple[i]
            a, b = bound
            unnorm = dict[param_name]
            normalised[i] = (unnorm - a)/(b-a)

        return normalised

    def cost(self, x):
        params = self.normalised_to_dict(x)
        try:
            derived = self.variator.get_derived(params)
        except ConvergenceError as exc:
            if self.rank == 0:
                print(f"solve diverged at {params}: {exc}, scoring it 0", flush=True)
            return 0.0  # all three quantities are negative where they are interesting

        if self.optimise_for == "pres":
            cost = derived.min_surf_pressure_ano_hpa()
        elif self.optimise_for == "vort":
            cost = derived.min_surf_vort()
        elif self.optimise_for == "wind":
            cost = -derived.max_surf_wind_speed()
        else:
            raise ValueError

        return cost

    def driver_cost(self, x):
        self.comm.bcast(("eval", x), root=0)  # wake the other ranks, so every rank takes part in the collective solve
        f = self.cost(x)
        with open(f"optimise_{self.job_id}.csv", "a") as fh:
            fh.write(",".join(map(str, [*x, f])) + "\n")
        return f

    def optimise(self, max_evals):
        """Minimise the cost over the normalised parameter box with Py-BOBYQA, starting from the control.

        Py-BOBYQA runs on rank 0 only, while the other ranks wait for each point it asks for and
        solve alongside it. Returns (x, f) of the best point found on every rank.
        """
        if self.rank != 0:
            while True:
                cmd, payload = self.comm.bcast(None, root=0)
                if cmd == "stop":
                    return payload
                self.cost(payload)

        import pybobyqa
        result = None
        try:
            soln = pybobyqa.solve(
                self.driver_cost,
                x0=self.normalised_control,
                bounds=(np.zeros(self.dim), np.ones(self.dim)),
                rhobeg=0.1,
                rhoend=0.001,
                maxfun=max_evals,
                seek_global_minimum=True
            )
            print(soln)
            result = (soln.x, soln.f)
        finally:
            # Always release the other ranks, even if the optimiser raises, or they would wait forever
            self.comm.bcast(("stop", result), root=0)

        return result

    def all_data(self, x):
        params = self.normalised_to_dict(x)
        try:
            derived = self.variator.get_derived(params)
        # Same collective failure on every rank, so they all drop the same sample
        except ConvergenceError as exc:
            if self.rank == 0:
                print(f"solve diverged at {params}: {exc}, dropping the sample", flush=True)
            return None

        wind = derived.max_surf_wind_speed()
        vort = derived.min_surf_vort()
        pres = derived.min_surf_pressure_ano_hpa()
        dist = self.dist_to_control(x)
        trop = derived.min_dyn_tropopause_height()

        return wind, vort, pres, dist, trop

    def random_sample_data(self, num_samples):
        all_data = [self.all_data(self.sample_normalised()) for _ in range(num_samples)]
        all_data = [data for data in all_data if data is not None]
        # zip of nothing unpacks to nothing, so spell out the every-sample-diverged case
        wind, vort, pres, dist, trop = list(map(list, zip(*all_data))) if all_data else ([],) * 5

        return {
            "max_surface_wind": wind,
            "min_surf_vort": vort,
            "min_surf_pres": pres,
            "dist_to_control": dist,
            "min_dyn_trop_height": trop,
        }

def plot_trop_correlation(json_path, output_path="tex/plots/random_sample_trop_correlations.pdf"):
    """Scatter random_sample_data's results against tropopause height, coloured by distance from the control."""
    if not sweep.is_main_rank():
        return

    with open(json_path) as f:
        data = json.load(f)

    # Draw the furthest samples first, so those nearest the control sit on top
    order = np.argsort(data["dist_to_control"])[::-1]
    dist = np.asarray(data["dist_to_control"])[order]
    trop = np.asarray(data["min_dyn_trop_height"])[order]

    # Shrink and fade the dots as the sample count grows, so 10 points stay visible and 10000 don't become a blob
    n = len(dist)
    size = np.clip(200 / np.sqrt(n), 1.5, 16)
    alpha = np.clip(30 / np.sqrt(n), 0.3, 0.9)

    cmap = ListedColormap(plt.cm.inferno(np.linspace(0.0, 0.9, 256)))

    fig, axes = trop_correlation_axes()
    for ax, key in zip(axes, ["min_surf_pres", "max_surface_wind", "min_surf_vort"]):
        values = np.asarray(data[key])[order]
        points = ax.scatter(trop, values, c=dist, cmap=cmap,
                            vmin=0, vmax=dist.max(), s=size, alpha=alpha, linewidths=0)

        rho = spearmanr(trop, values).statistic
        ax.legend([Line2D([], [], linestyle='none')], [rf"$\rho = {rho:.2f}$"], loc='best', fontsize=9,
                  handlelength=0, handletextpad=0, borderpad=0.3, framealpha=0.8, edgecolor='none')

    colourbar = fig.colorbar(points, ax=axes, location='bottom', shrink=0.5, aspect=40)
    colourbar.set_label("Normalised distance from control")
    colourbar.solids.set_alpha(1)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    sweep.quiet_petsc()
    parser = argparse.ArgumentParser(description='Generate data for quantity variation plots')
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=4623, help='Random seed for the parameter samples - vary it to add new samples rather than repeat old ones')
    parser.add_argument('--plot', metavar='JSON_PATH', help='Plot the given results file instead of generating new data, then exit')
    parser.add_argument('-o', '--optimise_for', choices=['pres', 'vort', 'wind'],
                        help='Search for the parameters that extremise this quantity instead of sampling at random, spending at most --num_samples solves')
    args = parser.parse_args()

    if args.plot:
        plot_trop_correlation(args.plot)
        return

    sampler = ParamSampler(optimise_for=args.optimise_for, seed=args.seed, job_id=args.job_id)

    if args.optimise_for:
        x, _ = sampler.optimise(args.num_samples)
        # Only the parameters the Variator moves - PhysicalParams defaults the rest
        data = sampler.normalised_to_dict(x)
        out_path = f"optimised_{args.optimise_for}_{args.job_id}.json"
    else:
        data = sampler.random_sample_data(args.num_samples)
        out_path = f"random_samples_{args.job_id}.json"

    if not sweep.is_main_rank():
        return

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
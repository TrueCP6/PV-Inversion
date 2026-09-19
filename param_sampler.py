from hash_seed import use_same_hash
use_same_hash()
from variator import Variator
from firedrake import *
import numpy as np
from parameters import PhysicalParams
from dataclasses import asdict
import argparse

class ParamSampler:
    def __init__(self, optimise_for="pres", seed=4623):
        self.variator = Variator()
        self.param_tuple = self._get_effectual_params()
        self.dim = len(self.param_tuple)
        self.rank = COMM_WORLD.Get_rank()
        self.rng = np.random.default_rng(seed) # not shared between all ranks but all ranks will have the same seed
        self.normalised_control = self.record_to_normalised(PhysicalParams()) # A representation of the control in normalised space
        self.optimise_for = optimise_for

    def _get_effectual_params(self): # Reduce dimensionality by removing parameters that don't make a difference
        ineffectual_params = ["delta", "N_strat", "trop_width"]
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
        derived = self.variator.get_derived(params)

        if self.optimise_for == "pres":
            cost = derived.min_surf_pressure_ano_hpa()
        elif self.optimise_for == "vort":
            cost = derived.min_surf_vort()
        elif self.optimise_for == "wind":
            cost = derived.max_surf_wind_speed()
        else:
            raise ValueError

        return cost

    def all_data(self, x):
        params = self.normalised_to_dict(x)
        derived = self.variator.get_derived(params)

        wind = derived.max_surf_wind_speed()
        vort = derived.min_surf_vort()
        pres = derived.min_surf_pressure_ano_hpa()
        dist = self.dist_to_control(x)
        trop = derived.min_dyn_tropopause_height()

        return wind, vort, pres, dist, trop

def main():
    parser = argparse.ArgumentParser(description='Generate data for quantity variation plots')
    parser.add_argument('-n', '--num_points', type=int, default=10)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    args = parser.parse_args()

    vary = Variator()
    data = vary.varying_single_param_data(args.num_points)

    with open(f"variator_{args.job_id}.json", "w") as f:
        json.dump(data, f)
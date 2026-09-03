import argparse
import json
from firedrake import *
from barnes_atmosphere import BarnesAtmosphere
from derived_quantities import ResolvedAtmosphere
from domain_builder import DomainBuilder
from parameters import SolverParams, PhysicalParams
from solver import Solver
import matplotlib.pyplot as plt
import numpy as np
import plot_utils
from decimal import Decimal

class Variator:
    def __init__(self):
        solver_params = SolverParams(check_flux=False)
        phys_params = PhysicalParams()

        self.domain = DomainBuilder(solver_params, phys_params)
        self.comm = self.domain.mesh().comm
        atmos = BarnesAtmosphere(self.domain)
        self.solver = Solver(atmos, True)

    def _get_derived(self, params) -> ResolvedAtmosphere:
        phys_params = PhysicalParams(**params)

        atmos = BarnesAtmosphere(self.domain, phys_params)
        self.solver.update_atmosphere(atmos)

        self.solver.solve_psi()
        derived = ResolvedAtmosphere(self.solver.psi_soln, atmos)

        return derived

    @property
    def quantities_to_vary(self):
        mx = PhysicalParams.Lx / 2
        quantities_to_vary = [ #todo make writup more consistent with this notation, add scaling factor to fix units
            ("N_strat", (0.02, 0.03), 1, r"$\overline{N}_\text{strat} \in [min, max]$ [\unit{\per\second}]"),
            ("N_trop", (0.01, 0.02), 1, r"$\overline{N}_\text{trop} \in [min, max]$ [\unit{\per\second}]"),
            ("trop_width", (500, 2000), 1, r"$w_\text{trop} \in [min, max]$ [\unit{\meter}]"),
            ("trop_height", (10e3, 15e3), 1e-3, r"$z_\text{trop} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("theta_bar_bottom", (273.15, 273.15+30), 1, r"$\overline{\theta}(0) \in [min, max]$ [\unit{\kelvin}]"),
            ("p_bottom", (795 * 1e2, 1013.25 * 1e2), 1e-2, r"$\overline{p}(0) \in [min, max]$ [\unit{\hecto\pascal}]"),
            ("delta", (2, 10), 1, r"$\delta \in [min, max]$"),
            ("anomaly_z_trop_offset", (-2500, 2500), 1, r"$(z_\text{ano} - z_\text{trop}) \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_x_size", (100e3, 800e3), 1e-3, r"$x_\text{size} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("anomaly_y_size", (100e3, 800e3), 1e-3, r"$y_\text{size} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("anomaly_z_size", (3500, 7000), 1, r"$y_\text{size} \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_mag", (-4e-6, -1e-6), 1e6, r"$Q_\text{anomag} \in [min, max]$ [\unit{PVU}]"),
            ("jet_x_size", (100e3, 1000e3), 1e-3, r"$L_\text{jet} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("jet_z_size", (1e3, 4e3), 1, r"$z_\text{jet} \in [min, max]$ [\unit{\meter}]"),
            ("jet_magnitude", (10, 100), 1, r"$U_\text{jet} \in [min, max]$ [\unit{\meter\per\second}]"),
            ("jet_x_pos", (mx-500e3, mx+500e3), 1e-3, r"$x_\text{jet} \in [min, max]$ [\unit{\kilo\meter}]"),
        ]

        # helper function that outputs the float as a string with 3 significant figures, but not in scientific notation
        def format_num(num : float) -> str:
            return format(Decimal(f'{num:.3g}'), 'f')

        output = []
        for var_name, bound, scale_legend, legend in quantities_to_vary:
            min_str = format_num(bound[0] * scale_legend)
            max_str = format_num(bound[1] * scale_legend)

            legend = legend.replace("min", min_str).replace("max", max_str)
            output.append((var_name, bound, legend))

        return output

    def get_data(self, num_points : int):
        normalised_pts = np.linspace(0, 1, num_points)

        values_per_qty = []

        for param_name, bound, legend in self.quantities_to_vary:
            a, b = bound
            x_pts = a + (b-a)*normalised_pts
            wind_vals, vort_vals, trop_vals, pres_vals = [], [], [], []

            for x in x_pts:
                d = self._get_derived({param_name: x})
                wind_vals.append(d.max_surf_wind_speed())
                vort_vals.append(d.min_surf_vort())
                trop_vals.append(d.min_dyn_tropopause_height())
                pres_vals.append(d.min_surf_pressure_ano_hpa())

            values_per_qty.append({
                "legend_entry": legend,
                "wind_values": wind_vals,
                "vorticity_values": vort_vals,
                "trop_height_values": trop_vals,
                "pressure_values": pres_vals,
            })

        return {
            "normalised_pts": normalised_pts.tolist(),
            "values_per_qty": values_per_qty,
        }

def main():
    parser = argparse.ArgumentParser(description='Generate data for quantity variation plots')
    parser.add_argument('-n', '--num_points', type=int, default=10)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    args = parser.parse_args()

    vary = Variator()
    data = vary.get_data(args.num_points)

    with open(f"variator_{args.job_id}.json", "w") as f:
        json.dump(data, f)

def plot_trop_correlation(json_path):
    with open(json_path) as f:
        variator_results = json.load(f)["values_per_qty"]

    x, y = [], []

    for varied_var in variator_results:
        x.extend(varied_var["trop_height_values"])
        y.extend(varied_var["pressure_values"])

    plot_utils.apply_style()
    plt.figure(figsize=(6.3, 4.5))
    ax = plt.gca()

    ax.scatter(x,y)

    plot_utils.finish_figure(f"tex/plots/test.pdf", legend=False)

def plot_variator_results(json_path):
    plot_utils.apply_style()

    with open(json_path) as f:
        data = json.load(f)

    x = np.asarray(data["normalised_pts"])
    values_per_qty = data["values_per_qty"]

    quantities = [
        ("wind_values", "Maximum surface wind speed", r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]"),
        ("pressure_values", "Minimum surface pressure anomaly", r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]"),
        ("trop_height_values", "Minimum dynamical tropopause height", r"$\min\, z_\text{trop}$ [\unit{\meter}]"),
        ("vorticity_values", "Minimum surface vorticity", r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]"),
    ]

    # Colours from a 20-colour qualitative map, reordered so the 10 distinct hues are used before their lighter tab20 pairing repeats one - keeps neighbouring parameter variations as distinguishable as possible (the variations are categorical, not ordinal, so a sequential map like viridis would be misleading here).
    tab20 = plt.cm.tab20.colors
    colour_order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colours = [tab20[colour_order[i % len(colour_order)]] for i in range(len(values_per_qty))]

    for key, title, y_label in quantities:
        plt.figure(figsize=(6.3, 4.5))
        ax = plt.gca()

        for qty, colour in zip(values_per_qty, colours):
            ax.plot(x, qty[key], color=colour, linewidth=1.2)

        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Normalised parameter value")
        ax.set_ylabel(y_label)

        entries = [(qty[key][-1], colour, qty["legend_entry"])
                   for qty, colour in zip(values_per_qty, colours)]
        plot_utils.label_lines_at_end(ax, entries)

        lwr_case = title.replace(" ", "_").lower()
        plot_utils.finish_figure(f"tex/plots/{lwr_case}.pdf", legend=False)

if __name__ == "__main__":
    main()
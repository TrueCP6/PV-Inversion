import argparse
import json
import re
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
        quantities_to_vary = [ #todo make writeup more consistent with this notation, add scaling factor to fix units
            ("N_strat", (0.02, 0.03), 1, r"$\overline{N}_\text{strat} \in [min, max]$ [\unit{\per\second}]"),
            ("N_trop", (0.01, 0.02), 1, r"$\overline{N}_\text{trop} \in [min, max]$ [\unit{\per\second}]"),
            ("trop_width", (500, 2000), 1, r"$w_\text{trop} \in [min, max]$ [\unit{\meter}]"),
            ("trop_height", (10e3, 15e3), 1e-3, r"$z_\text{trop} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("temperature_bottom", (273.15, 273.15+30), 1, r"$\overline{T}(0) \in [min, max]$ [\unit{\kelvin}]"),
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
            ("jet_x_pos", (mx-1500e3, mx+1500e3), 1e-3, r"$x_\text{jet} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("latitude", (-40, -20), 1, r"$\varphi \in [\qty{min}{\degree}, \qty{max}{\degree}]$")
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
    plot_utils.apply_style()

    with open(json_path) as f:
        data = json.load(f)

    normalised_pts = data["normalised_pts"]
    values_per_qty = data["values_per_qty"]

    colours = plot_utils.qualitative_colours(len(values_per_qty))

    delta_qty = next(q for q in values_per_qty if q["legend_entry"].startswith(r"$\delta"))
    lo, hi = (float(v) for v in re.search(r'\[(-?[\d.]+),\s*(-?[\d.]+)\]', delta_qty["legend_entry"]).groups())
    delta_values = [lo + (hi - lo) * t for t in normalised_pts]
    control_idx = min(range(len(delta_values)), key=lambda i: abs(delta_values[i] - PhysicalParams().delta))

    control_x = delta_qty["trop_height_values"][control_idx]
    control_y = {
        "pressure_values": delta_qty["pressure_values"][control_idx],
        "wind_values": delta_qty["wind_values"][control_idx],
        "vorticity_values": delta_qty["vorticity_values"][control_idx],
    }

    quantities = [
        ("pressure_values", "Tropopause height vs pressure anomaly", r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]"),
        ("wind_values", "Tropopause height vs wind speed", r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]"),
        ("vorticity_values", "Tropopause height vs vorticity", r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]"),
    ]

    for key, title, y_label in quantities:
        plt.figure(figsize=plot_utils.SQUARE_HALF_FIGURE_SIZE)
        ax = plt.gca()

        for qty, colour in zip(values_per_qty, colours):
            ax.scatter(qty["trop_height_values"], qty[key], color=colour,
                       s=10, alpha=0.85, linewidths=0)

        ax.scatter([control_x], [control_y[key]], color='black', marker='*',
                   s=80, edgecolors='white', linewidths=0.5, zorder=5)
        ax.annotate("Control", (control_x, control_y[key]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')

        ax.set_xlabel(r"$\min\, z_\text{trop}$ [\unit{\meter}]")
        ax.set_ylabel(y_label)

        lwr_case = title.replace(" ", "_").lower()
        plot_utils.finish_figure(f"tex/plots/{lwr_case}.pdf", legend=False)

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

    colours = plot_utils.qualitative_colours(len(values_per_qty))

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
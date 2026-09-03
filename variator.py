import json
from dataclasses import asdict
from firedrake import *
from barnes_atmosphere import BarnesAtmosphere
from derived_quantities import ResolvedAtmosphere
from domain_builder import DomainBuilder
from parameters import SolverParams, PhysicalParams
from solver import Solver
import matplotlib.pyplot as plt
import numpy as np
import plot_utils

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
            ("N_strat", (0.02, 0.03), r"$\overline{N}_\text{strat} \in [min, max]$ [\unit{\per\second}]"),
            ("N_trop", (0.01, 0.02), r"$\overline{N}_\text{trop} \in [min, max]$ [\unit{\per\second}]"),
            ("trop_width", (500, 2000), r"$w_\text{trop} \in [min, max]$ [\unit{\meter}]"),
            ("trop_height", (10e3, 15e3), r"$z_\text{trop} \in [min, max]$ [\unit{\meter}]"),
            ("theta_bar_bottom", (273.15, 273.15+30), r"$\overline{\theta}(0) \in [min, max]$ [\unit{\kelvin}]"),
            ("p_bottom", (795 * 1e2, 1013.25 * 1e2), r"$\overline{p}(0) \in [min, max]$ [\unit{\pascal}]"),
            ("delta", (2, 10), r"$\delta \in [min, max]$"),
            ("anomaly_z_trop_offset", (-2500, 2500), r"$(z_\text{ano} - z_\text{trop}) \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_x_size", (100e3, 800e3), r"$x_\text{size} \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_y_size", (100e3, 800e3), r"$y_\text{size} \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_z_size", (3500, 7000), r"$y_\text{size} \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_mag", (-4e-6, -1e-6), r"$Q_\text{anomag} \in [min, max]$ [\unit{\kelvin\meter\squared\per\second\per\kg}]"),
            ("jet_x_size", (100e3, 1000e3), r"$L_\text{jet} \in [min, max]$ [\unit{\meter}]"),
            ("jet_z_size", (1e3, 4e3), r"$z_\text{jet} \in [min, max]$ [\unit{\meter}]"),
            ("jet_magnitude", (10, 300/3.6), r"$U_\text{jet} \in [min, max]$ [\unit{\meter\per\second}]"),
            ("jet_x_pos", (mx-500e3, mx+500e3), r"$x_\text{jet} \in [min, max]$ [\unit{\meter}]"),
        ]

        output = []
        for i in range(len(quantities_to_vary)):
            var_name, bound, legend = quantities_to_vary[i]
            legend = legend.replace("min", str(bound[0])).replace("max", str(bound[1]))
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

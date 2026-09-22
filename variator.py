from hash_seed import use_same_hash
use_same_hash()
import argparse
import json
import re
from mpi4py import MPI
from firedrake import *
from barnes_atmosphere import BarnesAtmosphere
from derived_quantities import ResolvedAtmosphere
from domain_builder import DomainBuilder
from parameters import SolverParams, PhysicalParams
from diagnostic_solver import DiagnosticSolver
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plot_utils
from decimal import Decimal
import sweep

# Related parameters grouped into one panel per figure, so each panel carries few enough lines/colours to stay readable
PARAMETER_GROUPS = [
    ("Stratification and tropopause structure", ["N_strat_variation", "N_trop", "trop_width", "trop_height_variation", "delta"]),
    ("Background state", ["temperature_bottom_variation", "p_bottom", "latitude"]),
    ("PV anomaly", ["anomaly_z_trop_offset", "anomaly_x_size", "anomaly_y_size", "anomaly_z_size", "anomaly_mag", "anomaly_clip"]),
    ("Jet parameters", ["jet_y_size", "jet_z_size", "jet_magnitude", "jet_y_pos"]),
]

# y-axis labels of the three tropopause-correlation panels, in panel order
TROP_PANEL_LABELS = [
    r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]",
    r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]",
    r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]",
]

def trop_correlation_axes():
    """Styled figure of three panels plotting surface pressure, wind and vorticity against tropopause height."""
    plot_utils.apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(plot_utils.FIGURE_SIZE[0], plot_utils.SQUARE_HALF_FIGURE_SIZE[1] * 0.85),
                              constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.02, w_pad=0.02, h_pad=0.02)

    for ax, y_label in zip(axes, TROP_PANEL_LABELS):
        ax.set_xlabel(r"$\min\, z_\text{trop}$ [\unit{\meter}]")
        ax.set_ylabel(y_label)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
    axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    return fig, axes

class Variator:
    def __init__(self):
        solver_params = SolverParams(check_flux=False)
        phys_params = PhysicalParams()

        self.domain = DomainBuilder(solver_params, phys_params)
        self.comm = self.domain.mesh().comm
        atmos = BarnesAtmosphere(self.domain)
        self.solver = DiagnosticSolver(atmos, True)

    def get_derived(self, params) -> ResolvedAtmosphere:
        phys_params = PhysicalParams(**params)

        atmos = BarnesAtmosphere(self.domain, phys_params)
        self.solver.update_atmosphere(atmos)

        self.solver.solve_psi()
        derived = ResolvedAtmosphere(self.solver.psi_soln, atmos)

        return derived

    @staticmethod
    def quantities_to_vary():
        mx = PhysicalParams.Lx / 2
        quantities_to_vary = [ #todo make writeup more consistent with this notation
            ("N_strat_variation", (-0.005, 0.003), 1, r"$\overline{N}'_\text{strat} \in [min, max]$ [\unit{\per\second}]"), #happy
            ("N_trop", (0.008, 0.014), 1, r"$\overline{N}_\text{trop} \in [min, max]$ [\unit{\per\second}]"), #happy
            ("trop_width", (500, 2000), 1, r"$w_\text{trop} \in [min, max]$ [\unit{\meter}]"), #happy
            ("trop_height_variation", (-2e3, 2e3), 1e-3, r"$z'_\text{trop} \in [min, max]$ [\unit{\kilo\meter}]"), #happy
            ("temperature_bottom_variation", (-10, 10), 1, r"$\overline{T}'(0) \in [min, max]$ [\unit{\kelvin}]"),
            ("p_bottom", (795 * 1e2, 1013.25 * 1e2), 1e-2, r"$\overline{p}(0) \in [min, max]$ [\unit{\hecto\pascal}]"),
            ("delta", (2, 10), 1, r"$\delta \in [min, max]$"), #happy
            ("anomaly_z_trop_offset", (-2500, 2500), 1e-3, r"$(z_\text{ano} - z_\text{trop}) \in [min, max]$ [\unit{\kilo\meter}]"),
            ("anomaly_x_size", (100e3, 800e3), 1e-3, r"$x_\text{size} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("anomaly_y_size", (100e3, 800e3), 1e-3, r"$y_\text{size} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("anomaly_z_size", (3500, 7000), 1, r"$z_\text{size} \in [min, max]$ [\unit{\meter}]"),
            ("anomaly_mag", (-4e-6, -1e-6), 1e6, r"$Q_\text{anomag} \in [min, max]$ [\unit{PVU}]"),
            ("anomaly_clip", (-2e-6, -1e-6), 1e6, r"$Q_\text{anoclip} \in [min, max]$ [\unit{PVU}]"),
            ("jet_y_size", (100e3, 1000e3), 1e-3, r"$L_\text{jet} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("jet_z_size", (1e3, 4e3), 1, r"$z_\text{jet} \in [min, max]$ [\unit{\meter}]"),
            ("jet_magnitude", (10, 100), 1, r"$U_\text{jet} \in [min, max]$ [\unit{\meter\per\second}]"),
            ("jet_y_pos", (mx-1500e3, mx+1500e3), 1e-3, r"$y_\text{jet} \in [min, max]$ [\unit{\kilo\meter}]"),
            ("latitude", (-50, -30), 1, r"$\varphi \in [\qty{min}{\degree}, \qty{max}{\degree}]$") #happy - makes rossby number a bit smaller
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

    def varying_single_param_data(self, num_points : int):
        normalised_pts = np.linspace(0, 1, num_points)

        values_per_qty = []

        for param_name, bound, legend in Variator.quantities_to_vary():
            a, b = bound
            x_pts = a + (b-a)*normalised_pts
            wind_vals, vort_vals, trop_vals, pres_vals = [], [], [], []

            for x in x_pts:
                d = self.get_derived({param_name: x})
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
    sweep.quiet_petsc()
    parser = argparse.ArgumentParser(description='Generate data for quantity variation plots')
    parser.add_argument('-n', '--num_points', type=int, default=10)
    parser.add_argument('-j', '--job_id', type=int, default=0)
    args = parser.parse_args()

    vary = Variator()
    data = vary.varying_single_param_data(args.num_points)

    if not sweep.is_main_rank():
        return

    with open(f"variator_{args.job_id}.json", "w") as f:
        json.dump(data, f)

def plot_trop_correlation(json_path):
    if MPI.COMM_WORLD.rank != 0:
        return

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

    fig, axes = trop_correlation_axes()

    for ax, key in zip(axes, ["pressure_values", "wind_values", "vorticity_values"]):
        for qty, colour in zip(values_per_qty, colours):
            ax.scatter(qty["trop_height_values"], qty[key], color=colour,
                       s=10, alpha=0.85, linewidths=0)

        ax.scatter([control_x], [control_y[key]], color='black', marker='*',
                   s=80, edgecolors='white', linewidths=0.5, zorder=5)
        ax.annotate("Control", (control_x, control_y[key]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')

    handles = [Line2D([0], [0], marker='o', linestyle='', color=colour, markersize=5)
               for colour in colours]
    labels = [qty["legend_entry"] for qty in values_per_qty]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.36),
               ncol=3, fontsize=6, frameon=False)

    plt.savefig("tex/plots/tropopause_height_correlations.pdf", bbox_inches='tight')
    plt.close()

def plot_variator_results(json_path):
    if MPI.COMM_WORLD.rank != 0:
        return

    plot_utils.apply_style()

    with open(json_path) as f:
        data = json.load(f)

    x = np.asarray(data["normalised_pts"])
    values_per_qty = data["values_per_qty"]

    var_names = [name for name, _, _ in Variator.quantities_to_vary()]
    assert len(var_names) == len(values_per_qty), \
        "quantities_to_vary() no longer matches the data this json was generated from"
    colours = plot_utils.qualitative_colours(len(values_per_qty))
    qty_by_name = dict(zip(var_names, values_per_qty))
    colour_by_name = dict(zip(var_names, colours))

    quantities = [
        ("wind_values", "Maximum surface wind speed", r"$\max\left|\mathbf{u}\right|_{z=0}$ [\unit{\meter\per\second}]"),
        ("pressure_values", "Minimum surface pressure anomaly", r"$\min\, p^*_{z=0}$ [\unit{\hecto\pascal}]"),
        ("trop_height_values", "Minimum dynamical tropopause height", r"$\min\, z_\text{trop}$ [\unit{\meter}]"),
        ("vorticity_values", "Minimum surface vorticity", r"$\min\, \zeta_g|_{z=0}$ [\unit{\per\second}]"),
    ]

    for key, title, y_label in quantities:
        all_vals = [v for qty in values_per_qty for v in qty[key]]
        pad = 0.05 * (max(all_vals) - min(all_vals))
        ylim = (min(all_vals) - pad, max(all_vals) + pad)

        fig, axes = plt.subplots(len(PARAMETER_GROUPS), 1, sharex=True,
                                  figsize=(plot_utils.FIGURE_SIZE[0], 1.8 * len(PARAMETER_GROUPS)))

        for ax, (group_title, group_vars) in zip(axes, PARAMETER_GROUPS):
            entries = []
            for var_name in group_vars:
                qty = qty_by_name[var_name]
                colour = colour_by_name[var_name]
                ax.plot(x, qty[key], color=colour, linewidth=1.2)
                entries.append((qty[key][-1], colour, qty["legend_entry"]))

            ax.set_xlim(0, 1)
            ax.set_ylim(*ylim)
            ax.set_ylabel(y_label)
            ax.set_title(group_title, fontsize=9, loc='left')
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            if key == "vorticity_values":
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plot_utils.label_lines_at_end(ax, entries)

        axes[-1].set_xlabel(r"Normalised parameter value")

        plt.tight_layout()
        lwr_case = title.replace(" ", "_").lower()
        plt.savefig(f"tex/plots/{lwr_case}.pdf", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
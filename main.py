from hash_seed import use_same_hash
use_same_hash()
from tests import PrognosticMMSTests
import unittest
import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt
from param_sampler import ParamSampler
from barnes_atmosphere import *
from domain_builder import *
from parameters import *

def main():
    # plot_trop_correlation("variator_1365465.json")
    # plot_variator_results("variator_1365465.json")

    # todo add support for updated params in derived_quantities

    domain = DomainBuilder(SolverParams(), PhysicalParams())
    jet_z_sizes = np.linspace(1000, 8000, 5)
    jet_magnitudes = np.linspace(5, 35, 5)

    for jet_z_size in jet_z_sizes:
        for jet_magnitude in jet_magnitudes:
            phys_params = PhysicalParams(jet_z_size=float(jet_z_size), jet_magnitude=float(jet_magnitude), N_strat_variation=0.008)
            atmos = BarnesAtmosphere(domain, phys_params)
            epv = Function(domain.cg_space()).interpolate(atmos.ertel_pv())

            max_epv = 1e6 * get_global_max(epv)

            PETSc.Sys.Print(f'z_jet = {jet_z_size:.3} | jet_mag = {jet_magnitude:.3} | N_strat = {phys_params.N_strat:.3} | max_epv = {max_epv:.3} PVU')

if __name__ == "__main__":
    main()
import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # plot_trop_correlation("variator_1296058.json")
    # plot_variator_results("variator_1296058.json")

    # todo q and theta_star advection
    # todo add working update q and theta_star function - will also need to update boundaries potensh
    # todo determine correct stability constraint for rk4

    phys_params = PhysicalParams()
    solver_params = SolverParams()
    domain = DomainBuilder(solver_params, phys_params)
    atmos = BarnesAtmosphere(domain)

if __name__ == "__main__":
    main()
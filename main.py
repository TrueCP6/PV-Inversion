import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # todo make these work even when run with mpi
    # plot_trop_correlation("variator_1296058.json")
    # plot_variator_results("variator_1296058.json")

    # todo change solver name
    # todo switch to theta_star and q initial
    # todo q and theta_star advection
    # todo add working update q and theta_star function - will also need to update boundaries potensh
    # todo determine correct stability constraint for rk4
    # todo separate and improve get_extrema helper function

    phys_params = PhysicalParams()
    solver_params = SolverParams()
    domain = DomainBuilder(solver_params, phys_params)
    atmos = BarnesAtmosphere(domain)
    PETSc.Sys.Print(atmos.theta_star())

if __name__ == "__main__":
    main()
from hash_seed import use_same_hash
from param_sampler import ParamSampler

use_same_hash()
from tests import PrognosticMMSTests
import unittest
import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # plot_trop_correlation("variator_1296058.json")
    # plot_variator_results("variator_1296058.json")

    # todo add support for updated params in derived_quantities

    # suite = unittest.TestLoader().loadTestsFromTestCase(PrognosticMMSTests)
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    sampler = ParamSampler()
    for i in range(5):
        point = sampler.sample_normalised()
        PETSc.Sys.Print(sampler.all_data(point))

if __name__ == "__main__":
    main()
from math_utils import use_same_hash
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

    # todo determine correct stability constraint for rk4
    # todo add support for updated params in derived_quantities

    suite = unittest.TestLoader().loadTestsFromTestCase(PrognosticMMSTests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    main()
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
    plot_trop_correlation("variator_1380684.json")
    plot_variator_results("variator_1380684.json")

    # vary = Variator()
    # base = vary.get_derived(PhysicalParams(p_bottom=1000e2))
    # base = base.psi.copy(deepcopy=True)
    #
    # perturb = vary.get_derived(PhysicalParams(p_bottom=800e2))
    # perturb = perturb.psi
    #
    # err = relative_error(base, perturb)
    # PETSc.Sys.Print(err)

if __name__ == "__main__":
    main()
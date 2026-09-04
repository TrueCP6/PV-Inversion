import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    plot_trop_correlation("variator_1293007.json")
    plot_variator_results("variator_1293007.json")

if __name__ == "__main__":
    main()
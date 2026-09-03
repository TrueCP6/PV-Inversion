import json
from petsc4py import PETSc
from variator import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    plot_variator_results("variator_test.json")

    #vary = Variator()
    #
    # data = vary.get_data(1)
    # with open("variator_test.json", "w") as f:
    #     json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
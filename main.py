import json
from petsc4py import PETSc
from variator import Variator
import numpy as np
import matplotlib.pyplot as plt

def main():
    vary = Variator()

    data = vary.get_data(1)
    with open("variator_test.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
from petsc4py import PETSc
from variator import Variator
import numpy as np
import matplotlib.pyplot as plt

def main():
    vary = Variator()

    PETSc.Sys.Print(vary.quantities_to_vary[0])

if __name__ == "__main__":
    main()
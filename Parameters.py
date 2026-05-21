from dataclasses import dataclass
from math import sin, pi


@dataclass
class PhysicalParams:
    Lx: float = 1000e3
    Ly: float = 1000e3
    H: float = 20e3
    f: float = 2 * 7.292e-5 * sin(-42 / 180 * pi)
    g: float = 9.80665


@dataclass
class SolverParams:
    nx: int = 30
    ny: int = 30
    nz: int = 30
    check_flux: bool = True
    output_file: str = "output.pvd"
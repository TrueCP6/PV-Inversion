from dataclasses import dataclass
from math import sin, pi
from firedrake import Constant

@dataclass
class DomainConfig:
    N_h: int = 30
    N_z: int = N_h
    x_max: float = 1000e3
    y_max: float = 1000e3
    z_top: float = 20e3

@dataclass
class PhysicsConfig:
    z_trop: float = 12500.0
    rho_bot: float = 1.225
    rho_top: float = 0.08803
    g: float = 9.81
    # -42 degrees latitude Coriolis
    f_coriolis: float = 2 * 7.292e-5 * sin(-42 / 180 * pi)
    u = Constant(4)
    v = Constant(0)

@dataclass
class AnomalyConfig:
    # Defaulting to the center of the domain
    x_pos: float = 500e3
    y_pos: float = 500e3
    z_pos: float = 10000.0
    x_size: float = 200e3
    y_size: float = 200e3
    z_size: float = 5000.0

@dataclass
class SolverConfig:
    check_flux: bool = True
    output_filename: str = "output.pvd"
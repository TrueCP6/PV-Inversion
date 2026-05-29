from dataclasses import dataclass
from math import sin, pi

@dataclass
class PhysicalParams:
    Lx: float = 2000e3
    Ly: float = 2000e3
    H: float = 20e3
    f: float = 2 * 7.292e-5 * sin(-42 / 180 * pi)
    g: float = 9.80665
    N_strat: float = 0.03
    N_trop: float = 0.01
    trop_width: float = 1000
    trop_height: float = 10000
    theta_bar_bottom: float = 273.15 + 20
    # Constants for dry air
    R: float = 287.05
    c_p : float = 1005
    p_bottom: float = 1000 * 1e2 # 1000 hpa
    p_s: float = 1000 * 1e2

    anomaly_x_pos = Lx / 2
    anomaly_y_pos = Ly / 2
    anomaly_z_pos = trop_height
    anomaly_x_size = 200e3
    anomaly_y_size = 200e3
    anomaly_z_size = 5000

@dataclass
class SolverParams:
    nx: int = 25
    ny: int = 25
    nz: int = 25
    check_flux: bool = False
    output_file: str = "output.pvd"
    quadrilateral: bool = True
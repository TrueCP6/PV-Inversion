from dataclasses import dataclass
from math import sin, pi, sqrt

@dataclass
class PhysicalParams:
    Lx: float = 7500e3
    Ly: float = 5000e3
    H: float = 20e3
    f: float = 2 * 7.292e-5 * sin(-42 / 180 * pi)
    g: float = 9.80665
    N_strat: float = 0.03
    N_trop: float = 0.01
    trop_width: float = 1000
    trop_height: float = 12500
    theta_bar_bottom: float = 273.15 + 20
    # Constants for dry air
    R: float = 287.05
    c_p : float = 1005
    p_bottom: float = 1000 * 1e2 # 1000 hpa
    p_s: float = 1000 * 1e2
    delta: float = 2

    anomaly_x_pos = Lx / 2
    anomaly_y_pos = Ly / 2
    anomaly_z_pos = trop_height
    anomaly_x_size = 200e3
    anomaly_y_size = 200e3
    anomaly_z_size = 5000
    jet_y_size = 500e3
    jet_z_size = 2e3
    jet_magnitude = 35

@dataclass
class SolverParams:
    nx: int = 40
    ny: int = 40
    nz: int = 40
    check_flux: bool = True
    output_file: str = "output.pvd"
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-8
    matfree_params = {
        "mat_type": "matfree",
        "ksp_type": "cg",
        "ksp_rtol": ksp_rtol,
        "ksp_atol": ksp_atol,
        # p-multigrid for outer preconditioner
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        # For p=2,3,4
        "pmg_mg_levels_ksp_type": "chebyshev",
        "pmg_mg_levels_pc_type": "jacobi",
        # Assemble the p=1 matrix and use solve directly using lu factorisation
        "pmg_mg_coarse_ksp_type": "preonly",  # Don't iterate, just apply the direct solver once
        "pmg_mg_coarse_pc_type": "python",
        "pmg_mg_coarse_pc_python_type": "firedrake.AssembledPC",  # Force assembly of ONLY the p=1 matrix
        "pmg_mg_coarse_assembled_pc_type": "lu"  # Apply LU to the explicitly assembled coarse matrix
    } # TODO ensure firedrake is using sum factorisation and/or fast diagonalisation
    assembled_mat_params = { # similar to above but use a fully assembled matrix instead - much faster but uses much more memory
        "mat_type": "aij",
        "ksp_type": "cg",
        "pc_type": "python",
        "ksp_rtol": ksp_rtol,
        "ksp_atol": ksp_atol,
        "ksp_monitor": None,
        "pc_python_type": "firedrake.PMGPC",
        "pmg_mg_levels_pc_type": "jacobi",
        "pmg_mg_coarse_pc_type": "lu"
    }
    gamma: float = 0
    polynomial_order: int = 4
from dataclasses import dataclass
from math import sin, pi, sqrt

# Let UFL estimate a quadrature degree.
ESTIMATE_QUADRATURE_DEGREE = -1

@dataclass
class PhysicalParams:
    Lx: float = 7500e3
    Ly: float = 7500e3
    H: float = 20e3
    latitude: float = -42
    g: float = 9.80665
    N_strat: float = 0.03
    N_trop: float = 0.01
    trop_width: float = 1000
    trop_height: float = 12500
    temperature_bottom: float = 273.15 + 20
    # Constants for dry air
    R: float = 287.05
    c_p : float = 1005
    p_bottom: float = 1000 * 1e2 # 1000 hpa
    p_ref: float = 1000 * 1e2
    delta: float = 2

    anomaly_z_trop_offset: float = 0
    anomaly_x_size: float = 200e3
    anomaly_y_size: float = 200e3
    anomaly_z_size: float = 5000
    anomaly_mag: float = -4e-6
    jet_y_size: float = 500e3
    jet_z_size: float = 2e3
    jet_magnitude: float = 35
    jet_y_pos: float = Ly / 2

    @property
    def theta_bar_bottom(self):
        return self.temperature_bottom * (self.p_ref / self.p_bottom)**self.kappa

    @property
    def kappa(self):
        return self.R / self.c_p

    @property
    def f(self):
        return 2 * 7.292e-5 * sin(self.latitude / 180 * pi)

    @property
    def anomaly_x_pos(self):
        return self.Lx / 2

    @property
    def anomaly_y_pos(self):
        return self.Ly / 2

    @property
    def anomaly_z_pos(self):
        return self.trop_height + self.anomaly_z_trop_offset

@dataclass
class SolverParams:
    nx: int = 40
    ny: int = 40
    nz: int = 40
    check_flux: bool = False
    ksp_rtol: float = 1e-9
    ksp_atol: float = 1e-3
    polynomial_order: int = 4
    quadrature_degree: int = None

    @property
    def form_compiler_params(self):
        """Form compiler options shared by every form built from these parameters.

        3p integrates the bilinear form exactly - it is a Q_p Laplacian with Q_p
        coefficients. The linear form is a deliberate truncation: u() and the pv anomaly are
        Gaussians, so no finite degree is exact for them. Validate a change here by rerunning
        a converged (p, N) point and checking the relative error is unmoved.
        """
        degree = self.quadrature_degree
        if degree == ESTIMATE_QUADRATURE_DEGREE:
            return {}
        if degree is None:
            degree = 3 * self.polynomial_order
        return {"quadrature_degree": degree}

    @property
    def matfree_params(self):
        return {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "ksp_rtol": self.ksp_rtol,
            "ksp_atol": self.ksp_atol,
            # p-multigrid for outer preconditioner
            "pc_type": "python",
            "pc_python_type": "firedrake.PMGPC",
            # For p=2,3,4
            "pmg_mg_levels_ksp_type": "chebyshev",
            "pmg_mg_levels_pc_type": "jacobi",
            "pmg_mg_coarse_ksp_type": "preonly",  # Don't iterate, just apply the direct solver once
            "pmg_mg_coarse_pc_type": "python",
            "pmg_mg_coarse_pc_python_type": "firedrake.AssembledPC",  # Force assembly of ONLY the p=1 matrix
            "pmg_mg_coarse_assembled_pc_type": "cholesky",
            "pmg_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
            "pmg_mg_coarse_assembled_mat_mumps_icntl_24": 1,  # detect & null out the known null pivot
            "ksp_converged_reason": None
        }

    @property
    def assembled_mat_params(self): # similar to above but use a fully assembled matrix instead - much faster but uses much more memory
        return {
            "mat_type": "aij",
            "ksp_type": "cg",
            "pc_type": "python",
            "ksp_rtol": self.ksp_rtol,
            "ksp_atol": self.ksp_atol,
            "pc_python_type": "firedrake.PMGPC",
            "pmg_mg_levels_pc_type": "jacobi",
            "pmg_mg_coarse_ksp_type": "preonly",
            "pmg_mg_coarse_pc_type": "hypre",
            "pmg_mg_coarse_pc_hypre_type": "boomeramg",
            "ksp_converged_reason": None
        }

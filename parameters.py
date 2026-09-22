from dataclasses import dataclass
from math import sin, pi, sqrt
import numpy as np

# Latitude-dependent means. Each is (latitudes, values) with latitudes increasing; values
# are held flat outside the tabulated range. Anchor points are stored as the source states
# them, so nothing here is a hand-computed intermediate.

# Annual-mean, zonal-mean thermal tropopause height [m], read off Grise et al. (2010)
# Fig. 1, p. 2278 at 5-degree intervals. See sources/bounds/trop_height.md.
_TROP_HEIGHT_MEAN = ((-50, -45, -40, -35, -30, -25, -20),
                     (10.0e3, 10.5e3, 11.3e3, 12.8e3, 14.5e3, 16.0e3, 16.5e3))

# ponytail: PLACEHOLDER, not from any source yet - replace before citing anything.
# Annual-mean, zonal-mean surface air temperature [K].
_TEMPERATURE_BOTTOM_MEAN = ((-50, -45, -40, -35, -30, -25, -20),
                            (281.15, 284.15, 288.15, 291.15, 294.15, 296.15, 297.15))

# Squared stratospheric buoyancy frequency [1/s^2]: Birner (2006) P28 gives about 4.5e-4
# "in the extratropics" and about 7.0e-4 "at the tropical edge". Interpolated in N^2, not
# in N, since N^2 is the measured quantity. See sources/bounds/N_strat.md.
_N_STRAT_SQ_MEAN = ((-45, -30), (4.5e-4, 7.0e-4))

def _lat_mean(latitude, table):
    """Linear interpolation of a latitude-tabulated mean, clamped outside the table."""
    latitudes, values = table
    return float(np.interp(latitude, latitudes, values))

def trop_height_mean(latitude):
    return _lat_mean(latitude, _TROP_HEIGHT_MEAN)

def temperature_bottom_mean(latitude):
    return _lat_mean(latitude, _TEMPERATURE_BOTTOM_MEAN)

def N_strat_mean(latitude):
    return sqrt(_lat_mean(latitude, _N_STRAT_SQ_MEAN))

def _positive(name, value):
    """Divides by zero somewhere downstream, so catch it here rather than letting a NaN
    propagate silently into a result."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

# Let UFL estimate a quadrature degree.
ESTIMATE_QUADRATURE_DEGREE = -1

@dataclass
class PhysicalParams:
    Lx: float = 5000e3
    Ly: float = 5000e3
    H: float = 20e3
    latitude: float = -42   # Barnes et al. (2022) p. 1293; see sources/bounds/latitude.md
    g: float = 9.80665
    N_strat_variation: float = 0
    N_trop: float = 0.01
    trop_width: float = 1000
    trop_height_variation: float = 0
    temperature_bottom_variation: float = 0
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
    anomaly_clip: float = -1.5e-6
    anomaly_clip_smoothing: float = 0.075 # a fraction of anomaly_mag
    jet_y_size: float = 500e3
    jet_z_size: float = 2e3
    jet_magnitude: float = 35
    jet_y_pos: float = Ly / 2

    @property
    def trop_height(self):
        """Tropopause height: the latitudinal mean, offset by the varied departure from it."""
        return trop_height_mean(self.latitude) + self.trop_height_variation

    @property
    def N_strat(self):
        """Stratospheric buoyancy frequency: the latitudinal mean, offset by the varied departure."""
        return _positive("N_strat",
                         N_strat_mean(self.latitude) + self.N_strat_variation)

    @property
    def temperature_bottom(self):
        """Surface temperature: the latitudinal mean, offset by the varied departure from it."""
        return temperature_bottom_mean(self.latitude) + self.temperature_bottom_variation

    @property
    def theta_bar_bottom(self):
        return self.temperature_bottom * (self.p_ref / self.p_bottom)**self.kappa

    @property
    def kappa(self):
        return self.R / self.c_p

    @property
    def rossby_number(self):
        """Jet Rossby number U/(|f| L). QG assumes this is small (Vallis 2017, p. 177),
        and it grows sharply toward the equator as f shrinks."""
        return self.jet_magnitude / (abs(self.f) * self.jet_y_size)

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

    @property
    def domain_volume(self):
        return self.Lx * self.Ly * self.H

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
            "pmg_mg_coarse_assembled_mat_mumps_icntl_24": 1  # detect & null out the known null pivot
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
            "pmg_mg_coarse_pc_hypre_type": "boomeramg"
        }

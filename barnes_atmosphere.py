from hash_seed import use_same_hash
use_same_hash()
from functools import lru_cache
from atmosphere_builder import *
from firedrake import *
from math_utils import *
from parameters import PhysicalParams

class BarnesAtmosphere(AtmosphereBuilder):
    def __init__(self, domain : DomainBuilder, phys_params : PhysicalParams = None):
        super().__init__(domain, phys_params)

        self.Lx = self.phys_params.Lx
        self.Ly = self.phys_params.Ly
        self.H = self.phys_params.H
        self.kappa = self.ufl_params.kappa

    @lru_cache(maxsize=1) # todo maybe allow height above tropopause of jet to vary
    def psi_bar(self):
        p = self.ufl_params
        return -(p.jet_magnitude * sqrt(pi) * p.jet_y_size / 2) \
            * erf((self.y - p.jet_y_pos) / p.jet_y_size) \
            * exp(-((self.z - p.trop_height) / p.jet_z_size) ** 2)

    def u(self): # zonal jet, a function of y and z
        return -self.psi_bar().dx(1)

    @lru_cache(maxsize=1)
    def v(self): # todo set this back once the firedrake issue has been fixed.
        return Function(self.cg_space)

    def vertical_boundary(self):
        return self.new_vertical_boundary(self.theta_star_init())

    def new_vertical_boundary(self, theta_star : float):
        # theta_star changes with every q, so it goes in as a Constant to keep this kernel reusable
        return self.ufl_params.g * Constant(theta_star) \
            / (self.ufl_params.f * self.theta_bar())

    @lru_cache(maxsize=1)
    def rho_bar(self):
        p_bar = self.p_bar()
        theta_bar = self.theta_bar()
        p_s = self.ufl_params.p_ref
        R = self.ufl_params.R
        kappa = self.kappa

        full_expr = p_bar**(1-kappa) * p_s**kappa / (R * theta_bar)
        fn = Function(self.cg_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def N_bar(self):
        full_expr = scaled_kink(
            self.z,
            self.ufl_params.delta,
            self.ufl_params.N_trop,
            self.ufl_params.N_strat,
            self.ufl_params.trop_width,
            self.ufl_params.trop_height
        )
        fn = Function(self.cg_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def theta_bar(self):
        integral = compute_vertical_integral(self.N_bar() ** 2, self.cg_space)
        full_expr = self.ufl_params.theta_bar_bottom * exp(integral / self.ufl_params.g)
        fn = Function(self.cg_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def p_bar(self):
        integral = compute_vertical_integral(1 / self.theta_bar(), self.cg_space)
        inner_term = (
                (self.kappa * self.ufl_params.g / self.ufl_params.R)
                * (self.ufl_params.p_ref / self.ufl_params.p_bottom) ** self.kappa
                * integral
            )
        return self.ufl_params.p_bottom * (1 - inner_term)**(1/self.kappa)

    @lru_cache(maxsize=1)
    def q_init(self):
        full_expr = ( # convert from ertel pv to qg pv
            self.ertel_pv() * self.rho_bar() * self.ufl_params.g
            / (self.theta_bar() * self.N_bar()**2)
            - self.ufl_params.f
        )
        return Function(self.dg_space).interpolate(full_expr)

    @lru_cache(maxsize=1)
    def geostrophic_vorticity(self):
        return self.v().dx(0) - self.u().dx(1)

    def ertel_from_qgpv(self, q):
        p = self.ufl_params
        return (q + p.f) * self.theta_bar() * (self.N_bar()**2) / (self.rho_bar() * p.g)

    def Q_bar(self):
        return self.ertel_from_qgpv(self.q_bar())

    def q_bar(self):
        f = self.ufl_params.f
        inside_deriv = self.rho_bar() * self.psi_bar().dx(2) / (self.N_bar() **2)
        return self.geostrophic_vorticity() + (f**2) * inside_deriv.dx(2) / self.rho_bar()

    @lru_cache(maxsize=1)
    def ertel_pv(self):
        background = self.Q_bar()

        # Specify anomaly
        ANO_exponent = -((self.z - self.ufl_params.anomaly_z_pos) / self.ufl_params.anomaly_z_size) ** 2 \
                       - ((self.x - self.ufl_params.anomaly_x_pos) / self.ufl_params.anomaly_x_size) ** 2 \
                       - ((self.y - self.ufl_params.anomaly_y_pos) / self.ufl_params.anomaly_y_size) ** 2

        ANO = max_value(Constant(-1.5e-6), self.ufl_params.anomaly_mag * exp(ANO_exponent))

        return background + ANO

    @lru_cache(maxsize=1)
    def theta_star_init(self):
        return self.calc_theta_star(self.q_init())

    def calc_theta_star(self, q):
        n = FacetNormal(self.mesh)
        lateral = (self.rho_bar() * self.v() * n[0] * ds_v((1, 2))
                   - self.rho_bar() * self.u() * n[1] * ds_v((3, 4)))
        numerator = assemble(self.rho_bar() * q * dx - lateral)
        denom = self._const_denom()
        theta_star = numerator / denom

        return theta_star

    @lru_cache(maxsize=1)
    def _const_denom(self): # the denominator in the theta_star calculation - unchanging with q
        N_bar = self.N_bar()
        theta_bar = self.theta_bar()
        rho_bar = self.rho_bar()

        x_mid = self.Lx / 2
        y_mid = self.Ly / 2
        top = [x_mid, y_mid, self.phys_params.H]
        bot = [x_mid, y_mid, 0]

        denom = (
            self.Lx * self.Ly * self.phys_params.g * self.phys_params.f * (
            rho_bar(top) / (N_bar(top) ** 2 * theta_bar(top))
            - rho_bar(bot) / (N_bar(bot) ** 2 * theta_bar(bot))
        ))

        return denom

    @property
    def dxy_min(self):
        return min(
            self.phys_params.Lx / self.solver_params.nx,
            self.phys_params.Ly / self.solver_params.ny
        )
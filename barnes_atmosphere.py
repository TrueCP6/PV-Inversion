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
        self.kappa = self.phys_params.kappa

    @lru_cache(maxsize=1)
    def u(self): # Function of x and z
        exponent = - ((self.x - self.phys_params.anomaly_x_pos) / self.phys_params.jet_x_size) ** 2 \
            - ((self.z - self.phys_params.trop_height) / self.phys_params.jet_z_size) ** 2

        return self.phys_params.jet_magnitude * exp(exponent)

    @lru_cache(maxsize=1)
    def v(self):
        return Function(self.func_space).assign(0)

    # todo check I don't need to evaulate these at z=0,H
    @lru_cache(maxsize=1)
    def top_boundary(self):
        return self.phys_params.g * self.theta_star() \
            / (self.phys_params.f * self.theta_bar())

    @lru_cache(maxsize=1)
    def bottom_boundary(self):
        return self.top_boundary()

    @lru_cache(maxsize=1)
    def rho_bar(self):
        p_bar = self.p_bar()
        theta_bar = self.theta_bar()
        p_s = self.phys_params.p_s
        R = self.phys_params.R
        kappa = self.kappa

        full_expr = p_bar**(1-kappa) * p_s**kappa / (R * theta_bar)
        fn = Function(self.func_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def N_bar(self):
        full_expr = scaled_kink(
            self.z,
            self.phys_params.delta,
            self.phys_params.N_trop,
            self.phys_params.N_strat,
            self.phys_params.trop_width,
            self.phys_params.trop_height
        )
        fn = Function(self.func_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def theta_bar(self):
        integral = compute_vertical_integral(self.N_bar()**2, self.func_space)
        full_expr = self.phys_params.theta_bar_bottom * exp(integral / self.phys_params.g)
        fn = Function(self.func_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def p_bar(self):
        integral = compute_vertical_integral(1/self.theta_bar(), self.func_space)
        inner_term = (
                (self.kappa * self.phys_params.g / self.phys_params.R)
                * (self.phys_params.p_s / self.phys_params.p_bottom)**self.kappa
                * integral
            )
        return self.phys_params.p_bottom * (1 - inner_term)**(1/self.kappa)

    @lru_cache(maxsize=1)
    def q(self):
        full_expr = ( # convert from ertel pv to qg pv
            self.ertel_pv() * self.rho_bar() * self.phys_params.g
            / (self.theta_bar() * self.N_bar()**2)
            - self.phys_params.f
        )
        fn = Function(self.func_space).interpolate(full_expr)
        return fn

    @lru_cache(maxsize=1)
    def geostrophic_vorticity(self):
        return self.v().dx(0) - self.u().dx(1)

    def Q_bar(self):
        # Background state
        vort = self.geostrophic_vorticity()
        background = self.phys_params.f * self.theta_bar() * self.N_bar() ** 2 \
                     / (self.phys_params.g * self.rho_bar()) \
                     * (1 + vort / self.phys_params.f)
        return background

    @lru_cache(maxsize=1)
    def ertel_pv(self):
        background = self.Q_bar()

        # Specify anomaly
        ANO_exponent = -((self.z - self.phys_params.anomaly_z_pos) / self.phys_params.anomaly_z_size) ** 2 \
                       - ((self.x - self.phys_params.anomaly_x_pos) / self.phys_params.anomaly_x_size) ** 2 \
                       - ((self.y - self.phys_params.anomaly_y_pos) / self.phys_params.anomaly_y_size) ** 2
        ANO = self.phys_params.anomaly_mag * exp(ANO_exponent) * 1e-6

        return background + ANO

    @lru_cache(maxsize=1)
    def theta_star(self):
        q = self.q()
        N_bar = self.N_bar()
        theta_bar = self.theta_bar()
        rho_bar = self.rho_bar()

        x_mid = self.Lx / 2
        y_mid = self.Ly / 2
        top = [x_mid, y_mid, self.phys_params.H]
        bot = [x_mid, y_mid, 0]

        denom = (
            self.Lx * self.Ly * self.phys_params.g * self.phys_params.f * (
            rho_bar(top) / (N_bar(top)**2 * theta_bar(top))
            - rho_bar(bot) / (N_bar(bot)**2 * theta_bar(bot))
        ))

        numerator = assemble(rho_bar * q * dx)
        theta_star = numerator / denom

        PETSc.Sys.Print("theta_star = ", theta_star)
        return theta_star
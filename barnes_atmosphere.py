from functools import cache
from atmosphere_builder import *
from firedrake import *
from math_utils import *
from parameters import PhysicalParams

class BarnesAtmosphere(AtmosphereBuilder):
    def __init__(self, mesh: Mesh, function_space: FunctionSpace, phys_params: PhysicalParams):
        super().__init__(mesh, function_space, phys_params)

        self.Lx = phys_params.Lx
        self.Ly = phys_params.Ly
        self.H = phys_params.H
        self.kappa = phys_params.R / phys_params.c_p

    @cache
    def u(self):
        return Constant(4)

    @cache
    def v(self):
        return Constant(0)

    @cache
    def top_boundary(self):
        return self.phys_params.g * self.theta_star() \
            / (self.phys_params.f * self.theta_bar())

    @cache
    def bottom_boundary(self):
        return self.top_boundary()

    @cache
    def rho_bar(self):
        p_bar = self.p_bar()
        theta_bar = self.theta_bar()
        p_s = self.phys_params.p_s
        R = self.phys_params.R
        kappa = self.kappa

        full_expr = p_bar**(1-kappa) * p_s**kappa / (R * theta_bar)
        return Function(self.func_space).interpolate(full_expr)

    @cache
    def N_bar(self):
        full_expr = scaled_kink(
            self.z,
            2,
            self.phys_params.N_trop,
            self.phys_params.N_strat,
            self.phys_params.trop_width,
            self.phys_params.trop_height
        )
        return Function(self.func_space).interpolate(full_expr)

    @cache
    def theta_bar(self):
        integral = compute_vertical_integral(self.N_bar()**2, self.func_space)
        full_expr = self.phys_params.theta_bar_bottom * exp(integral / self.phys_params.g)
        return Function(self.func_space).interpolate(full_expr)

    @cache
    def p_bar(self):
        integral = compute_vertical_integral(1/self.theta_bar(), self.func_space)
        inner_term = (
                (self.kappa * self.phys_params.g / self.phys_params.R)
                * (self.phys_params.p_s / self.phys_params.p_bottom)**self.kappa
                * integral
            )
        return self.phys_params.p_bottom * (1 - inner_term)**(1/self.kappa)


    @cache
    def q(self):
        return ( # convert from ertel pv to qg pv
            self.ertel_pv() * self.rho_bar() * self.phys_params.g
            / (self.theta_bar() * self.N_bar()**2)
            - self.phys_params.f
        )

    @cache
    def ertel_pv(self):
        # temporary approximation for background pv profile
        c1 = self.phys_params.trop_height
        c2 = self.phys_params.H
        c3 = ln(-self.phys_params.q_trop)
        c4 = ln(-self.phys_params.q_max)

        scalar = 1/(c1 - c2)
        A = scalar * (c3 - c4)
        B = scalar * (c1*c4 - c2*c3)

        return -exp(A*self.z + B) * 1e-6

    @cache
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

        return numerator / denom


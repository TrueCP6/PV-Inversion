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
        pass

    @cache
    def bottom_boundary(self):
        pass

    @cache
    def rho_bar(self):
        p_bar = self.p_bar()
        theta_bar = self.theta_bar()
        p_s = self.phys_params.p_s
        R = self.phys_params.R
        kappa = self.kappa

        full_expr = p_bar**(1-kappa) * p_s**kappa / (R * theta_bar)
        return full_expr

    @cache
    def N_bar(self):
        return scaled_kink(
            self.z,
            2,
            self.phys_params.N_trop,
            self.phys_params.N_strat,
            self.phys_params.trop_width,
            self.phys_params.trop_height
        )

    @cache
    def theta_bar(self):
        integral = compute_vertical_integral(self.N_bar()**2, self.func_space)
        return self.phys_params.theta_bar_bottom * exp(integral / self.phys_params.g)

    @cache
    def p_bar(self):
        integral = compute_vertical_integral(1/self.theta_bar(), self.func_space)
        inner_term = (
                (self.kappa * self.phys_params.g / self.phys_params.R)
                * (self.phys_params.p_s / self.phys_params.p_bottom)**self.kappa
                * integral
            )
        full_expr = self.phys_params.p_bottom * (1 - inner_term)**(1/self.kappa)
        return full_expr

    @cache
    def q(self):
        pass

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

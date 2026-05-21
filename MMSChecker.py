from functools import cache

from AtmosphereBuilder import AtmosphereBuilder
from firedrake import *
from Parameters import PhysicalParams

class MMSChecker(AtmosphereBuilder):
    def __init__(self, mesh: Mesh, function_space: FunctionSpace, phys_params: PhysicalParams):
        super().__init__(mesh, function_space, phys_params)

        self.L = self.phys_params.Lx
        if self.L != self.phys_params.Ly:
            raise Exception("Lx must be equal to Ly for testing using MMS")
        self.H = self.phys_params.H

        # todo vaguely realistic parameters here
        self.c_1 = 0
        self.c_2 = 1
        self.c_3 = 1.225
        self.c_4 = ln(0.08803 / self.c_3) / self.H
        self.N = 0.01
        self.c_5 = (self.phys_params.f/self.N)**2
        self.A = (self.c_2 - self.c_1)/(2*self.H)
        self.B = self.c_1

    def u(self):
        return (-2*pi / self.L) * cos(pi * self.x / self.L)

    def v(self):
        return Constant(0)

    def top_boundary(self):
        return Constant(self.c_2)

    def bottom_boundary(self):
        return Constant(self.c_1)

    def rho_bar(self):
        return self.c_3 * exp(self.c_4 * self.z)

    def N_bar(self):
        return Constant(self.N)

    def q(self):
        return (-5 * (pi**2) / (self.L**2)) * cos(pi * self.x / self.L) * sin(2 * pi * self.y / self.L) + self.c_5 * (2*self.A + self.c_4 * (2 * self.A * self.z + self.B))

    @cache
    def psi(self):
        return cos(pi * self.x / self.L) * sin(2*pi*self.y / self.L) + self.A * (self.z**2) + self.B * self.z

    def calc_error(self, psi_numerical: Function):
        # calculate mean offset between numerical and analytical solutions, as we don't know what constant the solver added to psi
        total_offset = assemble((psi_numerical - self.psi()) * dx)
        volume = self.L * self.L * self.H
        mean_offset = total_offset / volume

        # shift psi by that constant we have worked out
        numerical_shifted = Function(self.func_space).assign(psi_numerical - mean_offset)

        absolute_error = errornorm(self.psi(), numerical_shifted, norm_type='L2')

        # compute mean of exact solution
        exact_mean = assemble(self.psi() * dx) / volume
        # shift exact solution, to prevent similar problem to before
        exact_shifted = Function(self.func_space).interpolate(self.psi() - exact_mean)

        exact_norm = norm(exact_shifted, norm_type='L2')

        return absolute_error / exact_norm
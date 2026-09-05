from math_utils import use_same_hash
use_same_hash()
from functools import lru_cache
import math_utils
from atmosphere_builder import AtmosphereBuilder
from firedrake import *
from domain_builder import DomainBuilder
from parameters import PhysicalParams

class MMSChecker(AtmosphereBuilder):
    def __init__(self, domain : DomainBuilder):
        super().__init__(domain)

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

    def vertical_boundary(self):
        return conditional(self.z < self.H/2, self.c_1, self.c_2)

    def rho_bar(self):
        return self.c_3 * exp(self.c_4 * self.z)

    def N_bar(self):
        return Constant(self.N)

    def q_init(self):
        return (-5 * (pi**2) / (self.L**2)) * cos(pi * self.x / self.L) * sin(2 * pi * self.y / self.L) + self.c_5 * (2*self.A + self.c_4 * (2 * self.A * self.z + self.B))

    @lru_cache(maxsize=1)
    def psi(self):
        return cos(pi * self.x / self.L) * sin(2*pi*self.y / self.L) + self.A * (self.z**2) + self.B * self.z

    def calc_error(self, psi_numerical: Function):
        return math_utils.relative_error(self.psi(), psi_numerical)
from hash_seed import use_same_hash
use_same_hash()
from functools import lru_cache
import math_utils
from atmosphere_builder import AtmosphereBuilder
from barnes_atmosphere import BarnesAtmosphere
from firedrake import *
from domain_builder import DomainBuilder

class MMSChecker(AtmosphereBuilder):
    # Fitted amplitude, horizontal and vertical scales and centre height of the anomaly blob
    psi_ano = 8e6
    L_ano = 500e3
    H_ano = 5e3
    z_ano = 8.75e3
    # Fitted vertical shear of the theta* adjustment
    shear = 45.0

    # Simplified coefficient profiles, so that the source term stays short enough to write down
    N = 0.01
    rho_bottom = 1.225
    rho_scale_height = 8.5e3

    def __init__(self, domain : DomainBuilder):
        super().__init__(domain)
        self._barnes = BarnesAtmosphere(domain, self.phys_params)

    @lru_cache(maxsize=1)
    def _blob(self):
        p = self.ufl_params
        return self.psi_ano / (
            1 + ((self.x - p.Lx / 2)**2 + (self.y - p.Ly / 2)**2) / self.L_ano**2
            + ((self.z - self.z_ano) / self.H_ano)**2
        )

    @lru_cache(maxsize=1)
    def psi(self):
        return self._barnes.psi_bar() + self._blob() + self.shear * self.z

    def u(self):
        return -self.psi().dx(1)

    def v(self):
        # The blob is the only term that varies in x, so this is psi_x
        return self._blob().dx(0)

    def vertical_boundary(self):
        return self.psi().dx(2)

    def rho_bar(self):
        return self.rho_bottom * exp(-self.z / self.rho_scale_height)

    def N_bar(self):
        return Constant(self.N)

    def q_init(self):
        psi = self.psi()
        rho, N2, f = self.rho_bar(), self.N_bar()**2, self.ufl_params.f
        return (psi.dx(0).dx(0) + psi.dx(1).dx(1)
                + (f**2 / rho) * (rho * psi.dx(2) / N2).dx(2))

    def calc_error(self, psi_numerical: Function):
        return math_utils.relative_error(self.psi(), psi_numerical)

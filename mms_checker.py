from hash_seed import use_same_hash
use_same_hash()
from functools import lru_cache
import math_utils
from atmosphere_builder import AtmosphereBuilder
from barnes_atmosphere import BarnesAtmosphere
from firedrake import *
from domain_builder import DomainBuilder

class MMSChecker(AtmosphereBuilder):
    def __init__(self, domain : DomainBuilder):
        super().__init__(domain)
        self._barnes = BarnesAtmosphere(domain, self.phys_params)

    @lru_cache(maxsize=1)
    def _blob(self):
        # Stands in for the response to the PV anomaly. A Lorentzian, not a Gaussian: the
        # inversion spreads a Gaussian source into far broader tails. Amplitude and scales are a
        # least-squares fit to the control solution, which psi() reproduces to 3% relative L2.
        p = self.ufl_params
        return 8e6 / (
            1 + ((self.x - p.Lx / 2)**2 + (self.y - p.Ly / 2)**2) / 500e3**2
            + ((self.z - 8.75e3) / 5e3)**2
        )

    @lru_cache(maxsize=1)
    def psi(self):
        # Jet, anomaly response, and the fitted 45 m/s shear of the uniform theta* adjustment
        return self._barnes.psi_bar() + self._blob() + 45.0 * self.z

    def u(self):
        return -self.psi().dx(1)

    def v(self):
        # The blob is the only term that varies in x, so this is psi_x
        return self._blob().dx(0)

    def vertical_boundary(self):
        return self.psi().dx(2)

    def rho_bar(self): # simplified profiles, so the source term stays short enough to write down
        return 1.225 * exp(-self.z / 8.5e3)

    def N_bar(self):
        return Constant(0.01)

    def q_init(self):
        psi = self.psi()
        rho, N2, f = self.rho_bar(), self.N_bar()**2, self.ufl_params.f
        return (psi.dx(0).dx(0) + psi.dx(1).dx(1)
                + (f**2 / rho) * (rho * psi.dx(2) / N2).dx(2))

    def calc_error(self, psi_numerical: Function):
        return math_utils.relative_error(self.psi(), psi_numerical)

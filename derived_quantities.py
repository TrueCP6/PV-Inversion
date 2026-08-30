from firedrake import *
from functools import lru_cache
from barnes_atmosphere import BarnesAtmosphere
from mpi4py import MPI

class ResolvedAtmosphere:
    def __init__(self, psi : Function, atmosphere : BarnesAtmosphere):
        self.psi = psi
        self.atmos = atmosphere
        self.phys_params = atmosphere.phys_params
        self.solver_params = atmosphere.solver_params
        self.func_space = psi.function_space()
        self.mesh = self.func_space.mesh()

    def _interp(self, expr):
        return Function(self.func_space).interpolate(expr)

    @lru_cache(maxsize=1)
    def u(self):
        return self._interp(-self.psi.dx(1))

    @lru_cache(maxsize=1)
    def v(self):
        return self._interp(self.psi.dx(0))

    @lru_cache(maxsize=1)
    def horizontal_wind_speed(self):
        return self._interp(sqrt(self.u()**2 + self.v()**2))

    @lru_cache(maxsize=1)
    def geostrophic_vorticity(self):
        return self._interp(self.v().dx(0) - self.u().dx(1))

    @lru_cache(maxsize=1)
    def potential_temperature_anomaly(self):
        f = self.phys_params.f
        g = self.phys_params.g
        theta_bar = self.atmos.theta_bar()
        psi_z = self.psi.dx(2)

        return self._interp(f * theta_bar * psi_z / g)

    @lru_cache(maxsize=1)
    def potential_temperature(self):
        return self._interp(self.atmos.theta_bar() + self.potential_temperature_anomaly())

    def _psi_0(self):
        pass

    @lru_cache(maxsize=1)
    def pressure_anomaly(self):
        rho_bar = self.atmos.rho_bar()
        f = self.phys_params.f
        psi_0 = self._psi_0()

        return self._interp(rho_bar * f * (self.psi - psi_0))
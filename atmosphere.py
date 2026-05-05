from firedrake import SpatialCoordinate, exp, ln, Function, project
from config import PhysicsConfig, DomainConfig
from math_utils import *

class AtmosphereBackground:
    def __init__(self, mesh, function_space, phys: PhysicsConfig, domain: DomainConfig):
        self.mesh = mesh
        self.phys = phys
        self.domain = domain
        self.x, self.y, self.z = SpatialCoordinate(mesh)
        self.V = function_space

        # Physical constants
        self.g = Constant(phys.g)
        self.f = Constant(phys.f_coriolis)
        self.z_trop = Constant(phys.z_trop)
        self.u = phys.u
        self.v = phys.v

        self._setup_profiles()

    def _setup_profiles(self):
        # Vertical stability
        self.Nbar = scaled_kink(self.z, 2, 0.01, 0.03, 1000, self.z_trop)
        self.Nbar = Function(self.V).interpolate(self.Nbar)

        # Vertical integral of Nbar^2
        self.Nbar_int = compute_vertical_integral(self.Nbar**2, self.V)

        # Potential temperature
        self.thetaBar = Constant(20 + 273.15) * exp((1 / self.g) * self.Nbar_int)
        self.thetaBar = Function(self.V).interpolate(self.thetaBar)

        # Density approximation (TODO: proper rho and p)
        A = self.phys.rho_bot
        B = -ln(self.phys.rho_top / A) / self.domain.z_top
        self.rho = A * exp(-B * self.z)
        self.rho = Function(self.V).interpolate(self.rho)
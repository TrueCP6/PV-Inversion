from firedrake import SpatialCoordinate, Constant, conditional, exp, ln
from config import PhysicsConfig, DomainConfig


class AtmosphereBackground:
    def __init__(self, mesh, phys: PhysicsConfig, domain: DomainConfig):
        self.mesh = mesh
        self.phys = phys
        self.domain = domain
        self.x, self.y, self.z = SpatialCoordinate(mesh)

        # Physical constants
        self.g = Constant(phys.g)
        self.f = Constant(phys.f_coriolis)
        self.f2 = self.f ** 2
        self.z_trop = Constant(phys.z_trop)

        self._setup_profiles()

    def _setup_profiles(self):
        # Vertical stability (TODO: implement kink function)
        self.Nbar = conditional(self.z < self.z_trop, 0.01, 0.03)
        self.Nbar_int = conditional(self.z < self.z_trop, self.z, 9 * self.z - 8 * self.z_trop) * 1e-4

        # Potential temperature
        self.thetaBar = Constant(20 + 273.15) * exp((1 / self.g) * self.Nbar_int)

        # Density approximation (TODO: proper rho and p)
        A = self.phys.rho_bot
        B = -ln(self.phys.rho_top / A) / self.domain.z_top
        self.rho = A * exp(-B * self.z)
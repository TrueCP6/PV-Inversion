from abc import ABC, abstractmethod
from firedrake import Mesh, FunctionSpace, Constant
from ufl import SpatialCoordinate

from domain_builder import DomainBuilder
from parameters import PhysicalParams

class AtmosphereBuilder(ABC):
    def __init__(self, domain : DomainBuilder, phys_params : PhysicalParams = None):
        self.domain = domain
        self.mesh = domain.mesh()
        self.func_space = domain.func_space()
        self.solver_params = domain.solver_params
        self.x, self.y, self.z = SpatialCoordinate(self.mesh)

        if phys_params is None:
            self.phys_params = domain.phys_params
        else:
            self.phys_params = phys_params
    @abstractmethod
    def u(self):
        pass

    @abstractmethod
    def v(self):
        pass

    @abstractmethod
    def vertical_boundary(self):
        pass

    @abstractmethod
    def rho_bar(self):
        pass

    @abstractmethod
    def N_bar(self):
        pass

    @abstractmethod
    def q_init(self):
        pass
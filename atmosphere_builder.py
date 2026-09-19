from abc import ABC, abstractmethod
from firedrake import Mesh, FunctionSpace, Constant
from ufl import SpatialCoordinate

from domain_builder import DomainBuilder
from parameters import PhysicalParams

class ParamConstants:
    """Every PhysicalParams field and property as a firedrake Constant, for use in UFL."""
    def __init__(self, phys_params : PhysicalParams):
        self._phys_params = phys_params
        self._constants = {}

    def __getattr__(self, name):
        if name not in self._constants:
            self._constants[name] = Constant(getattr(self._phys_params, name))
        return self._constants[name]

class AtmosphereBuilder(ABC):
    def __init__(self, domain : DomainBuilder, phys_params : PhysicalParams = None):
        self.domain = domain
        self.mesh = domain.mesh()
        self.cg_space = domain.cg_space()
        self.dg_space = domain.dg_space()
        self.solver_params = domain.solver_params
        self.x, self.y, self.z = SpatialCoordinate(self.mesh)

        if phys_params is None:
            self.phys_params = domain.phys_params
        else:
            self.phys_params = phys_params
        self.ufl_params = ParamConstants(self.phys_params)

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
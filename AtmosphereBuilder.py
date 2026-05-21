from abc import ABC, abstractmethod
from firedrake import Mesh, FunctionSpace, Constant
from ufl import SpatialCoordinate
from Parameters import PhysicalParams

class AtmosphereBuilder(ABC):
    def __init__(self, mesh: Mesh, function_space: FunctionSpace, phys_params: PhysicalParams):
        self.mesh = mesh
        self.func_space = function_space
        self.phys_params = phys_params
        self.x, self.y, self.z = SpatialCoordinate(mesh)

    @abstractmethod
    def u(self):
        pass

    @abstractmethod
    def v(self):
        pass

    @abstractmethod
    def top_boundary(self):
        pass

    @abstractmethod
    def bottom_boundary(self):
        pass

    @abstractmethod
    def rho_bar(self):
        pass

    @abstractmethod
    def N_bar(self):
        pass

    @abstractmethod
    def q(self):
        pass
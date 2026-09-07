from math_utils import use_same_hash
use_same_hash()
from functools import lru_cache
from firedrake import *
from parameters import SolverParams, PhysicalParams
import numpy as np

class DomainBuilder:
    def __init__(self, solver_params : SolverParams, phys_params : PhysicalParams):
        self.solver_params = solver_params
        self.phys_params = phys_params

    @lru_cache(maxsize=1)
    def mesh(self):
        temp_mesh = RectangleMesh(
            self.solver_params.nx, self.solver_params.ny,
            self.phys_params.Lx, self.phys_params.Ly,
            quadrilateral=True # This is crucial for the vertical integrator and sum factorisation
        )

        # Extrude the stretched mesh
        mesh = ExtrudedMesh(
            temp_mesh,
            layers=self.solver_params.nz,
            layer_height=(self.phys_params.H / self.solver_params.nz)
        )
        return mesh

    @lru_cache(maxsize=1)
    def cg_space(self):
        mesh = self.mesh()
        p = self.solver_params.polynomial_order
        return FunctionSpace(mesh, "Q", p)

    @lru_cache(maxsize=1)
    def dg_space(self):
        mesh = self.mesh()
        p = self.solver_params.polynomial_order
        return FunctionSpace(mesh, "DQ", p)
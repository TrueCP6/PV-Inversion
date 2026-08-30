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
    def func_space(self):
        mesh = self.mesh()
        p = self.solver_params.polynomial_order
        V = FunctionSpace(mesh, "Q", p)

        total_dofs = V.dim() # can also be calculated as (degree*N+1)^3
        PETSc.Sys.Print(f"Created function space with {total_dofs} degrees of freedom")

        return V
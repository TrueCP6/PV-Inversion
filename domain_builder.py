from functools import cache
from firedrake import *
from parameters import SolverParams, PhysicalParams
import numpy as np

class DomainBuilder:
    def __init__(self, solver_params : SolverParams, phys_params : PhysicalParams):
        self.solver_params = solver_params
        self.phys_params = phys_params

    @cache
    def mesh(self):
        temp_mesh = RectangleMesh(
            self.solver_params.nx, self.solver_params.ny,
            self.phys_params.Lx, self.phys_params.Ly,
            quadrilateral=self.solver_params.quadrilateral
        )

        # Extract reference to raw coordinates array
        coords = temp_mesh.coordinates.dat.data

        Lx = self.phys_params.Lx
        Ly = self.phys_params.Ly

        # Apply coordinate stretching in both directions
        coords[:, 0] = self._concentrate_centre(coords[:, 0], Lx)
        coords[:, 1] = self._concentrate_centre(coords[:, 1], Ly)

        # Extrude the stretched mesh
        mesh = ExtrudedMesh(
            temp_mesh,
            layers=self.solver_params.nz,
            layer_height=(self.phys_params.H / self.solver_params.nz)
        )
        PETSc.Sys.Print("Built extruded mesh")
        return mesh

    def _concentrate_centre(self, z, L):
        gamma = self.solver_params.gamma
        if gamma == 0:
            return z

        # Map from [0,L] to [-1, 1]
        xi = (2 * z / L) - 1
        # Apply scaling function
        xi_new = np.sinh(gamma * xi) / np.sinh(gamma)
        # Scale back to [0,L]
        return 0.5 * L * (xi_new + 1)

    @cache
    def func_space(self):
        mesh = self.mesh()
        p = self.solver_params.polynomial_order
        V = FunctionSpace(mesh, "Q", p)

        total_dofs = V.dof_dset.layout_vec.getSize()  # can also be calculated as (degree*N+1)^3
        PETSc.Sys.Print(f"Created function space with {total_dofs} degrees of freedom")

        return V
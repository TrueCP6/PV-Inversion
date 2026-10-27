# AI-generated (Claude Code) for the README and videos only. Not part of the thesis code:
# the project's own solvers and experiments are the files in the repository root.
"""Sample Firedrake Functions onto regular grids, so the showcase renders can run without Firedrake."""
import numpy as np
from firedrake import Function, FunctionSpace, VertexOnlyMesh

# Keep sample points off the walls, where round-off can place a point outside every cell
INSET = 1.0  # m


def axis(lo, hi, n):
    return np.linspace(lo + INSET, hi - INSET, n)


class GridSampler:
    """Evaluates Functions at the nodes of a regular grid, returned as an array on rank 0
    (None on the other ranks). The point mesh is built once and reused for every sample."""

    def __init__(self, mesh, *axes):
        self.axes = axes
        self.shape = tuple(a.size for a in axes)
        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=-1)
        vom = VertexOnlyMesh(mesh, points)
        self._at_points = Function(FunctionSpace(vom, "DG", 0))
        self._ordered = Function(FunctionSpace(vom.input_ordering, "DG", 0))

    def __call__(self, f):
        self._at_points.interpolate(f)
        self._ordered.interpolate(self._at_points)
        data = self._ordered.dat.data_ro
        if data.size != np.prod(self.shape):
            return None  # not the rank holding the input ordering
        return data.reshape(self.shape).astype(np.float32)

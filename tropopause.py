"""Locating the dynamical tropopause - the 1.5 PVU surface bounding the stratosphere.

What makes this more than a search for the lowest stratospheric dof is connectivity:
isolated pockets of stratospheric air sit below the tropopause proper and must not be
mistaken for it, so only the region that reaches the model top is counted. Within that
region the surface is snapped to the nearest dof.
"""
from firedrake import Function, SpatialCoordinate
from functools import lru_cache
from mpi4py import MPI
import numpy as np
from scipy.ndimage import label

PVU = 1e-6
DYNAMICAL_TROPOPAUSE = 1.5 * PVU

class ColumnLayout:
    """The dofs of an extruded function space placed on their (x, y, z) lattice.

    The base mesh is a uniform rectangle of quadrilaterals extruded by a constant layer
    height, so the dofs sit on a regular grid and a dof's place in it reads straight off
    its coordinates. That is what lets the stratosphere be found by labelling a boolean
    array rather than by building a graph over the mesh. All of it is fixed by the mesh,
    so it is built once per function space and reused by every solve.
    """
    def __init__(self, V):
        self.comm = V.mesh().comm
        coords = [self._coordinate(V, axis) for axis in range(3)]
        axes = [self._axis(c) for c in coords]
        self.shape = tuple(a.size for a in axes)
        self.z = axes[2]

        i, j, k = (self._index(a, c) for a, c in zip(axes, coords))
        self.ij = (i, j)
        self.k = k

    @staticmethod
    def _coordinate(V, axis):
        return Function(V).interpolate(SpatialCoordinate(V.mesh())[axis]).dat.data_ro_with_halos

    def _axis(self, coord):
        """The distinct coordinates along one axis, in order, as every rank sees them.

        A rank holds only part of the domain, so the values are pooled before the duplicates
        two cells share either side of a face are merged away.
        """
        values = np.unique(np.concatenate(self.comm.allgather(np.unique(coord))))
        distinct = np.diff(values) > 1e-8 * (values[-1] - values[0])
        return values[np.concatenate(([True], distinct))]

    @staticmethod
    def _index(axis, coord):
        """Where each dof sits along that axis, by nearest value rather than exact equality."""
        above = np.clip(np.searchsorted(axis, coord), 1, axis.size - 1)
        return np.where(coord - axis[above - 1] <= axis[above] - coord, above - 1, above)

@lru_cache(maxsize=4)
def column_layout(V):
    return ColumnLayout(V)

def min_height(pv, coriolis):
    """Lowest height of the dynamical tropopause of an ertel pv field, in metres.

    Stratospheric air is |Q| >= 1.5 PVU of the same sign as f, and only the part of it
    joined to the model top counts, so a pocket cut off underneath the tropopause is
    ignored however deep it reaches. The height is that of the lowest dof of that region.
    Collective: every rank returns the same height.
    """
    layout = column_layout(pv.function_space())
    stratospheric = pv.dat.data_ro_with_halos * np.sign(coriolis) >= DYNAMICAL_TROPOPAUSE

    # Pool the local dofs onto the lattice. Halos and partition boundaries overlap, and the
    # ranks that share a dof agree on it, so OR-ing is enough to assemble the whole domain.
    grid = np.zeros(layout.shape, dtype=bool)
    grid[layout.ij[0], layout.ij[1], layout.k] = stratospheric
    layout.comm.Allreduce(MPI.IN_PLACE, grid, op=MPI.LOR)

    # Face connectivity is what the label call defaults to, and it is the right one: air is
    # joined through the lattice, not across its diagonals.
    region, _ = label(grid)
    joined = np.isin(region, np.unique(region[:, :, -1])) & grid

    reached = joined.any(axis=2)
    if not reached.any():
        return float(layout.z[-1])  # no stratosphere in the domain

    return float(layout.z[joined.argmax(axis=2)[reached]].min())

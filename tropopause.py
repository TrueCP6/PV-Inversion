"""Locating the dynamical tropopause - the 1.5 PVU surface bounding the stratosphere.

Two things make this more than a search for the lowest stratospheric dof. Isolated pockets
of stratospheric air sit below the tropopause proper and must not be mistaken for it, so
only the |Q| = 1.5 PVU contour bounding the region that reaches the model top is counted;
and Q is a degree p polynomial inside each element, so the contour is found by root finding
along the column rather than snapped to the nearest dof.
"""
from firedrake import Function, SpatialCoordinate
from functools import lru_cache
from mpi4py import MPI
import numpy as np
from scipy.ndimage import label

PVU = 1e-6
DYNAMICAL_TROPOPAUSE = 1.5 * PVU

# Enough halvings to put a bracket of several hundred metres well under a micrometre, and
# few enough that the bracket never closes to the point where its midpoint rounds onto an
# end - the ends are dofs, which the interpolant below may not be evaluated at.
BISECTION_STEPS = 40

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
        self.degree = V.ufl_element().degree()[1]

        coords = [self._coordinate(V, axis) for axis in range(3)]
        axes = [self._axis(c) for c in coords]
        self.shape = tuple(a.size for a in axes)
        self.z = axes[2]

        i, j, k = (self._index(a, c) for a, c in zip(axes, coords))
        self.ij = (i, j)
        self.k = k

        # Ranks only ever split the base mesh, so every column a rank touches is whole and
        # sorting the local dofs by (i, j, k) lays them out one whole column at a time.
        order = np.lexsort((k, j, i))
        self.dofs = order.reshape(-1, self.shape[2])
        self.column = (i[self.dofs[:, 0]], j[self.dofs[:, 0]])
        assert np.array_equal(k[self.dofs], np.broadcast_to(np.arange(self.shape[2]), self.dofs.shape)), \
            "every local column must be whole"

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
    ignored however deep it reaches. Collective: every rank returns the same height.
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

    bottom = joined.argmax(axis=2)  # the lowest joined level, and so the foot of the surface
    reached = joined.any(axis=2)
    if not reached.any():
        return float(layout.z[-1])  # no stratosphere in the domain
    if (bottom[reached] == 0).any():
        return 0.0  # stratospheric down to the ground

    return _lowest_crossing(layout, pv, coriolis, bottom, reached)

def _lowest_crossing(layout, pv, coriolis, bottom, reached):
    """The lowest height at which the stratosphere is entered, reduced over every rank.

    The crossing lies between the last tropospheric level and the first stratospheric one,
    so a column whose bracket starts above every other bracket's end cannot hold the
    minimum. Dropping those leaves a handful of columns to root find in.
    """
    ceiling = layout.z[bottom[reached]].min()
    foot = bottom[layout.column]
    mine = np.nonzero(reached[layout.column] & (layout.z[foot - 1] < ceiling))[0]

    local = _refine(layout, pv, coriolis, mine, foot[mine]).min(initial=np.inf)
    return float(layout.comm.allreduce(local, op=MPI.MIN))

def _refine(layout, pv, coriolis, column, bottom):
    """Height of the 1.5 PVU crossing in each column, to sub-element accuracy.

    Between its two bracketing levels Q is the degree p polynomial the element carries, so
    bisecting that polynomial resolves the crossing continuously instead of rounding it to
    whichever dof happens to lie nearest.
    """
    p = layout.degree
    levels = ((bottom - 1) // p)[:, None] * p + np.arange(p + 1)  # the element spanning the bracket
    nodes = layout.z[levels]
    values = pv.dat.data_ro_with_halos[layout.dofs[column[:, None], levels]] * np.sign(coriolis)

    below, above = layout.z[bottom - 1], layout.z[bottom]
    for _ in range(BISECTION_STEPS):
        middle = (below + above) / 2
        stratospheric = _interpolate(nodes, values, middle) >= DYNAMICAL_TROPOPAUSE
        above = np.where(stratospheric, middle, above)
        below = np.where(stratospheric, below, middle)

    return above

def _interpolate(nodes, values, at):
    """The degree p interpolant through each row of (nodes, values), at one height per row.

    The barycentric form needs `at` to miss every node, which it does: it is always strictly
    inside a bracket whose ends are themselves nodes.
    """
    gaps = nodes[:, :, None] - nodes[:, None, :]
    diagonal = np.arange(nodes.shape[1])
    gaps[:, diagonal, diagonal] = 1
    scaled = 1 / (np.prod(gaps, axis=2) * (at[:, None] - nodes))
    return (scaled * values).sum(axis=1) / scaled.sum(axis=1)

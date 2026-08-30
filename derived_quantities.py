from firedrake import *
from functools import lru_cache
from barnes_atmosphere import BarnesAtmosphere
from mpi4py import MPI
import numpy as np

class ResolvedAtmosphere:
    def __init__(self, psi : Function, atmosphere : BarnesAtmosphere):
        self.psi = psi
        self.atmos = atmosphere
        self.phys_params = atmosphere.phys_params
        self.solver_params = atmosphere.solver_params
        self.func_space = psi.function_space()
        self.mesh = self.func_space.mesh()

    def _interp(self, expr):
        return Function(self.func_space).interpolate(expr)

    @lru_cache(maxsize=1)
    def u(self):
        return self._interp(-self.psi.dx(1))

    @lru_cache(maxsize=1)
    def v(self):
        return self._interp(self.psi.dx(0))

    @lru_cache(maxsize=1)
    def horizontal_wind_speed(self):
        return self._interp(sqrt(self.u()**2 + self.v()**2))

    @lru_cache(maxsize=1)
    def geostrophic_vorticity(self):
        return self._interp(self.v().dx(0) - self.u().dx(1))

    @lru_cache(maxsize=1)
    def potential_temperature_anomaly(self):
        f = self.phys_params.f
        g = self.phys_params.g
        theta_bar = self.atmos.theta_bar()
        psi_z = self.psi.dx(2)

        return self._interp(f * theta_bar * psi_z / g)

    @lru_cache(maxsize=1)
    def potential_temperature(self):
        return self._interp(self.atmos.theta_bar() + self.potential_temperature_anomaly())

    def _z_levels(self):
        """The z level each DoF of func_space sits on, as (number of levels, level index).
        The index covers the local DoFs including the halo; data_ro is its owned prefix.
        """
        z = self._interp(SpatialCoordinate(self.mesh)[2]).dat.data_ro_with_halos

        z_sorted = np.sort(z)
        tol = 1e-8 * (z_sorted[-1] - z_sorted[0])
        levels = z_sorted[np.concatenate(([True], np.diff(z_sorted) > tol))]

        # Every rank holds whole columns, so it sees every level and numbers them alike.
        expected = self.func_space.ufl_element().degree()[1] * (self.mesh.layers - 1) + 1
        assert levels.size == expected, f"found {levels.size} z levels, expected {expected}"

        above = np.clip(np.searchsorted(levels, z), 1, levels.size - 1)
        index = np.where(z - levels[above - 1] <= levels[above] - z, above - 1, above)

        return levels.size, index

    def _psi_0(self):
        r"""The lateral boundary average of psi at each height,

            \psi_0(z) = \frac{1}{2(L_x + L_y)} \oint_{\partial B(z)} \psi(x, y, z) ds,

        as a Function on func_space that varies with z alone: every column is written the
        same profile, so psi_0 is horizontally uniform to the last bit rather than to some
        tolerance, and no horizontal structure can leak into the pressure anomaly.
        """
        n_levels, level = self._z_levels()

        boundary_mass = assemble(
            TestFunction(self.func_space) * ds_v,
            form_compiler_parameters=self.solver_params.form_compiler_params
        )

        # data_ro is owned DoFs only, so the reduction counts each one once even though
        # ranks share the columns along their partition boundaries.
        weights = boundary_mass.dat.data_ro
        psi = self.psi.dat.data_ro
        owned_level = level[:weights.size]

        local = np.concatenate((
            np.bincount(owned_level, weights=weights * psi, minlength=n_levels),
            np.bincount(owned_level, weights=weights, minlength=n_levels)
        ))
        totals = np.empty_like(local)
        self.mesh.comm.Allreduce(local, totals, op=MPI.SUM)

        profile = totals[:n_levels] / totals[n_levels:]

        psi_0 = Function(self.func_space, name="psi_0")
        # Fill the halo as well: its columns carry the same profile as the owned ones.
        psi_0.dat.data_with_halos[:] = profile[level]

        return psi_0

    @lru_cache(maxsize=1)
    def pressure_anomaly(self):
        rho_bar = self.atmos.rho_bar()
        f = self.phys_params.f
        psi_0 = self._psi_0()

        return self._interp(rho_bar * f * (self.psi - psi_0))

    @lru_cache(maxsize=1)
    def pressure_anomaly_hpa(self):
        return self._interp(1e-2 * self.pressure_anomaly())

    @lru_cache(maxsize=1)
    def pressure_hpa(self):
        p_bar_hpa = self.atmos.p_bar() * 1e-2
        return self._interp(p_bar_hpa + self.pressure_anomaly_hpa())

    @lru_cache(maxsize=1)
    def temperature_anomaly(self):
        p_bar = self.atmos.p_bar()
        p_s = self.phys_params.p_s
        kappa = self.phys_params.kappa
        theta_star = self.potential_temperature_anomaly()
        theta_bar = self.atmos.theta_bar()
        p_star = self.pressure_anomaly()

        expr = (p_bar / p_s)**kappa * (theta_star + kappa * theta_bar * p_star / p_bar)
        return self._interp(expr)

    # todo add function to calculate min surface vorticity and min surface pressure
    # todo add function to calculate min dynamical tropopause height
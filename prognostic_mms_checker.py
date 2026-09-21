from hash_seed import use_same_hash
use_same_hash()
import unittest
import numpy as np
from firedrake import *
from parameters import SolverParams, PhysicalParams
from prognostic_solver import PrognosticSolver

class PrognosticMMSChecker(PrognosticSolver):
    """Prescribes the velocity instead of inverting q for it, and adds the source that makes
    q_exact solve dq/dt + u.grad(q) = S. This checks the transport and RK4 alone - the
    inversion has its own MMS in mms_checker.py.

    u is non-divergent, and it crosses every lateral boundary, exercising both inflow and outflow."""
    def __init__(self, solver_params : SolverParams, phys_params : PhysicalParams):
        self.L = phys_params.Lx
        if self.L != phys_params.Ly:
            raise Exception("Lx must be equal to Ly for testing using MMS")
        self.H = phys_params.H
        self.U0, self.U1 = 20.0, 10.0
        self.V0, self.V1 = 10.0, 5.0
        super().__init__(solver_params, phys_params, matfree=True)

    def u_exact(self):
        x, y, z = SpatialCoordinate(self._dg_space.mesh())
        return self.U0 + self.U1 * sin(2 * pi * y / self.L)

    def v_exact(self):
        x, y, z = SpatialCoordinate(self._dg_space.mesh())
        return self.V0 + self.V1 * sin(2 * pi * x / self.L)

    def q_exact(self, t):
        x, y, z = SpatialCoordinate(self._dg_space.mesh())
        # Time dependence is confined to a bump that vanishes on the lateral boundary, so the
        # inflow data is constant in time. Time-dependent inflow imposed at the RK4 stage times
        # triggers order reduction (Carpenter et al. 1995) - a moving wave here floored the
        # error at ~3e-7 for p=4. The production solver freezes q_in, so this matches it.
        bump = sin(pi * x / self.L)**2 * sin(pi * y / self.L)**2
        return sin(2 * pi * x / self.L) * cos(2 * pi * y / self.L) * (1 + z / self.H) + bump * sin(2 * pi * t / 2e4)

    def _q_initial(self):
        return self.q_exact(0.0)

    def _q_in(self):
        return self.q_exact(self._t)

    def _source(self):
        t = variable(self._t)
        q = self.q_exact(t)
        return diff(q, t) + self.u_exact() * q.dx(0) + self.v_exact() * q.dx(1)

    def _velocity(self):
        return as_vector([self.u_exact(), self.v_exact(), 0])

    def _update_velocity(self, q):
        pass # u is prescribed and steady, so there is nothing to bring back into step

    def calc_error(self):
        # Not math_utils.relative_error: that removes the mean, which q (unlike psi) must get right
        exact = self.q_exact(self.t)
        return errornorm(exact, self.q) / norm(Function(self._dg_space).interpolate(exact))

    def run(self, T):
        self._update_velocity(self.q)
        n_steps = int(np.ceil(T / self.dt()))
        for _ in range(n_steps):
            self.step(T / n_steps) # land exactly on T
        return self.calc_error()
from math_utils import use_same_hash
use_same_hash()
from firedrake import *
import numpy as np
from domain_builder import DomainBuilder
from barnes_atmosphere import BarnesAtmosphere
from diagnostic_solver import DiagnosticSolver
from derived_quantities import ResolvedAtmosphere
from math_utils import get_global_max, get_global_min
from parameters import PhysicalParams, SolverParams

class PrognosticSolver:
    """Evolves q and theta_star forward in time under

        dq/dt + u dq/dx + v dq/dy = 0
        d(theta_star)/dt + u d(theta_star)/dx + v d(theta_star)/dy = 0

    with (u, v) = (-dpsi/dy, dpsi/dx) diagnosed from the current (q, theta_star)
    via DiagnosticSolver at every RK4 stage. q and theta_star live in the DQ
    space, so both are advected with an upwind DG flux.
    """

    def __init__(self, phys_params: PhysicalParams, solver_params: SolverParams, mat_free: bool = False):
        self.phys_params = phys_params
        self.solver_params = solver_params

        self.domain = DomainBuilder(solver_params, phys_params)
        self.atmos = BarnesAtmosphere(self.domain, phys_params)
        self.mesh = self.domain.mesh()
        self.dg_space = self.domain.dg_space()

        self.diagnostic_solver = DiagnosticSolver(self.atmos, mat_free)

        self.q = Function(self.dg_space).interpolate(self.atmos.q_init())
        self.theta_star = Function(self.dg_space).interpolate(self.atmos.theta_star_init())

        self.time = 0.0

        # Placeholders the cached tendency solvers below read from; _tendency()
        # just re-interpolates into these and re-solves, rather than rebuilding
        # the form (and re-doing PETSc/TSFC setup) on every RK4 stage. u, v are
        # shared since both fields are advected by the same diagnosed velocity.
        self._tend_u = Function(self.atmos.cg_space)
        self._tend_v = Function(self.atmos.cg_space)

        # q is a PV *anomaly*, ~0 away from the jet/PV anomaly, so "nothing flows
        # in" (zero exterior value) is the sane default at inflow boundaries.
        # theta_star is not an anomaly - its typical value is the ~-0.15
        # compatibility constant from calc_theta_star, not 0 - so the same
        # assumption there would inject a fake ambient value at every inflow
        # boundary. Use the interior trace instead: a field that's already
        # uniform at the boundary then sees no gradient and stays exactly put.
        self._q_tend_field, self._q_tend_result, self._q_tendency_solver = \
            self._build_tendency_solver(zero_exterior_value=True)
        self._theta_tend_field, self._theta_tend_result, self._theta_tendency_solver = \
            self._build_tendency_solver(zero_exterior_value=False)

    def _velocity(self, q, theta_star):
        """Diagnose (u, v) from the streamfunction implied by an intermediate
        (q, theta_star) RK4 stage, via the existing DiagnosticSolver."""
        self.diagnostic_solver.step(q, theta_star)
        self.diagnostic_solver.solve_psi()
        resolved = ResolvedAtmosphere(self.diagnostic_solver.psi_soln, self.atmos)
        return resolved.u(), resolved.v()

    def _build_tendency_solver(self, zero_exterior_value: bool):
        """Weak, upwind-DG form of df/dt = -(u df/dx + v df/dy), built once
        against placeholder Functions that _tendency() later re-interpolates
        into (see the __init__ comment for why q and theta_star each get their
        own solver, differing only in the exterior boundary term).

        field lives in the discontinuous DQ space, so its gradient is only
        defined cell-by-cell - there's no single well-defined value of df/dx,
        df/dy at an inter-element face. Getting a scheme where neighbouring
        cells actually exchange information means integrating by parts onto
        the test function instead, which trades the missing face gradient for
        a face *value* of field - still two-valued (one per side) - and that
        second ambiguity is resolved by picking the upwind side's value
        (info should flow from upstream to downstream, matching the sign of
        u.n on that face).

        Integrating by parts moves the time derivative under the test
        function on the left as a mass matrix (p*dfdt*dx) rather than
        leaving df/dt free at each node, so recovering df/dt as a Function
        means solving that mass-matrix system against the assembled
        right-hand side below - it isn't available by simply evaluating an
        expression. The DQ mass matrix has no coupling between cells (every
        basis function is supported on a single element), so it factors
        exactly, one cell-local block at a time, in a single PC application -
        "ilu" here does no approximation, that's just PETSc's name for the
        one-shot block factorisation.
        """
        V = self.dg_space
        p = TestFunction(V)
        dfdt = TrialFunction(V)
        n = FacetNormal(self.mesh)
        velocity = as_vector([self._tend_u, self._tend_v, 0])
        un = 0.5 * (dot(velocity, n) + abs(dot(velocity, n)))  # = (u.n)+, upwind switch between two interior traces
        field = Function(V)

        exterior_term = un if zero_exterior_value else dot(velocity, n)

        a_mass = p * dfdt * dx
        L = (
            dot(grad(p), velocity) * field * dx
            - (un('+') * field('+') - un('-') * field('-')) * jump(p) * dS_v
            - exterior_term * field * p * ds_v
        )

        result = Function(V)
        problem = LinearVariationalProblem(a_mass, L, result, constant_jacobian=True)
        solver = LinearVariationalSolver(problem, solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        })
        return field, result, solver

    def _tendency(self, field, u, v, zero_exterior_value: bool):
        field_ph, result, solver = (self._q_tend_field, self._q_tend_result, self._q_tendency_solver) \
            if zero_exterior_value else \
            (self._theta_tend_field, self._theta_tend_result, self._theta_tendency_solver)

        field_ph.interpolate(field)
        self._tend_u.interpolate(u)
        self._tend_v.interpolate(v)
        solver.solve()
        return result.copy(deepcopy=True)

    def _rk4_stage(self, q, theta_star):
        u, v = self._velocity(q, theta_star)
        return (
            self._tendency(q, u, v, zero_exterior_value=True),
            self._tendency(theta_star, u, v, zero_exterior_value=False),
        )

    def step(self, dt):
        """Advance (q, theta_star) by one explicit RK4 step of size dt.

        See CFL_RESEARCH_NOTES.md for the stability limit on dt - not enforced
        here, the caller picks dt.
        """
        q0, th0 = self.q, self.theta_star

        k1_q, k1_th = self._rk4_stage(q0, th0)
        k2_q, k2_th = self._rk4_stage(q0 + dt / 2 * k1_q, th0 + dt / 2 * k1_th)
        k3_q, k3_th = self._rk4_stage(q0 + dt / 2 * k2_q, th0 + dt / 2 * k2_th)
        k4_q, k4_th = self._rk4_stage(q0 + dt * k3_q, th0 + dt * k3_th)

        self.q = Function(self.dg_space).interpolate(
            q0 + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q))
        self.theta_star = Function(self.dg_space).interpolate(
            th0 + dt / 6 * (k1_th + 2 * k2_th + 2 * k3_th + k4_th))

        self.time += dt
        return self.q, self.theta_star


def demo():
    """Self-checks for the DG upwind advection + RK4 combination in _tendency.

    Runs against a prescribed constant velocity (bypassing the psi -> u, v
    diagnosis) so it's fast and deterministic, exercising the same
    sign-sensitive upwind flux and RK4 wiring `step()` uses without depending
    on DiagnosticSolver converging (see the conversation note on q/theta_star
    flux compatibility - not exercised here).
    """
    solver_params = SolverParams(nx=8, ny=8, nz=2, polynomial_order=2, check_flux=False)
    phys_params = PhysicalParams()
    solver = PrognosticSolver(phys_params, solver_params, mat_free=True)

    Lx, Ly = phys_params.Lx, phys_params.Ly
    x, y, z = SpatialCoordinate(solver.mesh)

    u_speed = 30.0
    u_const = Function(solver.atmos.cg_space).assign(u_speed)
    v_const = Function(solver.atmos.cg_space).assign(0.0)

    def rk4_advect(field, dt, n_steps, zero_exterior_value):
        for _ in range(n_steps):
            k1 = solver._tendency(field, u_const, v_const, zero_exterior_value)
            k2 = solver._tendency(field + dt / 2 * k1, u_const, v_const, zero_exterior_value)
            k3 = solver._tendency(field + dt / 2 * k2, u_const, v_const, zero_exterior_value)
            k4 = solver._tendency(field + dt * k3, u_const, v_const, zero_exterior_value)
            field = Function(solver.dg_space).interpolate(field + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        return field

    # 1) A q-like anomaly bump (interior, zero-exterior-value boundary) under
    #    solid-body translation should keep its mass and move by u*dt*n_steps.
    x0, y0, sigma = Lx * 0.3, Ly * 0.5, Lx * 0.08
    bump = Function(solver.dg_space).interpolate(exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2))
    total_before = assemble(bump * dx)
    centroid_before = assemble(x * bump * dx) / total_before

    dt, n_steps = 3000.0, 10
    bump = rk4_advect(bump, dt, n_steps, zero_exterior_value=True)

    assert np.isfinite(bump.dat.data_ro).all(), "advection blew up"

    total_after = assemble(bump * dx)
    rel_mass_change = abs(total_after - total_before) / total_before
    PETSc.Sys.Print(f"Bump: relative mass change over {n_steps} steps: {rel_mass_change}")
    assert rel_mass_change < 1e-3, "a divergence-free translation should conserve mass"

    centroid_after = assemble(x * bump * dx) / total_after
    expected_shift = u_speed * dt * n_steps
    rel_shift_error = abs((centroid_after - centroid_before) - expected_shift) / expected_shift
    PETSc.Sys.Print(f"Bump: relative translation-distance error: {rel_shift_error}")
    assert rel_shift_error < 1e-2, "bump should translate at exactly u_speed"

    # 2) A theta_star-like uniform ambient field (interior-trace boundary) has
    #    zero gradient everywhere, including at the domain edge, so it must
    #    stay exactly uniform under pure advection - this is what broke before
    #    the interior-trace boundary fix (see the conversation).
    uniform = Function(solver.dg_space).assign(-0.15)
    uniform = rk4_advect(uniform, dt, n_steps, zero_exterior_value=False)
    drift = get_global_max(uniform) - get_global_min(uniform)
    PETSc.Sys.Print(f"Uniform field: spread after {n_steps} steps: {drift}")
    assert drift < 1e-10, "a uniform field must stay exactly uniform under pure advection"

    PETSc.Sys.Print("prognostic_solver smoke test passed")


if __name__ == "__main__":
    demo()

import math_utils
from barnes_atmosphere import BarnesAtmosphere
from diagnostic_solver import DiagnosticSolver
from domain_builder import DomainBuilder
from parameters import SolverParams, PhysicalParams
from firedrake import *

class PrognosticSolver:
    def __init__(self, solver_params : SolverParams, phys_params : PhysicalParams, matfree : bool):
        self._solver_params = solver_params
        self._phys_params = phys_params

        domain = DomainBuilder(self._solver_params, self._phys_params)
        self._cg_space = domain.cg_space()
        self._dg_space = domain.dg_space()

        self._atmos = BarnesAtmosphere(domain)
        self._matfree = matfree

        self.t = 0.0
        self._t = Constant(0.0) # time of the stage being evaluated, seen by _q_in and _source
        self._u = Function(self._cg_space)
        self._v = Function(self._cg_space)
        self.q = Function(self._dg_space).interpolate(self._q_initial()) # prognostic state
        self._temp_q = Function(self._dg_space) # RHS input
        self._stage_q = Function(self._dg_space) # RHS output
        self._prognostic_solver = self._build_solver()

    # Hooks - override these to drive the transport with something other than the Barnes
    # atmosphere (see prognostic_mms_checker.py)
    def _q_initial(self):
        return self._atmos.q_init()

    def _q_in(self): # keep q_in frozen for boundaries in the prognostic equation
        return self._atmos.q_init()

    def _source(self):
        return Constant(0.0)

    def _update_velocity(self, q):
        if not hasattr(self, "_diag_solver"): # built on first use, so overrides never pay for it
            self._diag_solver = DiagnosticSolver(self._atmos, self._matfree)
        self._diag_solver.update_q(q)
        self._diag_solver.solve_psi()
        psi = self._diag_solver.psi_soln
        self._u.interpolate(-psi.dx(1))
        self._v.interpolate(psi.dx(0))

    def _build_solver(self):
        V = self._dg_space
        u = as_vector([self._u, self._v, 0])
        q = self._temp_q
        q_in = self._q_in()

        trial = TrialFunction(V)
        phi = TestFunction(V)

        a = phi * trial * dx

        n = FacetNormal(V.mesh())
        un = Constant(0.5)*(dot(u,n) + abs(dot(u,n)))

        L = (q * div(phi * u) * dx
            + phi * self._source() * dx
            - conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds_v
            - conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds_v
            - (phi('+') - phi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS_v) # u has no vertical component, so horizontal facets contribute nothing

        form_compiler_params = self._solver_params.form_compiler_params
        problem = LinearVariationalProblem(
            a, L, self._stage_q,
            constant_jacobian=True,
            form_compiler_parameters=form_compiler_params
        )

        params = {'mat_type': 'matfree', 'ksp_type': 'cg', 'ksp_rtol': 1e-12, 'ksp_atol': 0, 'pc_type': 'jacobi'}

        return LinearVariationalSolver(problem, solver_parameters=params)

    def dt(self):
        dx = self._atmos.dxy_min # CFL is set by the smallest cell
        vel = Function(self._cg_space).interpolate(sqrt(self._u**2 + self._v**2))
        max_vel = math_utils.get_global_max(vel)
        p = self._solver_params.polynomial_order

        safety = 0.8

        return safety * dx / (max_vel*(2*p+1))

    def RHS(self, q, t=None) -> Function:
        self._t.assign(self.t if t is None else t)
        self._update_velocity(q)

        # Now solve the mass matrix
        self._temp_q.interpolate(q)
        self._prognostic_solver.solve()
        return self._stage_q.copy(deepcopy=True)

    def step(self, dt=None):
        q, t = self.q, self.t
        k1 = self.RHS(q, t) # also sets u, v for the current state, which dt() needs
        if dt is None:
            dt = self.dt()
        k2 = self.RHS(q + dt/2 * k1, t + dt/2)
        k3 = self.RHS(q + dt/2 * k2, t + dt/2)
        k4 = self.RHS(q + dt * k3, t + dt)
        q.assign(q + dt/6 * (k1 + 2*k2 + 2*k3 + k4))
        self.t = t + dt
        return dt

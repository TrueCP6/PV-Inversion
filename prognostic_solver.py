import math_utils
from barnes_atmosphere import BarnesAtmosphere
from diagnostic_solver import DiagnosticSolver
from domain_builder import DomainBuilder
from parameters import SolverParams, PhysicalParams
from firedrake import *

# Fraction of the CFL limit a step is taken at, unless the caller asks for another.
SAFETY = 0.6

class ZhangShuLimiter:
    """Zhang & Shu's (2010) bound-preserving limiter: scales q about each cell's mean, just
    far enough to bring the cell back inside [lower, upper].

    The mean is left alone, so mass is conserved, and so is the order of accuracy wherever q
    is smooth and already in bounds (theta = 1 there). The bounds are the extremes of the
    initial q: with no source and q_in frozen at the initial q, the exact q never leaves them.

    The bounds are checked at Gauss-Lobatto nodes rather than at q's own nodes. DQ's default
    nodes are Gauss-Legendre points, all interior, so they miss the cell edges the upwind flux
    reads. Zhang & Shu's guarantee (under SSP stepping and their CFL condition) is for a
    specific point set on the edges - Gauss-Legendre along a face, Gauss-Lobatto across it -
    which the GLL grid only approximates, so here it is close to bound-preserving, not exactly.
    """
    def __init__(self, q, p):
        mesh = q.function_space().mesh()
        self._nodes = Function(FunctionSpace(mesh, "DQ", p, variant="gll"))
        self._limited = Function(q.function_space())

        W0 = FunctionSpace(mesh, "DQ", 0)
        self._test = TestFunction(W0)
        self._volume = assemble(self._test * dx)
        self._domain_volume = assemble(Constant(1.0) * dx(domain=mesh))
        self._cell_sum = Cofunction(W0.dual())
        self._mean, self._min, self._max, self._theta = (Function(W0) for _ in range(4))

        self._nodes.interpolate(q)
        self.lower, self.upper = math_utils.get_global_extrema(self._nodes)
        self.limited_fraction = 0.0 # of the domain's volume, in the last apply()

    def apply(self, q):
        self._nodes.interpolate(q)
        self._min.assign(float('inf'))
        self._max.assign(float('-inf'))
        instructions = """
        for i
            lo[0] = fmin(lo[0], f[i])
            hi[0] = fmax(hi[0], f[i])
        end
        """
        par_loop(("{[i]: 0 <= i < f.dofs}", instructions), dx,
                 {"lo": (self._min, RW), "hi": (self._max, RW), "f": (self._nodes, READ)})

        assemble(q * self._test * dx, tensor=self._cell_sum)
        self._mean.dat.data_wo[:] = self._cell_sum.dat.data_ro / self._volume.dat.data_ro

        lo, hi, mean = self._min, self._max, self._mean
        L, U = self.lower, self.upper
        # A mean already out of bounds can't be fixed by scaling about it, so that cell goes flat (theta = 0)
        upper = conditional(hi > U, conditional(mean < U, (U - mean) / (hi - mean), 0.0), 1.0)
        lower = conditional(lo < L, conditional(mean > L, (L - mean) / (lo - mean), 0.0), 1.0)
        self._theta.interpolate(max_value(0.0, min_value(1.0, min_value(upper, lower))))

        self._limited.interpolate(mean + self._theta * (q - mean))
        q.assign(self._limited)

        self.limited_fraction = assemble(conditional(self._theta < 1.0, 1.0, 0.0) * dx) / self._domain_volume
        return q

class PrognosticSolver:
    def __init__(self, solver_params : SolverParams, phys_params : PhysicalParams, matfree : bool,
                 limit : bool = True):
        self._solver_params = solver_params
        self._phys_params = phys_params

        domain = DomainBuilder(self._solver_params, self._phys_params)
        self._dg_space = domain.dg_space()

        self._atmos = BarnesAtmosphere(domain)
        self._matfree = matfree

        self.t = 0.0
        self._t = Constant(0.0) # time of the stage being evaluated, seen by _q_in and _source
        self._dt = Constant(0.0) # the step's dt changes with the CFL limit, so it goes in UFL as a Constant
        self.q = Function(self._dg_space).interpolate(self._q_initial()) # prognostic state
        self._temp_q = Function(self._dg_space) # RHS input
        self._stage_q = Function(self._dg_space) # RHS output
        self._q1 = Function(self._dg_space) # SSPRK3 stages
        self._q2 = Function(self._dg_space)
        # Off for anything with a source (the MMS): q is then free to leave its initial bounds
        self.limiter = ZhangShuLimiter(self.q, solver_params.polynomial_order) if limit else None
        self._prognostic_solver = self._build_solver()

        self._x_max = { # positions of GLL node closest to x=1, p varying
            2: 0,
            3: 0.44721360,
            4: 0.65465367,
            5: 0.76505532,
            6: 0.83022390,
            7: 0.87174015,
            8: 0.8997579954,
            9: 0.9195393082
        }

    # Hooks - override these to drive the transport with something other than the Barnes
    # atmosphere (see prognostic_mms_checker.py)
    def _q_initial(self):
        return self._atmos.q_init()

    def _q_in(self): # keep q_in frozen for boundaries in the prognostic equation
        return self._atmos.q_init()

    def _source(self):
        return Constant(0.0)

    def _velocity(self):
        if not hasattr(self, "_diag_solver"): # built on first use, so overrides never pay for it
            self._diag_solver = DiagnosticSolver(self._atmos, self._matfree)
        psi = self._diag_solver.psi_soln
        return as_vector([-psi.dx(1), psi.dx(0), 0])

    def _update_velocity(self, q):
        self._diag_solver.update_q(q) # _build_solver went through _velocity, so this exists
        self._diag_solver.solve_psi()

    def _build_solver(self):
        V = self._dg_space
        u = self._velocity()
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

    @property
    def atmos(self):
        return self._atmos

    @property
    def psi(self):
        """Streamfunction of the state the velocity was last updated from."""
        return self._diag_solver.psi_soln

    def resolve(self): # An extra solve purely for diagnostic
        self._update_velocity(self.q)

    def dt(self, safety=SAFETY):
        dx = self._atmos.dxy_min # CFL is set by the smallest cell
        u = self._velocity()
        # Into the dg space, not the cg one: u is discontinuous, and interpolating it somewhere continuous would average the peaks
        vel = Function(self._dg_space).interpolate(sqrt(u[0]**2 + u[1]**2))
        max_vel = math_utils.get_global_max(vel)
        p = self._solver_params.polynomial_order

        scale = 0.5 * (1 - self._x_max[p])
        dx_eff = dx*scale

        return safety * dx_eff / max_vel

    def RHS(self, q, t=None) -> Function:
        self._t.assign(self.t if t is None else t)
        self._update_velocity(q)

        # Now solve the mass matrix
        self._temp_q.interpolate(q)
        self._prognostic_solver.solve()
        return self._stage_q.copy(deepcopy=True)

    def _limit(self, q):
        return self.limiter.apply(q) if self.limiter is not None else q

    def step(self, dt=None):
        """One SSPRK3 step (Shu & Osher 1988). Each stage is a convex combination of forward
        Euler steps, so any bound forward Euler keeps, the whole step keeps - which is what
        the limiter, applied after every stage, relies on. Classical RK4 has no such property.
        """
        q, t = self.q, self.t
        k = self.RHS(q, t) # also sets u, v for the current state, which dt() needs
        if dt is None:
            dt = self.dt()
        self._dt.assign(dt)
        h = self._dt
        q1 = self._limit(self._q1.assign(q + h * k))
        q2 = self._limit(self._q2.assign(0.75 * q + 0.25 * (q1 + h * self.RHS(q1, t + dt))))
        self._limit(q.assign(q / 3 + 2 / 3 * (q2 + h * self.RHS(q2, t + dt / 2))))
        self.t = t + dt
        return dt

import time
from firedrake import *
import math_utils
from atmosphere_builder import AtmosphereBuilder
from parameters import SolverParams

class Solver:
    def __init__(self, atmos : AtmosphereBuilder, solver_params : SolverParams, firedrake_params):
        self.atmos = atmos
        self.func_space = atmos.func_space
        self.solver_params = solver_params
        self.mesh = atmos.mesh
        self.phys_params = atmos.phys_params
        self.firedrake_params = firedrake_params
        self.phi = TestFunction(self.func_space)
        self.psi_soln = Function(self.func_space)  # Solution to system will be stored here

        self._setup_solver()

    def _specify_equation(self):
        psi = TrialFunction(self.func_space)
        phi = self.phi
        n = FacetNormal(self.mesh)

        f = self.phys_params.f

        top_boundary = self.atmos.top_boundary()
        bottom_boundary = self.atmos.bottom_boundary()

        rho_bar = self.atmos.rho_bar()
        u = self.atmos.u()
        v = self.atmos.v()
        N_bar = self.atmos.N_bar()
        q = self.atmos.q()
        rho_N2 = Function(self.func_space).interpolate(rho_bar / (N_bar**2))

        x_base = rho_bar * phi * v * n[0]
        y_base = rho_bar * phi * -u * n[1]
        z_base = n[2] * (f**2) * phi * rho_N2

        L_x = x_base * ds_v((1, 2))
        L_y = y_base * ds_v((3, 4))
        L_bottom = z_base * bottom_boundary * ds_b
        L_top = z_base * top_boundary * ds_t
        L_vol = -q * rho_bar * phi * dx

        # Linear form
        L = L_x + L_y + L_bottom + L_top + L_vol

        # Bilinear weak form
        a = (rho_bar * psi.dx(0) * phi.dx(0) +
             rho_bar * psi.dx(1) * phi.dx(1) +
             (f**2) * rho_N2 * psi.dx(2) * phi.dx(2)) * dx

        return a, L

    def _setup_solver(self):
        a, L = self._specify_equation()
        PETSc.Sys.Print("Set up equation")

        if self.solver_params.check_flux:
            PETSc.Sys.Print("Calculating net flux...")
            flux = assemble(replace(L, {self.phi: Constant(1)}))
            PETSc.Sys.Print(f"Net flux is {flux}")

        nullspace = VectorSpaceBasis(constant=True)

        problem = LinearVariationalProblem(a, L, self.psi_soln)
        self.solver = LinearVariationalSolver(
            problem,
            solver_parameters=self.firedrake_params,
            nullspace=nullspace,
        )
        PETSc.Sys.Print(f"Completed solver setup")

    def solve_psi(self):
        # Reset initial guess and clear cache
        math_utils._get_vertical_integral_solver.cache_clear()
        self.psi_soln.assign(0)

        start_time = time.perf_counter()
        self.solver.solve()
        solve_time = time.perf_counter()-start_time
        PETSc.Sys.Print(f"Solve completed in {solve_time:0.2f} sec")

        return solve_time
from firedrake import *
from atmosphere_builder import AtmosphereBuilder
from parameters import SolverParams


class Solver:
    def __init__(self, atmos : AtmosphereBuilder, solver_params : SolverParams):
        self.atmos = atmos
        self.func_space = atmos.func_space
        self.solver_params = solver_params
        self.mesh = atmos.mesh
        self.phys_params = atmos.phys_params
        self.phi = TestFunction(self.func_space)
        self.psi_soln = Function(self.func_space) # Solution to system will be stored here

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

        x_base = rho_bar * phi * v * n[0]
        y_base = rho_bar * phi * -u * n[1]
        z_base = n[2] * (f**2) * phi * rho_bar / (N_bar**2)

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
             (f**2) * rho_bar * psi.dx(2) * phi.dx(2) / (N_bar**2)) * dx

        return a, L

    def solve_psi(self):
        a, L = self._specify_equation()

        if self.solver_params.check_flux:
            PETSc.Sys.Print("Calculating net flux...")
            flux = assemble(replace(L, {self.phi : Constant(1)}))
            PETSc.Sys.Print(f"Net flux is {flux}")

        nullspace = VectorSpaceBasis(constant=True)

        firedrake_params = { #todo move these into actual parameters class
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-6,
        }

        PETSc.Sys.Print("Completed solver setup. Now solving...")
        solve(a == L, self.psi_soln, solver_parameters=firedrake_params, nullspace=nullspace)
        PETSc.Sys.Print("Solve completed")

        return self.psi_soln


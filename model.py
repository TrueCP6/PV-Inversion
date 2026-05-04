from firedrake import *
from config import DomainConfig, AnomalyConfig, SolverConfig
from atmosphere import AtmosphereBackground


class PVInversionModel:
    def __init__(self, mesh, function_space, atm: AtmosphereBackground, domain: DomainConfig, anom: AnomalyConfig, sol_cfg: SolverConfig):
        self.mesh = mesh
        self.atm = atm
        self.domain = domain
        self.anom = anom
        self.sol_cfg = sol_cfg

        self.V = function_space
        self.psi = TrialFunction(self.V)
        self.phi = TestFunction(self.V)
        self.n = FacetNormal(mesh)

        # Boundary wind components
        self.u = Function(self.V).interpolate(atm.u)
        self.v = Function(self.V).interpolate(atm.v)

        self.q = self._build_pv_field()
        self.psi_solution = Function(self.V, name="Streamfunction")

    def _build_pv_field(self):
        relative_vorticity = self.v.dx(0) - self.u.dx(1)
        q_bg = relative_vorticity + self.atm.f + self.atm.f * (self.atm.thetaBar / self.atm.thetaBar.dx(2)).dx(2)

        # Debug PV at a specific point
        if self.sol_cfg.evaluate_q_debug:
            q_func = project(q_bg, self.V, name="Evaluated_q")
            test_point = [self.domain.x_max / 2.0, self.domain.y_max / 2.0, 11285]
            q_value = q_func(test_point)
            PETSc.Sys.Print(f"Background q evaluated at {test_point} is: {q_value * 1e6}")

        # PV Anomaly field
        ANO_exponent = -((self.atm.z - self.anom.z_pos) / self.anom.z_size) ** 2 \
                       - ((self.atm.x - self.anom.x_pos) / self.anom.x_size) ** 2 \
                       - ((self.atm.y - self.anom.y_pos) / self.anom.y_size) ** 2
        ANO = min_value(-1.5, -4 * exp(ANO_exponent))

        return q_bg + ANO * 1e-6

    def _calculate_theta_star(self):
        PETSc.Sys.Print("Computing thetaStar for zero net flux...")

        top = [0, 0, self.domain.z_top]
        bot = [0, 0, 0]

        Nbar_bot = self.atm.Nbar(bot)
        Nbar_top = self.atm.Nbar(top)
        rho_bot = self.atm.rho(bot)
        rho_top = self.atm.rho(top)
        theta_bot = self.atm.thetaBar(bot)
        theta_top = self.atm.thetaBar(top)

        numerator = assemble(self.atm.rho * self.q * dx)
        denominator = self.domain.x_max * self.domain.y_max * self.atm.f * self.atm.g * \
                      (-rho_bot / (Nbar_bot ** 2 * theta_bot) + rho_top / (Nbar_top ** 2 * theta_top))

        thetaStar = float(numerator / denominator)
        PETSc.Sys.Print(f"thetaStar = {thetaStar}")
        return thetaStar

    def solve(self):
        thetaStar = self._calculate_theta_star()
        top_boundary = (self.atm.g * thetaStar) / (self.atm.f * self.atm.thetaBar)
        bottom_boundary = top_boundary

        # Bilinear weak form
        a = (self.atm.rho * self.psi.dx(0) * self.phi.dx(0) +
             self.atm.rho * self.psi.dx(1) * self.phi.dx(1) +
             (self.atm.f**2) * self.atm.rho * self.psi.dx(2) * self.phi.dx(2) / (self.atm.Nbar ** 2)) * dx

        # Linear Form components
        x_base = self.atm.rho * self.phi * self.v * self.n[0]
        y_base = self.atm.rho * self.phi * -self.u * self.n[1]
        z_base = self.n[2] * (self.atm.f**2) * self.phi * self.atm.rho / (self.atm.Nbar ** 2)

        L_x = x_base * ds_v((1, 2))
        L_y = y_base * ds_v((3, 4))
        L_z_bottom = (z_base * bottom_boundary) * ds_b
        L_z_top = (z_base * top_boundary) * ds_t
        L_vol = -self.q * self.atm.rho * self.phi * dx

        L = L_x + L_y + L_z_bottom + L_z_top + L_vol

        if self.sol_cfg.check_flux:
            L_scalar = replace(L, {self.phi: Constant(1.0)})
            net_flux = assemble(L_scalar)
            PETSc.Sys.Print(f"Compatibility condition check. Net flux is: {net_flux}")

        nullspace = VectorSpaceBasis(constant=True)
        solver_params = {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-6,
        }

        PETSc.Sys.Print('Solving weak form...')
        solve(a == L, self.psi_solution, solver_parameters=solver_params, nullspace=nullspace)

    def save_output(self):
        PETSc.Sys.Print('Calculating wind components...')
        x_vel = -self.psi_solution.dx(1)
        y_vel = self.psi_solution.dx(0)
        speed = sqrt(x_vel ** 2 + y_vel ** 2)

        horizontal_wind_speed = Function(self.V, name="Horizontal wind speed").interpolate(speed)
        pv = Function(self.V, name="Potential Vorticity").interpolate(self.q)

        outfile = VTKFile(self.sol_cfg.output_filename)
        outfile.write(self.psi_solution, horizontal_wind_speed, pv)
        PETSc.Sys.Print(f"Output saved to {self.sol_cfg.output_filename}")
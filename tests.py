from hash_seed import use_same_hash
use_same_hash()
import gc
import unittest
import numpy as np
from domain_builder import *
from math_utils import *
from parameters import *
from diagnostic_solver import *
from mms_checker import *
from barnes_atmosphere import *
from derived_quantities import *
from prognostic_solver import *
from prognostic_mms_checker import *
import tropopause
import petsctools

class UtilTests(unittest.TestCase):
    def test_resolutions_for_dofs(self):
        """The sweep floor scales with the ranks the job was launched under, and the
        resolutions handed back bracket the range asked for."""
        import sweep
        from mpi4py import MPI

        self.assertEqual(sweep.min_dofs(), sweep.MIN_DOFS_PER_RANK * MPI.COMM_WORLD.size)

        max_dofs = 100 * sweep.min_dofs()
        Ns = sweep.resolutions_for_dofs(max_dofs, 8, p=4)

        self.assertTrue(all(a < b for a, b in zip(Ns, Ns[1:])))  # unique and rising
        # Rounding to a whole N moves an end point by less than one element either way
        self.assertGreater(sweep.dof_count(4, int(Ns[0])), sweep.min_dofs() / 2)
        self.assertLess(sweep.dof_count(4, int(Ns[-1])), 2 * max_dofs)

        # A max below the floor would otherwise hand back a descending range, and with it
        # meshes too coarse to solve on at all
        with self.assertRaises(ValueError):
            sweep.resolutions_for_dofs(sweep.min_dofs() - 1, 3, p=4)

    def test_optimised_params_rebuild_physical_params(self):
        """The dict written out by the optimise entry point has to be usable as
        PhysicalParams(**dict), so every key must be a real field and land in bounds."""
        from param_sampler import ParamSampler

        sampler = ParamSampler(optimise_for="pres")
        params = sampler.normalised_to_dict(sampler.normalised_control)

        rebuilt = PhysicalParams(**params)
        for name, (low, high), _ in sampler.param_tuple:
            self.assertAlmostEqual(getattr(rebuilt, name), getattr(PhysicalParams(), name))
            self.assertTrue(low <= params[name] <= high)

    def test_cost_penalises_a_diverged_solve_instead_of_raising(self):
        """A single point the solver cannot converge on used to abort the whole optimise job."""
        from param_sampler import ParamSampler
        from firedrake.exceptions import ConvergenceError

        sampler = ParamSampler(optimise_for="pres")

        def blow_up(_params):
            raise ConvergenceError("DIVERGED_LINEAR_SOLVE")
        sampler.variator.get_derived = blow_up

        self.assertEqual(sampler.cost(sampler.normalised_control), 0.0)
        self.assertIsNone(sampler.all_data(sampler.normalised_control))
        self.assertEqual(sampler.random_sample_data(3)["min_surf_pres"], [])

    def test_vertical_integral(self):
        mesh2d = UnitSquareMesh(10, 10, quadrilateral=True)
        mesh = ExtrudedMesh(mesh2d, layers=10, layer_height=0.1)

        # Define the Function space
        V = FunctionSpace(mesh, "Q", 4)
        x,y,z = SpatialCoordinate(mesh)

        test_cases = [ # Define test cases with known analytical solutions
            (z, (z**2) / 2),
            (cos(z), sin(z)),
            (exp(z), exp(z)-1),
            (Constant(1), z)
        ]

        for integrand, exact in test_cases:
            num_soln = compute_vertical_integral(integrand, V)
            error = errornorm(exact, num_soln)
            PETSc.Sys.Print(f"Vertical integration error: {error}")
            self.assertLess(error, 1e-6)

    def _cg1_space(self, n=4):
        mesh = UnitSquareMesh(n, n)
        return FunctionSpace(mesh, "CG", 1)

    def test_relative_error_zero_for_identical_functions(self):
        V = self._cg1_space()
        x, y = SpatialCoordinate(V.mesh())
        f = Function(V).interpolate(x ** 2 + y)
        error = math_utils.relative_error(f, f.copy(deepcopy=True))
        self.assertLess(error, 1e-10)

    def test_relative_error_ignores_constant_offset(self):
        # solver only determines psi up to an additive constant - an arbitrary shift between numerical and exact should therefore leave the error at zero.
        V = self._cg1_space()
        x, y = SpatialCoordinate(V.mesh())
        exact = Function(V).interpolate(x ** 2 + y)
        numerical = Function(V).interpolate(x ** 2 + y + 7.3)
        error = math_utils.relative_error(exact, numerical)
        self.assertLess(error, 1e-10)

    def test_relative_error_scales_linearly_with_numerical(self):
        # With numerical = c * exact, the mean-offset removal collapses the ratio to exactly |1 - c| for any function, norm type or mesh - a closed-form check of the actual arithmetic (offset removal, errornorm, norm) inside relative_error.
        V = self._cg1_space()
        x, y = SpatialCoordinate(V.mesh())
        exact = Function(V).interpolate(x)

        for norm_type in ['L2', 'H1']:
            for c in [2.0, 0.5, -1.0]:
                numerical = Function(V).interpolate(c * x)
                error = math_utils.relative_error(exact, numerical, norm_type=norm_type)
                self.assertAlmostEqual(error, abs(1 - c), places=6)

    def test_relative_error_accepts_ufl_expression_for_exact(self):
        V = self._cg1_space()
        x, y = SpatialCoordinate(V.mesh())
        numerical = Function(V).interpolate(2 * x)
        error = math_utils.relative_error(x, numerical)
        self.assertAlmostEqual(error, 1.0, places=6)

    def test_temperature_tracks_latitude_but_trop_height_does_not(self):
        """temperature_bottom is a read-only latitudinal mean plus the varied departure;
        trop_height is a plain field, since its range is limited by the numerics."""
        from parameters import temperature_bottom_mean

        # Tabulated latitudes come back exactly, and the variation is a pure offset.
        self.assertAlmostEqual(PhysicalParams(latitude=-50).temperature_bottom, 279.20)
        self.assertAlmostEqual(
            PhysicalParams(latitude=-30, temperature_bottom_variation=5).temperature_bottom,
            temperature_bottom_mean(-30) + 5)

        # Interpolated between table points, and clamped outside them.
        self.assertAlmostEqual(temperature_bottom_mean(-42.5), 0.5 * (282.38 + 285.80))
        self.assertAlmostEqual(temperature_bottom_mean(-80), 279.20)
        self.assertAlmostEqual(temperature_bottom_mean(0), 296.47)

        # Read-only: the derived value cannot be set directly.
        with self.assertRaises(AttributeError):
            PhysicalParams().temperature_bottom = 290

        # The tropopause, and the anomaly riding on it, stay put as latitude changes.
        self.assertEqual(PhysicalParams(latitude=-50).trop_height, PhysicalParams(latitude=-30).trop_height)
        self.assertAlmostEqual(
            PhysicalParams(latitude=-25, trop_height=11e3).anomaly_z_pos, 11e3)

    def test_N_strat_tracks_latitude(self):
        """Birner's stratospheric N steepens toward the tropics; trop_width does not vary."""
        from parameters import N_strat_mean

        # P28: N^2 rises from 4.5e-4 in the extratropics to 7.0e-4 at the tropical edge.
        self.assertAlmostEqual(N_strat_mean(-45) ** 2, 4.5e-4, places=6)
        self.assertAlmostEqual(N_strat_mean(-30) ** 2, 7.0e-4, places=6)
        self.assertGreater(N_strat_mean(-25), N_strat_mean(-50))

        # Variation is a pure offset, and N_strat is read-only.
        self.assertAlmostEqual(
            PhysicalParams(latitude=-30, N_strat_variation=0.001).N_strat,
            N_strat_mean(-30) + 0.001)
        with self.assertRaises(AttributeError):
            PhysicalParams().N_strat = 0.03

        # A variation that drives N_strat non-positive divides by zero downstream, so it
        # has to fail loudly rather than hand back a profile full of NaNs.
        with self.assertRaises(ValueError):
            PhysicalParams(N_strat_variation=-1).N_strat

        # The stratosphere must stay stabler than the troposphere, or there is no tropopause.
        self.assertGreater(PhysicalParams().N_strat, PhysicalParams().N_trop)

class MMSTests(unittest.TestCase):
    def test_mms(self):
        PETSc.Sys.Print("Testing solution lines up with MMS")
        N = 20
        solver_params = SolverParams(
            check_flux=False,
            nx=N, ny=N, nz=N
        )
        phys_params = PhysicalParams()

        domain = DomainBuilder(solver_params, phys_params)

        atmos = MMSChecker(domain)

        for save_memory in [True, False]:
            solver = DiagnosticSolver(atmos, save_memory)
            solver.solve_psi()
            error = atmos.calc_error(solver.psi_soln)
            self.assertLess(error, 1e-4)

class StabilityTests(unittest.TestCase):
    def test_peclet(self):
        PETSc.Sys.Print("Testing the peclet number is sufficiently small")
        solver_params = SolverParams(nx=8, ny=8, nz=1000)
        phys_params = PhysicalParams()

        domain = DomainBuilder(solver_params, phys_params)
        cg_space = domain.cg_space()
        atmos = BarnesAtmosphere(domain)

        N2 = atmos.N_bar() ** 2
        rho = atmos.rho_bar()
        expr = abs(ln(rho / N2).dx(2))
        fun = Function(cg_space).interpolate(expr)

        with fun.dat.vec_ro as v:
            global_max = v.max()[1]

        PETSc.Sys.Print(f"Max ratio between advection and diffusion coefficients: {global_max}")
        self.assertLess(global_max, 0.01)

class DerivedQuantityTests(unittest.TestCase):
    def test_psi_0(self):
        PETSc.Sys.Print("Testing psi_0 is the lateral boundary average of psi")
        solver_params = SolverParams(nx=6, ny=5, nz=4, polynomial_order=4)
        phys_params = PhysicalParams(Lx=3e6, Ly=2e6, H=15e3)
        Lx, Ly = phys_params.Lx, phys_params.Ly

        domain = DomainBuilder(solver_params, phys_params)
        cg_space = domain.cg_space()
        atmos = BarnesAtmosphere(domain)
        x, y, z = SpatialCoordinate(domain.mesh())

        # psi_0 depends on z alone, so it takes one value per z level and no more
        n_levels = solver_params.polynomial_order * solver_params.nz + 1

        test_cases = [ # Perimeters integrated by hand, and normalised by 2(Lx + Ly)
            ((x + y) * z, (Lx + Ly) * z / 2), # oint (x + y) dl = (Lx + Ly)^2
            (x * z**3, Lx * z**3 / 2), # oint x dl = Lx (Lx + Ly)
            (z * (1 + (x - Lx/2) * (y - Ly/2) / (Lx * Ly)), z), # an anomaly that averages away
        ]

        for psi_expr, exact in test_cases:
            psi = Function(cg_space).interpolate(psi_expr)
            psi_0 = ResolvedAtmosphere(psi, atmos)._psi_0()

            # Every case is a polynomial the function space holds exactly, so the only
            # error left is rounding - psi_0 itself introduces no quadrature error.
            error = math_utils.relative_error(exact, psi_0)
            PETSc.Sys.Print(f"Boundary average error: {error}")
            self.assertLess(error, 1e-12)
            self.assertEqual(np.unique(psi_0.dat.data_ro).size, n_levels)

class TropopauseTests(unittest.TestCase):
    """The dynamical tropopause is the 1.5 PVU contour bounding the stratosphere, so these
    check what distinguishes it from the lowest stratospheric dof anywhere in the domain:
    which side of the contour a given pocket of air is on.
    """
    CORIOLIS = -1e-4 # southern hemisphere, so stratospheric air is Q <= -1.5 PVU
    H = 10e3

    def _cg_space(self):
        phys_params = PhysicalParams(Lx=1e6, Ly=1e6, H=self.H)
        return DomainBuilder(SolverParams(nx=8, ny=8, nz=8, polynomial_order=4), phys_params).cg_space()

    def _height(self, V, pvu):
        """Tropopause height of a field written as a positive number of PVU, negated to put
        it in the hemisphere CORIOLIS belongs to."""
        pv = Function(V).interpolate(-tropopause.PVU * pvu)
        return tropopause.min_height(pv, self.CORIOLIS)

    def _level_spacing(self, V):
        return np.diff(tropopause.column_layout(V).z).max()

    def test_pocket_under_the_tropopause_is_not_the_tropopause(self):
        """A ball of stratospheric air sitting on its own in the middle troposphere. It is
        the lowest such air in the domain, and it is not the tropopause."""
        V = self._cg_space()
        x, y, z = SpatialCoordinate(V.mesh())
        trop, centre, radius = 5000, 2000, 800

        pocket = (x - 5e5)**2 + (y - 5e5)**2 + (z - centre)**2 < radius**2
        stratospheric = Or(z >= trop, pocket)
        height = self._height(V, conditional(stratospheric, 2.0, 0.5))

        # The pocket is resolved and is far lower - it is what a search over dofs returns
        lowest = get_global_min(Function(V).interpolate(conditional(stratospheric, z, 1e30)))
        PETSc.Sys.Print(f"Tropopause at {height} m, lowest stratospheric dof at {lowest} m")
        self.assertLess(lowest, centre)
        self.assertAlmostEqual(height, trop, delta=self._level_spacing(V))

    def test_fold_joined_to_the_stratosphere_does_count(self):
        """A tongue of stratospheric air sloping down and away from an otherwise flat
        tropopause. Tropospheric air lies above it, so a column by column search from the
        model top misses it, but it joins the stratosphere at the shallow end and the
        tropopause follows it all the way down."""
        V = self._cg_space()
        x, _, z = SpatialCoordinate(V.mesh())
        trop, descent, thickness = 5000, 3000, 1500

        tongue_top = trop - descent * x / 1e6
        tongue = And(z <= tongue_top, z >= tongue_top - thickness)
        height = self._height(V, conditional(Or(z >= trop, tongue), 2.0, 0.5))

        PETSc.Sys.Print(f"Tropopause follows the fold down to {height} m")
        self.assertAlmostEqual(height, trop - descent - thickness, delta=self._level_spacing(V))

    def test_columns_with_nothing_to_cross(self):
        """Air that is stratospheric everywhere puts the tropopause on the ground, and air
        that is stratospheric nowhere leaves it at the model top."""
        V = self._cg_space()

        self.assertEqual(self._height(V, Constant(2.0)), 0.0)
        self.assertEqual(self._height(V, Constant(0.5)), self.H)

class BasicStateTests(unittest.TestCase):
    def test_background_inverts_back_to_the_jet(self):
        """With the anomaly switched off, q is the QGPV of psi_bar by construction, so the
        inversion has to return psi_bar itself (up to the usual additive constant). Fails if
        the jet moves back into v, if the wind stops being non-divergent, or if q_bar drops
        the stretching term - the three ways the basic state can stop being self-consistent.
        """
        phys_params = PhysicalParams(anomaly_mag=0)
        solver_params = SolverParams()
        atmos = BarnesAtmosphere(DomainBuilder(solver_params, phys_params))

        solver = DiagnosticSolver(atmos, True)
        solver.solve_psi()

        error = math_utils.relative_error(atmos.psi_bar(), solver.psi_soln)
        PETSc.Sys.Print(f"Background inversion relative error: {error}")
        self.assertLess(error, 1e-3)

class SolverTests(unittest.TestCase):
    def _test_upd_atmos(self, matfree : bool):
        n = 30
        solver_params = SolverParams(nx=n, ny=n, nz=n)
        phys_params_1 = PhysicalParams(latitude=-45)

        domain = DomainBuilder(solver_params, phys_params_1)
        atmos_1 = BarnesAtmosphere(domain)

        solver = DiagnosticSolver(atmos_1, matfree)
        solver.solve_psi()
        psi_1 = solver.psi_soln.copy(deepcopy=True)

        phys_params_2 = PhysicalParams(latitude=-10)
        atmos_2 = BarnesAtmosphere(domain, phys_params_2)
        solver.update_atmosphere(atmos_2)

        solver.solve_psi()
        psi_2 = solver.psi_soln

        rel_error = math_utils.relative_error(psi_1, psi_2)
        PETSc.Sys.Print(f"Relative error between atmospheres: {rel_error}")
        self.assertGreater(rel_error, 0.1)

    def _test_step_atmos(self, matfree: bool):
        n = 30
        solver_params = SolverParams(nx=n, ny=n, nz=n)
        phys_params_1 = PhysicalParams()

        domain = DomainBuilder(solver_params, phys_params_1)
        atmos_1 = BarnesAtmosphere(domain)

        solver = DiagnosticSolver(atmos_1, matfree)
        solver.solve_psi()
        psi_1 = solver.psi_soln.copy(deepcopy=True)

        # Create new parameters, but only alter the anomaly, and only pass through the new anomaly to the step function
        phys_params_2 = PhysicalParams(anomaly_mag=-1e-6)
        atmos_2 = BarnesAtmosphere(domain, phys_params_2)
        solver.update_q(atmos_2.q_init())

        solver.solve_psi()
        psi_2 = solver.psi_soln

        rel_error = math_utils.relative_error(psi_1, psi_2)
        PETSc.Sys.Print(f"Relative error between anomalies: {rel_error}")
        self.assertGreater(rel_error, 0.05)

    def test_update_atmosphere(self):
        for matfree in [True, False]:
            for case in (self._test_upd_atmos, self._test_step_atmos):
                case(matfree)
                gc.collect() # Firedrake defers PETSc destroys in parallel, so free this N=30 solver before the next
                PETSc.garbage_cleanup(COMM_WORLD)

class PrognosticTests(unittest.TestCase):
    def test_rhs_is_advection(self):
        """The DG operator should approximate -u.grad(q) for a smooth q. Fails on a sign error,
        a wrong facet flux, or u and v swapped. The anomaly is kept below ertel_pv's -1.5e-6
        clip and the domain shrunk so it is resolved."""
        phys_params = PhysicalParams(Lx=2e6, Ly=2e6, jet_y_pos=1e6, anomaly_mag=-1e-6)
        solver = PrognosticSolver(SolverParams(nx=16, ny=16, nz=32), phys_params, True)
        rhs = solver.RHS(solver.q)
        u = solver._velocity() # the velocity the form actually transports with
        exact = -(u[0] * solver.q.dx(0) + u[1] * solver.q.dx(1))

        error = errornorm(exact, rhs) / norm(exact)
        PETSc.Sys.Print(f"RHS vs -u.grad(q) relative error: {error}")
        self.assertLess(error, 0.02)

    def test_background_is_steady(self):
        """Without the anomaly, q depends on y and z only and the jet blows along x, so a
        step must leave q (almost) unchanged. Fails if RK4 updates or the state buffer are wrong."""
        solver = PrognosticSolver(SolverParams(nx=10, ny=10, nz=10), PhysicalParams(anomaly_mag=0), True)
        q_0 = solver.q.copy(deepcopy=True)
        dt = solver.step()

        change = errornorm(q_0, solver.q) / norm(q_0)
        PETSc.Sys.Print(f"dt = {dt}, relative change in background q: {change}")
        self.assertGreater(dt, 0)
        self.assertLess(change, 1e-3)

class PrognosticMMSTests(unittest.TestCase):
    def test_convergence(self):
        PETSc.Sys.Print("Testing prognostic solver converges to the MMS solution")
        p = 4
        phys_params = PhysicalParams(Lx=1e6, Ly=1e6, H=20e3)
        ns = [4, 8, 16]

        errors = []
        for n in ns:
            checker = PrognosticMMSChecker(SolverParams(nx=n, ny=n, nz=1, polynomial_order=p), phys_params)
            errors.append(checker.run(T=2e4))
            PETSc.Sys.Print(f"n={n}: relative L2 error {errors[-1]}")

        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        PETSc.Sys.Print(f"Convergence rates: {rates}")
        # dt shrinks with h, so RK4's O(dt^4) sits under the O(h^(p+1)) spatial error
        self.assertGreater(rates[-1], p + 0.5)
        self.assertLess(errors[-1], 1e-3)

if __name__ == '__main__':
    unittest.main()
    petsctools.print_citations_at_exit()

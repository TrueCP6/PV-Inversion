import unittest
import numpy as np
from domain_builder import *
from math_utils import *
from parameters import *
from diagnostic_solver import *
from mms_checker import *
from barnes_atmosphere import *
from derived_quantities import *

class UtilTests(unittest.TestCase):
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

    def test_relative_error_cross_mesh(self):
        # Different-resolution meshes force relative_error onto its cross-mesh
        # interpolation path. x and c*x are linear, so they interpolate exactly onto
        # any CG1 mesh - the |1-c| identity above should therefore still hold exactly,
        # regardless of which side is finer or which mesh compare_on picks.
        V_coarse = self._cg1_space(2)
        V_fine = self._cg1_space(8)
        c = 3.0

        for exact_space, numerical_space in [(V_fine, V_coarse), (V_coarse, V_fine)]:
            x_e, _ = SpatialCoordinate(exact_space.mesh())
            x_n, _ = SpatialCoordinate(numerical_space.mesh())
            exact = Function(exact_space).interpolate(x_e)
            numerical = Function(numerical_space).interpolate(c * x_n)

            for compare_on in ['fine', 'coarse']:
                error = math_utils.relative_error(exact, numerical, compare_on=compare_on)
                self.assertAlmostEqual(error, abs(1 - c), places=6)

class MMSTests(unittest.TestCase):
    def test_mms(self):
        PETSc.Sys.Print("Testing solution lines up with MMS")
        N = 20
        solver_params = SolverParams(
            check_flux=False,
            nx=N, ny=N, nz=N
        )
        phys_params = PhysicalParams(
            Lx=1e6, Ly=1e6, H = 20e3
        )

        domain = DomainBuilder(solver_params, phys_params)

        atmos = MMSChecker(domain)

        for save_memory in [True, False]:
            solver = DiagnosticSolver(atmos, save_memory)
            solver.solve_psi()
            error = atmos.calc_error(solver.psi_soln)
            self.assertLess(error, 1e-6)

class StabilityTests(unittest.TestCase):
    def test_peclet(self):
        PETSc.Sys.Print("Testing the peclet number is sufficiently small")
        solver_params = SolverParams(nx=8, ny=8, nz=10000)
        phys_params = PhysicalParams()

        domain = DomainBuilder(solver_params, phys_params)
        func_space = domain.func_space()
        atmos = BarnesAtmosphere(domain)

        N2 = atmos.N_bar() ** 2
        rho = atmos.rho_bar()
        expr = abs(ln(rho / N2).dx(2))
        fun = Function(func_space).interpolate(expr)

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
        func_space = domain.func_space()
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
            psi = Function(func_space).interpolate(psi_expr)
            psi_0 = ResolvedAtmosphere(psi, atmos)._psi_0()

            # Every case is a polynomial the function space holds exactly, so the only
            # error left is rounding - psi_0 itself introduces no quadrature error.
            error = math_utils.relative_error(exact, psi_0)
            PETSc.Sys.Print(f"Boundary average error: {error}")
            self.assertLess(error, 1e-12)
            self.assertEqual(np.unique(psi_0.dat.data_ro).size, n_levels)

class SolverTests(unittest.TestCase):
    def _test_upd_atmos(self, matfree : bool):
        solver_params = SolverParams()
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
        PETSc.Sys.Print(f"Relative error between solutions: {rel_error}")
        self.assertGreater(rel_error, 1)

    def test_update_atmosphere(self):
        self._test_upd_atmos(False)
        self._test_upd_atmos(True)

if __name__ == '__main__':
    unittest.main()

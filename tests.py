import unittest
from domain_builder import *
from math_utils import *
from parameters import *
from solver import *
from mms_checker import *
from barnes_atmosphere import *

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
            solver = Solver(atmos, save_memory)
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

if __name__ == '__main__':
    unittest.main()

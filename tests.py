import unittest
from domain_builder import *
from math_utils import *
from parameters import *
from solver import *
from mms_checker import *

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
            self.assertLess(error, 1e-6)

class MMSTests(unittest.TestCase):
    def test_mms(self):
        N = 20
        solver_params = SolverParams(
            check_flux=False,
            nx=N, ny=N, nz=N
        )
        phys_params = PhysicalParams(
            Lx=1e6, Ly=1e6, H = 20e3
        )

        domain_builder = DomainBuilder(solver_params, phys_params)
        mesh = domain_builder.mesh()
        V = domain_builder.func_space()

        atmos = MMSChecker(mesh, V, phys_params)

        for save_memory in [True, False]:
            solver = Solver(atmos, solver_params, save_memory)
            solver.solve_psi()
            error = atmos.calc_error(solver.psi_soln)
            self.assertLess(error, 1e-6)


if __name__ == '__main__':
    unittest.main()

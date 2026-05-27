import unittest
from math_utils import *

class MyTestCase(unittest.TestCase):
    def test_verical_integral_linear_z(self):
        mesh2d = UnitSquareMesh(10, 10)
        mesh = ExtrudedMesh(mesh2d, layers=10, layer_height=0.1)

        # Define the Continuous Galerkin space
        V = FunctionSpace(mesh, "CG", 2)

        x,y,z = SpatialCoordinate(mesh)

        integrand = z
        exact_soln = (z**2) / 2
        num_soln = compute_vertical_integral(integrand, V)

        error = errornorm(exact_soln, num_soln)

        self.assertLess(error, 1e-6)


if __name__ == '__main__':
    unittest.main()

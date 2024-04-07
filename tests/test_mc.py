import unittest
import numpy as np
from montpy.mc import MonteCarloSolver, ImportanceSampler

class TestMonteCarloSolver(unittest.TestCase):
    def test_single_variable_integration(self):
        def f(samples):
            x = samples.T
            return x**2

        domain = [(0, 1)]

        solver = ImportanceSampler(f, domain)
        integral = solver.integrate(num_samples=1000)

        self.assertAlmostEqual(integral, 1/3, delta=0.1)

    def test_multi_variable_integration(self):
        def f(samples):
            x, y = samples.T
            return x**2 + y**2

        domain = [(0, 1), (0, 1)]

        solver = ImportanceSampler(f, domain)
        integral = solver.integrate(num_samples=1000)

        self.assertAlmostEqual(integral, 2/3, delta=0.1)

class TestImportanceSampler(unittest.TestCase):
    def test_uniform_distribution(self):
        def f(x):
            return x**2

        domain = [(0, 1)]

        sampler = ImportanceSampler(f, domain, pdf="uniform")
        integral = sampler.integrate(num_samples=1000)

        self.assertAlmostEqual(integral, 1/3, delta=0.1)

    def test_gaussian_distribution(self):
        def f(samples):
            x, y = samples.T
            return x**2 + y**2

        domain = [(0, 1), (0, 1)]

        sampler = ImportanceSampler(f, domain, pdf="gaussian")
        integral = sampler.integrate(num_samples=1000)

        self.assertAlmostEqual(integral, 2/3, delta=0.3)

if __name__ == "__main__":
    unittest.main()

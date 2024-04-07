import unittest
from montpy.qmc import QuasiMonteCarloSolver

class TestQuasiMonteCarloSolver(unittest.TestCase):
    def setUp(self):
        # Define a multivariate function
        def f_multivariate(x, y):
            return x**2 + y**2
        
        # Define a univariate function
        def f_univariate(x):
            return x**2
        
        # Define the domain for multivariate functions
        self.domain_multivariate = [(0, 1), (0, 1)]
        
        # Define the domain for univariate functions
        self.domain_univariate = [(0, 1)]
        
        # Create a Quasi-Monte Carlo solver for multivariate functions
        self.solver_multivariate = QuasiMonteCarloSolver(f_multivariate, self.domain_multivariate)
        
        # Create a Quasi-Monte Carlo solver for univariate functions
        self.solver_univariate = QuasiMonteCarloSolver(f_univariate, self.domain_univariate)
    
    def test_integrate_sobol_multivariate(self):
        integral = self.solver_multivariate.integrate_sobol(num_samples=1000)
        self.assertAlmostEqual(integral, 2/3, delta=0.1)

    def test_integrate_halton_multivariate(self):
        integral = self.solver_multivariate.integrate_halton(num_samples=1000)
        self.assertAlmostEqual(integral, 2/3, delta=0.1)

    def test_integrate_hammersley_multivariate(self):
        integral = self.solver_multivariate.integrate_hammersley(num_samples=1000)
        self.assertAlmostEqual(integral, 2/3, delta=0.1)

    def test_integrate_sobol_univariate(self):
        integral = self.solver_univariate.integrate_sobol(num_samples=1000)
        self.assertAlmostEqual(integral, 1/3, delta=0.1)

    def test_integrate_halton_univariate(self):
        integral = self.solver_univariate.integrate_halton(num_samples=1000)
        self.assertAlmostEqual(integral, 1/3, delta=0.1)

    def test_integrate_hammersley_univariate(self):
        integral = self.solver_univariate.integrate_hammersley(num_samples=1000)
        self.assertAlmostEqual(integral, 1/3, delta=0.1)

if __name__ == "__main__":
    unittest.main()

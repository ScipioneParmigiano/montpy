# Montpy: Montecarlo and Quasi-Montecarlo Integration Library
<div style="text-align:center">
    <a href="https://pypi.org/project/montpy/">
        <img src="https://img.shields.io/pypi/dm/montpy.svg" alt="Monthly Downloads">
    </a>
</div>


This Python library provides implementations of Monte Carlo and Quasi-Monte Carlo methods for evaluating integrals. Monte Carlo methods are stochastic techniques based on random sampling, while Quasi-Monte Carlo methods use low-discrepancy sequences for sampling. These methods are particularly useful for high-dimensional integrals where other numerical methods may struggle.

## Installation
You can install the library via pip:
```{bash}
pip install montpy
```

## Usage
```{Python}
from montpy.mc import MonteCarloSolver, ImportanceSampler
from montpy.qmc import QuasiMonteCarloSolver

# Define the function to integrate
def f(samples):
    x, y = samples.T
    return x**2 + y**2
    
# Define the domain (multivariate case)
domain = [(0, 1), (0, 1)]

# Create the Monte Carlo solver (uniform distribution)
solver_mc = ImportanceSampler(f, domain)
integral_mc = solver_mc.integrate(num_samples=1000)
print("Estimated integral with Monte Carlo:", integral_mc)

# Define the function to integrate
def g(x, y):
    return x*y
    
# Create the Quasi-Monte Carlo solver
solver_qmc = QuasiMonteCarloSolver(g, domain)

# Perform integration using Sobol sequence
integral_sobol = solver_qmc.integrate_sobol(num_samples=1000)
print("Estimated integral with Sobol sequence:", integral_sobol)
```

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

## License
This library is licensed under the MIT License. See the [LICENSE](https://github.com/ScipioneParmigiano/montpy/blob/main/LICENSE) file for details.
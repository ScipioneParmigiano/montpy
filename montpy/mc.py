import numpy as np

class MonteCarloSolver:
    def __init__(self, func, domain):
        """
        Initialize the Monte Carlo solver with a function and domain.
        
        Parameters:
            func (callable): The function to integrate.
            domain (tuple or list of tuples): The integration domain. 
                For single-variable functions, domain should be a tuple (a, b).
                For multi-variable functions, domain should be a list of tuples [(a1, b1), (a2, b2), ...].
        """
        self.func = func
        self.domain = domain
    
    def integrate(self, num_samples=1000):
        """
        Perform Monte Carlo integration.
        
        Parameters:
            num_samples (int): The number of random samples to generate.
        
        Returns:
            float: The estimated integral value.
        """
        samples = self._sample_from_distribution(num_samples)
        
        if isinstance(self.domain, tuple):
            integral = np.mean(self.func(*samples.T)) * (self.domain[1] - self.domain[0])
        elif isinstance(self.domain, list):
            bounds = np.array(self.domain)
            integral = np.mean(self.func(samples)) * np.prod(bounds[:, 1] - bounds[:, 0])
        
        return integral
    
    def _sample_from_distribution(self, num_samples):
        raise NotImplementedError("Subclasses must implement _sample_from_distribution method.")

class ImportanceSampler(MonteCarloSolver):
    def __init__(self, func, domain, pdf="uniform", custom_pdf=None):
        """
        Initialize the importance sampler with a function, domain, and probability distribution.
        
        Parameters:
            func (callable): The function to integrate.
            domain (tuple or list of tuples): The integration domain. 
                For single-variable functions, domain should be a tuple (a, b).
                For multi-variable functions, domain should be a list of tuples [(a1, b1), (a2, b2), ...].
            pdf (str): Probability distribution to use. Options are "uniform", "gaussian", "exponential", or "custom".
            custom_pdf (callable): Custom probability density function provided by the user. Required if pdf="custom".
        """
        super().__init__(func, domain)
        self.pdf = pdf
        self.custom_pdf = custom_pdf
    
    def _sample_from_distribution(self, num_samples):
        if self.pdf == "uniform":
            if isinstance(self.domain, tuple):
                a, b = self.domain
                samples = np.random.uniform(a, b, num_samples)
            elif isinstance(self.domain, list):
                bounds = np.array(self.domain)
                num_dims = len(bounds)
                samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, num_dims))
        elif self.pdf == "gaussian":
            if isinstance(self.domain, tuple):
                mu, sigma = (self.domain[0] + self.domain[1]) / 2, (self.domain[1] - self.domain[0]) / 6
                samples = np.random.normal(mu, sigma, num_samples)
            elif isinstance(self.domain, list):
                means = [(low + high) / 2 for low, high in self.domain]
                cov = np.diag([(high - low) / 6 for low, high in self.domain])
                samples = np.random.multivariate_normal(means, cov, num_samples)
        elif self.pdf == "custom":
            if self.custom_pdf is None:
                raise ValueError("Custom PDF must be provided for 'custom' distribution.")
            samples = self.custom_pdf(num_samples)
        else:
            raise ValueError("Invalid PDF.")
        
        return samples


